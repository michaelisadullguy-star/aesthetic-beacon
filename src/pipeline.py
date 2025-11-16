"""High-level algorithms orchestrating the aesthetic feedback pipeline.

This module implements the six stages outlined in the project description. Each
stage focuses on a distinct learning problem and is therefore exposed as an
independent function or helper class. The goal is not to prescribe a single
monolithic training script but to provide reusable building blocks that can be
assembled in notebooks, command line tools, or real-time feedback systems.

The steps are summarised as follows:

1. Convert hexadecimal EEG dumps into 8×3200 tensors ready for model
   consumption.
2. Learn a *to viewer* model that predicts aesthetic enjoyment from paired EEG
   and audio segments that participants labelled as enjoyable.
3. Evaluate long-form EEG recordings with the viewer model to locate passages
   that generated the highest enjoyment while reading an article.
4. Train a *to creator* model using the creator's own EEG while authoring the
   article, focusing on the brain states observed during the production of
   "excellent" passages.
5. Learn a mapping from EEG segments to Processing parameters derived from
   colour and texture descriptors extracted from reference imagery.
6. Combine the creator and Processing models inside a feedback loop that drives
   a live Processing sketch when the creator is in a *creatable* brainwave
   state.

Although the functions are designed to interoperate, they intentionally avoid
hard-coding dependencies between stages so that each can be deployed
individually in different research workflows.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from .brainwave_dataset import (
    BrainwaveProcessingDataset,
    collate_pad_batch,
    z_score_normalise,
)
from .model import BrainwaveToProcessingModel, TemporalFeatureEncoder, create_model

# ---------------------------------------------------------------------------
# Step 1 – Hexadecimal EEG conversion
# ---------------------------------------------------------------------------


def _clean_hex_stream(hex_stream: str | bytes) -> str:
    """Return a hex string stripped of whitespace and separators."""

    if isinstance(hex_stream, bytes):
        hex_stream = hex_stream.decode("ascii")

    filtered = [ch for ch in hex_stream if ch in "0123456789abcdefABCDEF"]
    if not filtered:
        raise ValueError("No hexadecimal characters found in the provided stream.")
    return "".join(filtered)


def decode_hex_eeg_segment(
    hex_stream: str | bytes,
    *,
    channels: int = 8,
    sample_rate: int = 640,
    segment_seconds: int = 5,
    dtype: np.dtype = np.int16,
) -> np.ndarray:
    """Convert a hexadecimal EEG dump into an ``(channels, samples)`` array.

    The EEG recorder stores little-endian integer amplitudes in hexadecimal text
    form. ``dtype`` controls the integer interpretation; by default a 16-bit
    signed representation is assumed. The resulting floating-point array is
    normalised to ``[-1, 1]`` by dividing through the maximum representable
    value of the dtype. Any trailing samples that do not fill a complete time
    step across all channels are discarded.
    """

    cleaned = _clean_hex_stream(hex_stream)
    raw_bytes = bytes.fromhex(cleaned)
    if len(raw_bytes) < np.dtype(dtype).itemsize:
        raise ValueError("The hex stream is too short to contain a single sample.")

    integer_values = np.frombuffer(raw_bytes, dtype=dtype)
    if integer_values.size % channels != 0:
        integer_values = integer_values[: integer_values.size - (integer_values.size % channels)]

    samples_per_channel = integer_values.size // channels
    expected_samples = sample_rate * segment_seconds
    if samples_per_channel < expected_samples:
        raise ValueError(
            f"Segment too short: expected at least {expected_samples} samples per channel, "
            f"got {samples_per_channel}."
        )

    matrix = integer_values[: channels * expected_samples].reshape(expected_samples, channels).T

    info = np.iinfo(dtype)
    max_value = float(max(abs(info.min), abs(info.max)))
    matrix = matrix.astype(np.float32) / max_value
    return matrix


def iter_brainwave_segments(
    hex_stream: str | bytes,
    *,
    channels: int = 8,
    sample_rate: int = 640,
    segment_seconds: int = 5,
    dtype: np.dtype = np.int16,
) -> Iterator[np.ndarray]:
    """Yield consecutive 5-second EEG segments decoded from a hex stream."""

    cleaned = _clean_hex_stream(hex_stream)
    raw_bytes = bytes.fromhex(cleaned)
    integer_values = np.frombuffer(raw_bytes, dtype=dtype)
    if integer_values.size % channels != 0:
        integer_values = integer_values[: integer_values.size - (integer_values.size % channels)]

    samples_per_channel = integer_values.size // channels
    segment_length = sample_rate * segment_seconds

    if samples_per_channel < segment_length:
        return

    info = np.iinfo(dtype)
    max_value = float(max(abs(info.min), abs(info.max)))
    matrix = integer_values.reshape(samples_per_channel, channels).T.astype(np.float32) / max_value

    for start in range(0, samples_per_channel - segment_length + 1, segment_length):
        yield matrix[:, start : start + segment_length]


# ---------------------------------------------------------------------------
# Step 2 – To viewer model
# ---------------------------------------------------------------------------


def _mel_filter_bank(num_mels: int, n_fft: int, sample_rate: int) -> Tensor:
    """Create a mel filter bank compatible with :func:`torch.stft` output."""

    def hz_to_mel(freq: np.ndarray) -> np.ndarray:
        return 2595.0 * np.log10(1.0 + freq / 700.0)

    def mel_to_hz(mel: np.ndarray) -> np.ndarray:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    fft_bins = n_fft // 2 + 1
    mel_min = hz_to_mel(np.array([0.0]))[0]
    mel_max = hz_to_mel(np.array([sample_rate / 2]))[0]
    mel_points = np.linspace(mel_min, mel_max, num_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_indices = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    filters = np.zeros((num_mels, fft_bins), dtype=np.float32)
    for m in range(1, num_mels + 1):
        left, centre, right = bin_indices[m - 1 : m + 2]
        if centre == left:
            centre += 1
        if right == centre:
            right += 1
        for k in range(left, centre):
            filters[m - 1, k] = (k - left) / max(centre - left, 1)
        for k in range(centre, right):
            filters[m - 1, k] = (right - k) / max(right - centre, 1)

    filters /= np.maximum(filters.sum(axis=1, keepdims=True), 1e-8)
    return torch.from_numpy(filters)


def compute_log_mel_spectrogram(
    waveform: Tensor | np.ndarray,
    *,
    sample_rate: int,
    n_fft: int = 1024,
    hop_length: int = 320,
    n_mels: int = 64,
) -> Tensor:
    """Compute a log-mel spectrogram for an audio snippet."""

    if isinstance(waveform, np.ndarray):
        waveform = torch.from_numpy(waveform)
    waveform = waveform.to(torch.float32)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    window = torch.hann_window(n_fft, device=waveform.device)
    stft = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        return_complex=True,
    )
    magnitude = stft.abs() ** 2
    mel_basis = _mel_filter_bank(n_mels, n_fft, sample_rate).to(magnitude.device, magnitude.dtype)
    mel_spec = torch.matmul(mel_basis, magnitude)
    log_mel = torch.log(mel_spec.clamp_min(1e-10))
    if log_mel.dim() == 3 and log_mel.size(0) == 1:
        log_mel = log_mel.squeeze(0)
    return log_mel


@dataclass(frozen=True)
class ViewerSample:
    """Container bundling EEG, audio paths, and enjoyment label for dataset construction."""

    brainwave_path: Path | str
    audio_path: Path | str
    enjoyed: bool


class ViewerEnjoymentDataset(Dataset[Tuple[Tensor, Tensor, Tensor]]):
    """Dataset providing paired EEG/audio segments with enjoyment labels."""

    def __init__(
        self,
        samples: Sequence[ViewerSample],
        *,
        audio_sample_rate: int = 44100,
        n_fft: int = 1024,
        hop_length: int = 320,
        n_mels: int = 64,
    ) -> None:
        if not samples:
            raise ValueError("ViewerEnjoymentDataset requires at least one sample.")
        self.samples = list(samples)
        self.audio_sample_rate = audio_sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

        # Load first sample to get dimensions
        first_eeg = np.load(self.samples[0].brainwave_path)
        first_audio = np.load(self.samples[0].audio_path)
        first_spec = compute_log_mel_spectrogram(
            first_audio, sample_rate=audio_sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )

        self.eeg_channels = first_eeg.shape[0]
        self.eeg_length = first_eeg.shape[1]
        self.audio_mels = first_spec.shape[0]
        self.audio_frames = first_spec.shape[1]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor]:
        sample = self.samples[index]

        # Load EEG
        eeg = np.load(sample.brainwave_path)
        eeg = torch.from_numpy(eeg).to(torch.float32)

        # Load and process audio
        audio_waveform = np.load(sample.audio_path)
        audio_spec = compute_log_mel_spectrogram(
            audio_waveform,
            sample_rate=self.audio_sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )

        # Create label
        label = torch.tensor([1.0 if sample.enjoyed else 0.0], dtype=torch.float32)

        return eeg, audio_spec, label


class AudioEncoder(nn.Module):
    """CNN that embeds log-mel spectrograms into a fixed-size vector."""

    def __init__(self, input_channels: int, hidden_channels: int = 32) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.output_dim = hidden_channels

    def forward(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(1)  # (batch, 1, mels, frames)
        x = self.network(x)
        x = self.pool(x)
        return x.view(x.size(0), -1)


class ViewerEnjoymentModel(nn.Module):
    """Bi-modal network predicting enjoyment from EEG and audio segments."""

    def __init__(
        self,
        eeg_channels: int = 8,
        eeg_length: int = 3200,
        audio_mels: int = 64,
        audio_frames: int = 688,
        hidden_channels: int = 128,
    ) -> None:
        super().__init__()
        self.eeg_channels = eeg_channels
        self.eeg_length = eeg_length
        self.audio_mels = audio_mels
        self.audio_frames = audio_frames
        self.eeg_encoder = TemporalFeatureEncoder(
            input_channels=eeg_channels,
            hidden_channels=hidden_channels,
            num_blocks=3,
        )
        self.audio_encoder = AudioEncoder(input_channels=audio_mels)
        fusion_dim = hidden_channels + self.audio_encoder.output_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_dim, 1),
        )

    def forward(self, eeg: Tensor, audio: Tensor) -> Tensor:
        eeg_features = self.eeg_encoder(eeg)
        audio_features = self.audio_encoder(audio)
        fused = torch.cat([eeg_features, audio_features], dim=1)
        return self.classifier(fused)


@dataclass
class TrainingHistory:
    losses: List[float]
    val_losses: List[float]


def _prepare_eeg_batch(eeg: Tensor) -> Tensor:
    if eeg.dim() == 2:
        return z_score_normalise(eeg)
    if eeg.dim() != 3:
        raise ValueError("EEG tensors must have shape (batch, channels, samples)")
    normalised = [z_score_normalise(sample) for sample in eeg]
    return torch.stack(normalised, dim=0)


def train_viewer_model(
    dataset: ViewerEnjoymentDataset,
    *,
    val_dataset: Optional[ViewerEnjoymentDataset] = None,
    epochs: int = 30,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-2,
    validation_split: float = 0.2,
    device: str | torch.device | None = None,
) -> Tuple[ViewerEnjoymentModel, TrainingHistory]:
    """Train the *to viewer* model from EEG/audio pairs.

    Args:
        dataset: Training dataset
        val_dataset: Optional validation dataset. If provided, validation_split is ignored.
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for regularization
        validation_split: Fraction of dataset to use for validation (ignored if val_dataset provided)
        device: Device to train on

    Returns:
        Trained model and training history
    """

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    if val_dataset is not None:
        # Use provided validation dataset
        train_subset = dataset
        val_subset = val_dataset
    else:
        # Split the dataset
        total_samples = len(dataset)
        val_size = int(total_samples * validation_split)
        train_size = total_samples - val_size
        if val_size > 0:
            train_subset, val_subset = torch.utils.data.random_split(dataset, [train_size, val_size])
        else:
            train_subset = dataset
            val_subset = None

    def create_loader(subset: Dataset[Tuple[Tensor, Tensor, Tensor]]) -> DataLoader:
        return DataLoader(subset, batch_size=batch_size, shuffle=True, drop_last=False)

    train_loader = create_loader(train_subset)
    val_loader = create_loader(val_subset) if val_subset is not None else None

    model = ViewerEnjoymentModel(
        eeg_channels=dataset.eeg_channels,
        eeg_length=dataset.eeg_length,
        audio_mels=dataset.audio_mels,
        audio_frames=dataset.audio_frames,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    history = TrainingHistory(losses=[], val_losses=[])

    for _ in range(epochs):
        model.train()
        total_loss = 0.0
        total_batches = 0
        for eeg, audio, labels in train_loader:
            eeg = _prepare_eeg_batch(eeg).to(device)
            audio = audio.to(device)
            labels = labels.to(device)

            optimiser.zero_grad()
            logits = model(eeg, audio)
            loss = criterion(logits, labels)
            loss.backward()
            optimiser.step()

            total_loss += loss.item()
            total_batches += 1

        history.losses.append(total_loss / max(total_batches, 1))

        if val_loader is None:
            history.val_losses.append(float("nan"))
            continue

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            val_batches = 0
            for eeg, audio, labels in val_loader:
                eeg = _prepare_eeg_batch(eeg).to(device)
                audio = audio.to(device)
                labels = labels.to(device)
                logits = model(eeg, audio)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                val_batches += 1
        history.val_losses.append(val_loss / max(val_batches, 1))

    return model, history


# ---------------------------------------------------------------------------
# Step 3 – Evaluate article-reading EEG with the viewer model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EnjoymentWindow:
    start_time: float
    end_time: float
    enjoyment_score: float  # Changed from 'probability' to match orchestrator


def evaluate_enjoyment_windows(
    model: ViewerEnjoymentModel,
    brainwave_array: Tensor | np.ndarray,  # Changed parameter name to match orchestrator
    *,
    sample_rate: int = 640,
    window_size: float = 5.0,  # Changed to float (seconds) to match orchestrator
    stride: float = 1.0,  # Changed to float (seconds) to match orchestrator
    audio_supplier: Optional[Callable[[int, int], Tensor]] = None,
    device: str | torch.device | None = None,
) -> List[EnjoymentWindow]:
    """Slide the viewer model across a full EEG recording."""

    if isinstance(brainwave_array, np.ndarray):
        brainwave_array = torch.from_numpy(brainwave_array)
    brainwave_array = brainwave_array.to(torch.float32)
    if brainwave_array.dim() != 2:
        raise ValueError("brainwave_array must have shape (channels, samples)")

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)
    model.eval()

    window_samples = int(window_size * sample_rate)
    stride_samples = max(1, int(stride * sample_rate))
    channels, total_samples = brainwave_array.shape
    results: List[EnjoymentWindow] = []

    with torch.no_grad():
        for start in range(0, total_samples - window_samples + 1, stride_samples):
            end = start + window_samples
            segment = brainwave_array[:, start:end]
            segment = _prepare_eeg_batch(segment)
            if segment.dim() == 2:
                segment = segment.unsqueeze(0)
            segment = segment.to(device)
            if audio_supplier is None:
                audio = torch.zeros(
                    (1, model.audio_mels, model.audio_frames), device=device
                )
            else:
                audio_segment = audio_supplier(start, end)
                audio = audio_segment.to(device=device, dtype=torch.float32)
            if audio.dim() == 2:
                audio = audio.unsqueeze(0)
            logits = model(segment, audio)
            probability = torch.sigmoid(logits.squeeze(0)).item()
            results.append(
                EnjoymentWindow(
                    start_time=start / sample_rate,
                    end_time=end / sample_rate,
                    enjoyment_score=probability,  # Changed field name
                )
            )

    return results


# ---------------------------------------------------------------------------
# Step 4 – To creator model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CreatorAnnotation:
    """Annotation marking an excellent passage in the creator timeline."""

    brainwave_path: Path | str  # Added field
    start_time: float
    end_time: float
    is_creatable: bool  # Added field


class CreatorStateDataset(Dataset[Tuple[Tensor, Tensor]]):
    """EEG segments annotated with whether the creator was in a creatable state."""

    def __init__(self, annotations: Sequence[CreatorAnnotation], sample_rate: int = 640):
        """
        Create dataset from annotations.

        Args:
            annotations: List of CreatorAnnotation objects
            sample_rate: EEG sampling rate (Hz)
        """
        if not annotations:
            raise ValueError("CreatorStateDataset requires at least one annotation.")
        self.annotations = list(annotations)
        self.sample_rate = sample_rate
        self.segments: List[Tuple[Tensor, float]] = []

        # Load and extract segments
        for ann in self.annotations:
            eeg_full = np.load(ann.brainwave_path)
            eeg_tensor = torch.from_numpy(eeg_full).to(torch.float32)

            start_sample = int(ann.start_time * sample_rate)
            end_sample = int(ann.end_time * sample_rate)

            if end_sample > eeg_tensor.shape[1]:
                end_sample = eeg_tensor.shape[1]
            if start_sample >= end_sample:
                continue

            segment = eeg_tensor[:, start_sample:end_sample]
            label = 1.0 if ann.is_creatable else 0.0
            self.segments.append((segment, label))

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        segment, label = self.segments[index]
        eeg = z_score_normalise(segment.to(torch.float32))
        target = torch.tensor([label], dtype=torch.float32)
        return eeg, target


class CreatorStateModel(nn.Module):
    """Single-modality model that classifies creatable brainwave states."""

    def __init__(self, channels: int = 8, samples: int = 3200, hidden_channels: int = 128) -> None:
        super().__init__()
        self.channels = channels
        self.samples = samples
        self.encoder = TemporalFeatureEncoder(
            input_channels=channels,
            hidden_channels=hidden_channels,
            num_blocks=4,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, eeg: Tensor) -> Tensor:
        features = self.encoder(eeg)
        return self.classifier(features)


def train_creator_model(
    dataset: CreatorStateDataset,
    *,
    epochs: int = 40,
    batch_size: int = 8,
    learning_rate: float = 5e-4,
    weight_decay: float = 1e-2,
    device: str | torch.device | None = None,
) -> Tuple[CreatorStateModel, TrainingHistory]:
    """Train the *to creator* model on annotated creator EEG segments."""

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    example, _ = dataset[0]
    channels, samples = example.shape
    model = CreatorStateModel(channels=channels, samples=samples).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    history = TrainingHistory(losses=[], val_losses=[])

    for _ in range(epochs):
        model.train()
        total_loss = 0.0
        batches = 0
        for eeg, labels in loader:
            eeg = eeg.to(device)
            labels = labels.to(device)
            optimiser.zero_grad()
            logits = model(eeg)
            loss = criterion(logits, labels)
            loss.backward()
            optimiser.step()
            total_loss += loss.item()
            batches += 1
        history.losses.append(total_loss / max(batches, 1))
        history.val_losses.append(float("nan"))

    return model, history


# ---------------------------------------------------------------------------
# Step 5 – EEG to Processing parameter regression
# ---------------------------------------------------------------------------


def train_processing_parameter_model(
    dataset_root: Path | str,
    *,
    epochs: int = 120,
    batch_size: int = 32,
    learning_rate: float = 5e-4,
    weight_decay: float = 1e-2,
    validation_split: float = 0.1,
    device: str | torch.device | None = None,
) -> Tuple[BrainwaveToProcessingModel, TrainingHistory, Sequence[str]]:
    """Train the regression model mapping EEG segments to Processing controls."""

    dataset = BrainwaveProcessingDataset(dataset_root, transform=z_score_normalise)
    parameter_names = dataset.parameter_names

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    val_size = int(len(dataset) * validation_split)
    train_size = len(dataset) - val_size
    if val_size > 0:
        train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    else:
        train_set = dataset
        val_set = None

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_pad_batch,
    )
    val_loader = (
        DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_pad_batch,
        )
        if val_set is not None
        else None
    )

    example_features, _ = dataset[0]
    channels, samples = example_features.shape
    model = create_model(input_shape=(channels, samples), parameter_dim=len(parameter_names)).to(device)

    criterion = nn.SmoothL1Loss()
    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    history = TrainingHistory(losses=[], val_losses=[])

    for _ in range(epochs):
        model.train()
        total_loss = 0.0
        batches = 0
        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)
            optimiser.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimiser.step()
            total_loss += loss.item()
            batches += 1
        history.losses.append(total_loss / max(batches, 1))

        if val_loader is None:
            history.val_losses.append(float("nan"))
            continue

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            val_batches = 0
            for features, targets in val_loader:
                features = features.to(device)
                targets = targets.to(device)
                outputs = model(features)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                val_batches += 1
        history.val_losses.append(val_loss / max(val_batches, 1))

    return model, history, parameter_names


# ---------------------------------------------------------------------------
# Step 6 – Real-time feedback loop
# ---------------------------------------------------------------------------


def run_feedback_loop(
    *,
    creator_model: CreatorStateModel,
    parameter_model: BrainwaveToProcessingModel,
    parameter_names: Sequence[str],
    eeg_supplier: Callable[[], np.ndarray | Tensor],  # Changed from eeg_source
    update_callback: Callable[[Dict[str, float], bool], None],  # Added is_creatable param
    creatable_threshold: float = 0.6,
    update_interval: float = 5.0,  # Added parameter
    device: str | torch.device | None = None,
) -> None:
    """Drive a Processing sketch based on live EEG readings.

    ``eeg_supplier`` should be a callable that returns five-second EEG segments
    with shape ``(channels, samples)``. The function is called every
    ``update_interval`` seconds. Whenever the creator model outputs a probability
    exceeding ``creatable_threshold`` the Processing model is executed and the
    resulting parameters are passed to ``update_callback``.

    Args:
        creator_model: Trained creator state detection model
        parameter_model: Trained EEG to Processing parameters model
        parameter_names: List of parameter names
        eeg_supplier: Callable that returns EEG segment when called
        update_callback: Callback(parameters_dict, is_creatable) called with updates
        creatable_threshold: Probability threshold for creatable state
        update_interval: Seconds between EEG reads
        device: Torch device for inference
    """

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    creator_model = creator_model.to(device).eval()
    parameter_model = parameter_model.to(device).eval()

    last_parameters: Optional[Dict[str, float]] = None

    with torch.no_grad():
        while True:
            try:
                # Get EEG segment
                segment = eeg_supplier()

                if isinstance(segment, np.ndarray):
                    segment = torch.from_numpy(segment)
                segment = segment.to(torch.float32)
                segment = z_score_normalise(segment)
                segment = segment.unsqueeze(0).to(device)

                # Check creator state
                logits = creator_model(segment)
                probability = torch.sigmoid(logits.squeeze(0)).item()
                is_creatable = probability >= creatable_threshold

                if is_creatable:
                    # Generate parameters
                    params = parameter_model(segment).squeeze(0)
                    parameter_dict = {
                        name: float(value)
                        for name, value in zip(parameter_names, params.tolist())
                    }
                    update_callback(parameter_dict, is_creatable)
                    last_parameters = parameter_dict
                elif last_parameters is not None:
                    # Maintain last parameters but notify not creatable
                    update_callback(last_parameters, is_creatable)

                # Wait for next interval
                time.sleep(update_interval)

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error in feedback loop: {e}")
                time.sleep(update_interval)


__all__ = [
    "decode_hex_eeg_segment",
    "iter_brainwave_segments",
    "compute_log_mel_spectrogram",
    "ViewerSample",
    "ViewerEnjoymentDataset",
    "ViewerEnjoymentModel",
    "train_viewer_model",
    "evaluate_enjoyment_windows",
    "EnjoymentWindow",
    "CreatorAnnotation",
    "CreatorStateDataset",
    "CreatorStateModel",
    "train_creator_model",
    "train_processing_parameter_model",
    "run_feedback_loop",
    "TrainingHistory",
]
