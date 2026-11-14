"""The six stages of the Aesthetic Beacon pipeline.

1. Convert the headset's hexadecimal recordings into 8×3200 band matrices.
2. Learn a *to viewer* model that predicts aesthetic enjoyment from paired EEG
   and audio segments that participants labelled as enjoyable.
3. Evaluate readers' EEG with the viewer model to find the passages of an
   article that were enjoyed most.
4. Train a *to creator* model on the creator's own EEG, labelled by whether the
   passage being written at that moment turned out to be an excellent one.
5. Learn a mapping from EEG to Processing parameters, using EEG recorded while
   viewing artworks and the parameters derived from those artworks.
6. Run the feedback loop: whenever the creator model reports a *creatable*
   state, refresh the Processing parameters from the current EEG; otherwise
   leave the display on the last parameter set.

Every stage is a plain function or class so it can be used from a notebook, a
script or :mod:`src.workflow_orchestrator`, which chains them automatically.
"""
from __future__ import annotations

import inspect
import json
import logging
import math
import time
import warnings
from dataclasses import dataclass, replace
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import numpy as np
import torch
from scipy import signal as scipy_signal
from scipy.io import wavfile
from torch import Tensor, nn
from torch.utils.data import Dataset

from .brainwave_dataset import BrainwaveProcessingDataset, z_score_normalise
from .eeg_features import (
    DEFAULT_FEATURE_CONFIG,
    BandBaseline,
    BaselineTracker,
    FeatureConfig,
    Recording,
    RecordingError,
    anchored_baseline,
    decode_interleaved_int16,
    load_recording,
    normalise_bands,
    recording_from_stream,
    reference_profile,
)
from .model import (
    BrainwaveToProcessingModel,
    CreatorStateModel,
    ViewerEnjoymentModel,
    create_model,
)
from .thinkgear import hex_text_to_bytes, looks_like_thinkgear
from .training import (
    TrainingError,
    TrainingHistory,
    augment_eeg,
    classification_metrics,
    fit,
    fit_temperature,
    grouped_split,
    make_loader,
    positive_class_weight,
    resolve_device,
    seed_everything,
)

logger = logging.getLogger(__name__)

AUDIO_SAMPLE_RATE = 22050


class PipelineError(RuntimeError):
    """A stage cannot run; the message says what is missing and how to fix it."""


def resolve_path(reference: Path | str, *bases: Path | str | None) -> Path:
    """Locate ``reference`` relative to any of ``bases`` (first existing match wins)."""

    candidate = Path(str(reference)).expanduser()
    if candidate.is_absolute():
        return candidate
    tried: List[Path] = []
    for base in bases:
        if base is None:
            continue
        path = Path(base) / candidate
        tried.append(path)
        if path.exists():
            return path
    return tried[0] if tried else candidate


def read_json(path: Path | str) -> Any:
    path = Path(path)
    try:
        with path.open("r", encoding="utf8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as error:
        raise PipelineError(f"{path} is not valid JSON: {error}") from error


# ---------------------------------------------------------------------------
# Step 1 – Hexadecimal EEG conversion
# ---------------------------------------------------------------------------


def _stream_bytes(hex_stream: str | bytes) -> bytes:
    if isinstance(hex_stream, (bytes, bytearray)):
        hex_stream = bytes(hex_stream).decode("ascii", errors="ignore")
    data = hex_text_to_bytes(hex_stream)
    if not data:
        raise ValueError("No hexadecimal characters found in the provided stream.")
    return data


def iter_brainwave_segments(
    hex_stream: str | bytes,
    *,
    channels: int = 8,
    sample_rate: int = 640,
    segment_seconds: float = 5,
    dtype: np.dtype = np.int16,
    device_sample_rate: int | str | None = "auto",
) -> Iterator[np.ndarray]:
    """Yield consecutive five-second band matrices decoded from a hex dump.

    The stream is expected to be the headset's packet stream (``AA AA 04 80 02
    …``); each yielded array is ``(8, sample_rate * segment_seconds)`` and holds
    the amplitude envelopes of the eight frequency bands.  If the device runs
    at a different rate than ``sample_rate`` (measured from its once-per-second
    status packets, or given as ``device_sample_rate``) the signal is resampled.

    A stream without headset packets is treated as the legacy layout of
    ``channels`` interleaved little-endian ``dtype`` values.
    """

    data = _stream_bytes(hex_stream)
    config = replace(DEFAULT_FEATURE_CONFIG, sample_rate=int(sample_rate), segment_seconds=float(segment_seconds))
    if looks_like_thinkgear(data):
        if channels != config.num_bands:
            raise ValueError(
                f"The headset signal is decomposed into {config.num_bands} bands; channels={channels} is not possible."
            )
        recording = recording_from_stream(data, config, device_sample_rate=device_sample_rate)
        if recording.n_samples < config.segment_samples:
            return
        matrix = recording.band_matrix()
    else:
        matrix = decode_interleaved_int16(data, channels=channels, dtype=dtype)

    length = config.segment_samples
    for start in range(0, matrix.shape[1] - length + 1, length):
        yield matrix[:, start : start + length]


def decode_hex_eeg_segment(
    hex_stream: str | bytes,
    *,
    channels: int = 8,
    sample_rate: int = 640,
    segment_seconds: float = 5,
    dtype: np.dtype = np.int16,
    device_sample_rate: int | str | None = "auto",
) -> np.ndarray:
    """Convert a hex dump into the ``(8, 3200)`` matrix of its first five seconds."""

    for segment in iter_brainwave_segments(
        hex_stream,
        channels=channels,
        sample_rate=sample_rate,
        segment_seconds=segment_seconds,
        dtype=dtype,
        device_sample_rate=device_sample_rate,
    ):
        return segment
    expected = int(sample_rate * segment_seconds)
    raise ValueError(
        f"Segment too short: fewer than {expected} samples ({segment_seconds} s) could be decoded."
    )


def convert_recording(
    path: Path | str,
    output_dir: Path | str,
    config: FeatureConfig = DEFAULT_FEATURE_CONFIG,
    *,
    device_sample_rate: int | str | None = "auto",
) -> Dict[str, object]:
    """Decode one recording, save its five-second segments and report on its quality.

    Writes ``<name>_segment_0000.npy`` … (each ``(8, 3200)``) to ``output_dir``
    and returns a report: duration, measured sampling rate, checksum errors,
    share of the signal with poor headset contact, and how many segments pass
    the quality gate.
    """

    path = Path(path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    recording = load_recording(path, config, device_sample_rate=device_sample_rate)
    matrix = recording.band_matrix()
    length = config.segment_samples

    segments = 0
    clean = 0
    for index, start in enumerate(range(0, matrix.shape[1] - length + 1, length)):
        np.save(output_dir / f"{path.stem}_segment_{index:04d}.npy", matrix[:, start : start + length])
        segments += 1
        if recording.window_quality(start).ok:
            clean += 1

    report: Dict[str, object] = {"file": path.name, "segments": segments, "clean_segments": clean}
    report.update(recording.info)
    return report


# ---------------------------------------------------------------------------
# Step 2 – To viewer model
# ---------------------------------------------------------------------------


@lru_cache(maxsize=8)
def _mel_filter_bank(num_mels: int, n_fft: int, sample_rate: int) -> Tensor:
    """Create a mel filter bank compatible with :func:`torch.stft` output."""

    def hz_to_mel(freq: np.ndarray) -> np.ndarray:
        return 2595.0 * np.log10(1.0 + freq / 700.0)

    def mel_to_hz(mel: np.ndarray) -> np.ndarray:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    fft_bins = n_fft // 2 + 1
    mel_points = np.linspace(hz_to_mel(np.array([0.0]))[0], hz_to_mel(np.array([sample_rate / 2]))[0], num_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_indices = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    bin_indices = np.clip(bin_indices, 0, fft_bins - 1)

    filters = np.zeros((num_mels, fft_bins), dtype=np.float32)
    for m in range(1, num_mels + 1):
        left, centre, right = (int(v) for v in bin_indices[m - 1 : m + 2])
        centre = max(centre, left + 1)
        right = max(right, centre + 1)
        for k in range(left, min(centre, fft_bins)):
            filters[m - 1, k] = (k - left) / max(centre - left, 1)
        for k in range(centre, min(right, fft_bins)):
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
    """Compute a log-mel spectrogram, shaped ``(n_mels, frames)``, for an audio snippet."""

    if isinstance(waveform, np.ndarray):
        waveform = torch.from_numpy(np.ascontiguousarray(waveform))
    waveform = waveform.to(torch.float32)
    if waveform.dim() == 2:
        waveform = waveform.mean(dim=0 if waveform.size(0) <= waveform.size(1) else 1)
    if waveform.numel() < n_fft:
        waveform = nn.functional.pad(waveform, (0, n_fft - waveform.numel()))

    window = torch.hann_window(n_fft, device=waveform.device)
    stft = torch.stft(
        waveform.unsqueeze(0),
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        return_complex=True,
    )
    power = stft.abs() ** 2
    mel_basis = _mel_filter_bank(n_mels, n_fft, sample_rate).to(power.device, power.dtype)
    return torch.log(torch.matmul(mel_basis, power).clamp_min(1e-10)).squeeze(0)


@lru_cache(maxsize=16)
def _load_audio_cached(path: str, modified: int, target_rate: int, assumed_rate: int) -> np.ndarray:
    del modified
    file = Path(path)
    suffix = file.suffix.lower()
    if suffix == ".npy":
        data = np.load(file, allow_pickle=False)
        rate = assumed_rate
    elif suffix == ".wav":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rate, data = wavfile.read(file)
    else:
        try:
            import soundfile  # type: ignore[import-not-found]
        except ImportError as error:
            raise PipelineError(
                f"Cannot read '{file.name}': only .wav files are supported out of the box. "
                "Convert the audio to WAV, or install the 'soundfile' package."
            ) from error
        data, rate = soundfile.read(str(file), always_2d=False)

    data = np.asarray(data)
    if data.dtype == np.uint8:
        samples = (data.astype(np.float32) - 128.0) / 128.0
    elif np.issubdtype(data.dtype, np.integer):
        samples = data.astype(np.float32) / float(np.iinfo(data.dtype).max)
    else:
        samples = data.astype(np.float32)
    if samples.ndim == 2:
        samples = samples.mean(axis=1 if samples.shape[1] <= 8 else 0)
    if samples.ndim != 1 or samples.size == 0:
        raise PipelineError(f"Audio file '{file}' is empty or has an unsupported layout.")
    if not np.all(np.isfinite(samples)):
        raise PipelineError(f"Audio file '{file}' contains invalid samples.")
    if int(rate) != target_rate:
        ratio = Fraction(target_rate, int(rate)).limit_denominator(1000)
        samples = scipy_signal.resample_poly(samples, ratio.numerator, ratio.denominator).astype(np.float32)
    return samples


def load_audio(
    path: Path | str,
    target_rate: int = AUDIO_SAMPLE_RATE,
    *,
    assumed_rate: int = 44100,
) -> np.ndarray:
    """Load audio as mono ``float32`` at ``target_rate``.

    ``.wav`` files carry their own sampling rate.  ``.npy`` arrays do not and
    are assumed to be sampled at ``assumed_rate``.
    """

    path = Path(path)
    if not path.is_file():
        raise PipelineError(f"Audio file not found: {path}")
    try:
        return _load_audio_cached(str(path.resolve()), path.stat().st_mtime_ns, int(target_rate), int(assumed_rate))
    except PipelineError:
        raise
    except Exception as error:
        raise PipelineError(f"Could not read audio file '{path}': {error}") from error


@dataclass(frozen=True)
class ViewerSample:
    """One five-second EEG window, the audio heard during it, and whether it was enjoyed.

    ``start_time`` locates the window inside a longer recording (seconds); when
    omitted the recording's middle five seconds are used.  ``audio_start_time``
    defaults to ``start_time``.  Samples sharing a ``group`` are kept on the
    same side of the train/validation split.
    """

    brainwave_path: Path | str
    audio_path: Path | str | None
    enjoyed: bool
    start_time: Optional[float] = None
    audio_start_time: Optional[float] = None
    group: Optional[str] = None


def _moment_interval(moment: Any, half_window: float) -> Tuple[float, float]:
    if isinstance(moment, (int, float)):
        return float(moment) - half_window, float(moment) + half_window
    if isinstance(moment, Mapping):
        if "time" in moment:
            return float(moment["time"]) - half_window, float(moment["time"]) + half_window
        return float(moment["start_time"]), float(moment["end_time"])
    if isinstance(moment, Sequence) and len(moment) == 2:
        return float(moment[0]), float(moment[1])
    raise PipelineError(f"Cannot interpret enjoyed moment {moment!r}; use a time or [start, end].")


def _tile(start: float, end: float, window: float, hop: float) -> List[float]:
    """Window start times covering ``[start, end]``; at least one, centred if it is short."""

    if end - start <= window:
        return [(start + end - window) / 2.0]
    count = int(math.floor((end - start - window) / hop + 1e-9)) + 1
    return [start + index * hop for index in range(count)]


def build_viewer_samples(
    annotations_path: Path | str,
    data_root: Path | str | None = None,
    config: FeatureConfig = DEFAULT_FEATURE_CONFIG,
    *,
    seed: int = 0,
    windows_per_moment: int = 3,
    jitter_seconds: float = 1.0,
) -> List[ViewerSample]:
    """Expand a Step 2 annotation file into training samples.

    Two layouts are understood and may be mixed:

    ``"samples"``
        A list of pre-cut five-second pairs
        ``{"eeg_file", "audio_file", "enjoyed"}``.

    ``"sessions"``
        Whole listening sessions ``{"eeg_file", "audio_file", "enjoyed": [...],
        "audio_offset": 0.0}`` where ``enjoyed`` lists the moments the listener
        named afterwards, each a time in seconds or a ``[start, end]`` interval.
        Windows around those moments become positive examples; an equal number
        of windows from the rest of the same session, clear of every enjoyed
        moment, become the matched controls.
    """

    annotations_path = Path(annotations_path)
    if not annotations_path.is_file():
        raise PipelineError(
            f"Annotations file not found: {annotations_path}\n"
            "Step 2 needs a JSON file describing which listening moments were enjoyed "
            "(see data/README.md)."
        )
    content = read_json(annotations_path)
    if not isinstance(content, dict):
        raise PipelineError(f"{annotations_path} must contain a JSON object.")
    bases = (data_root, annotations_path.parent, annotations_path.parent.parent, Path(data_root).parent if data_root else None)
    rng = np.random.default_rng(seed)
    window = config.segment_seconds
    samples: List[ViewerSample] = []

    for index, item in enumerate(content.get("samples", []) or []):
        try:
            eeg_path = resolve_path(item["eeg_file"], *bases)
            audio = item.get("audio_file")
            samples.append(
                ViewerSample(
                    brainwave_path=eeg_path,
                    audio_path=resolve_path(audio, *bases) if audio else None,
                    enjoyed=bool(item["enjoyed"]),
                    start_time=item.get("start_time"),
                    audio_start_time=item.get("audio_start_time"),
                    group=str(item.get("group", f"sample:{eeg_path}")),
                )
            )
        except KeyError as error:
            raise PipelineError(f"{annotations_path}: sample {index} is missing the field {error}.") from error

    for session_index, session in enumerate(content.get("sessions", []) or []):
        if "eeg_file" not in session:
            raise PipelineError(f"{annotations_path}: session {session_index} is missing 'eeg_file'.")
        eeg_path = resolve_path(session["eeg_file"], *bases)
        audio = session.get("audio_file")
        audio_path = resolve_path(audio, *bases) if audio else None
        offset = float(session.get("audio_offset", 0.0))
        try:
            duration = load_recording(eeg_path, config).duration
        except RecordingError as error:
            raise PipelineError(str(error)) from error
        if duration < window:
            raise PipelineError(f"{eeg_path} lasts {duration:.1f} s, shorter than one {window:.0f} s window.")
        latest = duration - window

        intervals = [_moment_interval(moment, window / 2.0) for moment in session.get("enjoyed", [])]
        positives = 0
        for moment_index, (start, end) in enumerate(intervals):
            starts = _tile(start, end, window, window / 2.0)
            if len(starts) == 1 and windows_per_moment > 1:
                starts = [starts[0] + shift for shift in np.linspace(-jitter_seconds, jitter_seconds, windows_per_moment)]
            for begin in sorted({round(min(max(s, 0.0), latest), 3) for s in starts}):
                samples.append(
                    ViewerSample(eeg_path, audio_path, True, begin, begin + offset, f"{eeg_path}:enjoyed:{moment_index}")
                )
                positives += 1

        guard = window / 2.0
        candidates = [
            begin
            for begin in np.arange(0.0, latest + 1e-9, window)
            if all(begin + window <= start - guard or begin >= end + guard for start, end in intervals)
        ]
        wanted = positives if intervals else len(candidates)
        if len(candidates) > wanted:
            chosen = sorted(rng.choice(len(candidates), size=wanted, replace=False).tolist())
            candidates = [candidates[i] for i in chosen]
        for begin in candidates:
            begin = float(round(begin, 3))
            samples.append(
                ViewerSample(eeg_path, audio_path, False, begin, begin + offset, f"{eeg_path}:control:{begin}")
            )

    if not samples:
        raise PipelineError(
            f"{annotations_path} lists no usable data. Add entries under 'sessions' or 'samples'."
        )
    return samples


class ViewerEnjoymentDataset(Dataset[Tuple[Tensor, Tensor, Tensor]]):
    """Paired EEG/audio segments with enjoyment labels.

    Each item is ``(eeg, audio, label)``: the normalised ``(8, 3200)`` band
    matrix, the ``(n_mels, frames)`` log-mel spectrogram of the audio heard
    during it, and a ``(1,)`` tensor that is 1.0 when the segment was enjoyed.
    Windows that fail the signal-quality check are left out.
    """

    def __init__(
        self,
        samples: Sequence[ViewerSample],
        *,
        audio_sample_rate: int = AUDIO_SAMPLE_RATE,
        n_fft: int = 1024,
        hop_length: int = 320,
        n_mels: int = 64,
        feature_config: FeatureConfig = DEFAULT_FEATURE_CONFIG,
    ) -> None:
        if not samples:
            raise ValueError("ViewerEnjoymentDataset requires at least one sample.")
        self.audio_sample_rate = audio_sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.feature_config = feature_config

        window = feature_config.segment_samples
        audio_length = int(round(feature_config.segment_seconds * audio_sample_rate))
        self.has_audio = any(sample.audio_path is not None for sample in samples)

        kept: List[ViewerSample] = []
        eeg: List[np.ndarray] = []
        audio: List[Tensor] = []
        labels: List[float] = []
        groups: List[str] = []
        self.skipped = 0

        for sample in samples:
            try:
                recording = load_recording(Path(sample.brainwave_path), feature_config)
                if sample.start_time is None:
                    start = max(0, (recording.n_samples - window) // 2)
                else:
                    start = int(round(float(sample.start_time) * feature_config.sample_rate))
                start = recording.clamp_start(start)
            except RecordingError as error:
                raise PipelineError(str(error)) from error
            if not recording.window_quality(start).ok:
                self.skipped += 1
                continue
            eeg.append(recording.normalised_window(start))

            waveform = np.zeros(audio_length, dtype=np.float32)
            if sample.audio_path is not None:
                source = load_audio(sample.audio_path, audio_sample_rate)
                if sample.audio_start_time is not None:
                    audio_start = float(sample.audio_start_time)
                elif sample.start_time is not None:
                    audio_start = float(sample.start_time)
                else:
                    audio_start = max(0.0, (len(source) / audio_sample_rate - feature_config.segment_seconds) / 2.0)
                first = int(round(audio_start * audio_sample_rate))
                lo, hi = max(first, 0), min(first + audio_length, len(source))
                if hi > lo:
                    waveform[lo - first : hi - first] = source[lo:hi]
            audio.append(
                compute_log_mel_spectrogram(
                    waveform, sample_rate=audio_sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
                )
            )
            labels.append(1.0 if sample.enjoyed else 0.0)
            groups.append(sample.group or f"{sample.brainwave_path}:{start}")
            kept.append(sample)

        if not kept:
            raise PipelineError(
                "None of the listening windows passed the signal-quality check. "
                "Inspect the recordings for poor headset contact."
            )
        if self.skipped:
            logger.warning("Left out %d of %d listening windows with poor signal quality.", self.skipped, len(samples))

        self.samples = kept
        self.eeg = torch.from_numpy(np.stack(eeg))
        self.audio = torch.stack(audio)
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
        self.groups = groups
        self.audio_mean = float(self.audio.mean()) if self.has_audio else 0.0
        self.audio_std = max(float(self.audio.std()), 1e-3) if self.has_audio else 1.0

        self.eeg_channels = int(self.eeg.shape[1])
        self.eeg_length = int(self.eeg.shape[2])
        self.audio_mels = int(self.audio.shape[1])
        self.audio_frames = int(self.audio.shape[2])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor]:
        return self.eeg[index], self.audio[index], self.labels[index]


def _prepare_eeg_batch(eeg: Tensor) -> Tensor:
    """Per-channel z-score for raw tensors handed in by external datasets."""

    if eeg.dim() not in (2, 3):
        raise ValueError("EEG tensors must have shape (batch, channels, samples)")
    return z_score_normalise(eeg)


def _dataset_tensors(dataset: Dataset, fields: Sequence[str]) -> List[Tensor]:
    """Tensors of a dataset: its pre-built attributes, or stacked items."""

    if all(hasattr(dataset, name) for name in fields):
        return [getattr(dataset, name) for name in fields]
    columns: List[List[Tensor]] = [[] for _ in fields]
    for index in range(len(dataset)):  # type: ignore[arg-type]
        item = dataset[index]
        for column, value in zip(columns, item):
            column.append(torch.as_tensor(value, dtype=torch.float32))
    return [torch.stack(column) for column in columns]


def _require_two_classes(labels: Tensor, what: str, positive: str, negative: str) -> None:
    positives = int((labels > 0.5).sum())
    negatives = int((labels <= 0.5).sum())
    if positives == 0 or negatives == 0:
        raise PipelineError(
            f"The {what} needs examples of both kinds, but the data has {positives} {positive} "
            f"and {negatives} {negative} windows."
        )


def _split_tensors(
    tensors: Sequence[Tensor],
    groups: Sequence[str],
    labels: Optional[Tensor],
    validation_split: float,
    seed: int,
) -> Tuple[List[Tensor], Optional[List[Tensor]]]:
    label_list = None if labels is None else [int(v > 0.5) for v in labels.flatten().tolist()]
    train_index, val_index = grouped_split(groups, label_list, validation_split, seed)
    if not val_index:
        return list(tensors), None
    train = torch.tensor(train_index)
    val = torch.tensor(val_index)
    return [t[train] for t in tensors], [t[val] for t in tensors]


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
    lr: Optional[float] = None,
    hidden_channels: int = 128,
    input_stride: int = 8,
    audio_dropout: float = 0.5,
    patience: int = 10,
    seed: int = 0,
) -> Tuple[ViewerEnjoymentModel, TrainingHistory]:
    """Train the *to viewer* model from EEG/audio pairs.

    Validation data is split off by listening moment (or taken from
    ``val_dataset``), the best epoch is restored and the output probability is
    calibrated.  ``history.metrics`` reports accuracy and AUC with audio and,
    as ``auc_eeg_only``, for EEG alone — the mode Step 3 uses on readers.
    """

    if lr is not None:
        learning_rate = lr
    seed_everything(seed)
    device = resolve_device(device)

    eeg, audio, labels = _dataset_tensors(dataset, ("eeg", "audio", "labels"))
    labels = labels.reshape(-1, 1).float()
    _require_two_classes(labels, "viewer model", "enjoyed", "not-enjoyed")
    groups = getattr(dataset, "groups", None) or [str(i) for i in range(len(labels))]
    use_audio = bool(getattr(dataset, "has_audio", True))

    if val_dataset is not None:
        train_tensors = [eeg, audio, labels]
        v_eeg, v_audio, v_labels = _dataset_tensors(val_dataset, ("eeg", "audio", "labels"))
        val_tensors: Optional[List[Tensor]] = [v_eeg, v_audio, v_labels.reshape(-1, 1).float()]
    else:
        train_tensors, val_tensors = _split_tensors([eeg, audio, labels], groups, labels, validation_split, seed)
    if val_tensors is None:
        logger.warning("Too little data for a validation split; the viewer model is trained without one.")

    model = ViewerEnjoymentModel(
        eeg_channels=int(eeg.shape[1]),
        eeg_length=int(eeg.shape[2]),
        audio_mels=int(audio.shape[1]),
        audio_frames=int(audio.shape[2]),
        hidden_channels=hidden_channels,
        input_stride=input_stride,
        audio_dropout=audio_dropout,
    ).to(device)
    model.audio_mean.fill_(float(getattr(dataset, "audio_mean", float(audio.mean()))))
    model.audio_std.fill_(float(getattr(dataset, "audio_std", max(float(audio.std()), 1e-3))))

    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_class_weight(train_tensors[2]).to(device))

    def step(batch: Sequence[Tensor], training: bool) -> Tuple[Tensor, Tensor]:
        batch_eeg, batch_audio, batch_labels = batch
        if training:
            batch_eeg = augment_eeg(batch_eeg)
        logits = model(batch_eeg, batch_audio if use_audio else None)
        return criterion(logits, batch_labels), logits

    history, (logits, targets) = fit(
        model,
        make_loader(train_tensors, batch_size, shuffle=True, seed=seed),
        make_loader(val_tensors, batch_size, shuffle=False, seed=seed) if val_tensors else None,
        step,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        device=device,
        patience=patience,
        label="viewer model",
    )

    evaluation = val_tensors if val_tensors is not None else train_tensors
    with torch.no_grad():
        eeg_only = torch.cat(
            [model(chunk.to(device), None).cpu() for chunk in evaluation[0].split(max(1, batch_size))]
        )
    if val_tensors is not None:
        model.set_temperature(fit_temperature(eeg_only, evaluation[2]))
    history.metrics = classification_metrics(logits, targets)
    history.metrics["auc_eeg_only"] = classification_metrics(eeg_only, evaluation[2])["auc"]
    history.metrics["temperature"] = float(model.temperature)
    history.metrics["evaluated_on"] = "validation" if val_tensors is not None else "training"  # type: ignore[assignment]
    history.metrics["train_windows"] = int(train_tensors[0].shape[0])
    model.eval()
    return model, history


# ---------------------------------------------------------------------------
# Step 3 – Evaluate article-reading EEG with the viewer model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EnjoymentWindow:
    start_time: float
    end_time: float
    enjoyment_score: float


def _as_recording(
    source: Recording | Path | str | Tensor | np.ndarray,
    config: FeatureConfig,
) -> Recording:
    if isinstance(source, Recording):
        return source
    if isinstance(source, (str, Path)):
        return load_recording(source, config)
    array = source.detach().cpu().numpy() if isinstance(source, Tensor) else np.asarray(source)
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 1:
        return Recording(kind="raw", config=config, raw=array)
    if array.ndim == 2:
        return Recording(kind="bands", config=config, bands=array)
    raise ValueError("brainwave_array must have shape (channels, samples)")


def evaluate_enjoyment_windows(
    model: ViewerEnjoymentModel,
    brainwave_array: Recording | Path | str | Tensor | np.ndarray,
    *,
    sample_rate: int = 640,
    window_size: float = 5.0,
    stride: float = 1.0,
    audio_supplier: Optional[Callable[[int, int], Tensor]] = None,
    device: str | torch.device | None = None,
    feature_config: Optional[FeatureConfig] = None,
    batch_size: int = 64,
) -> List[EnjoymentWindow]:
    """Slide the viewer model across a full EEG recording.

    ``brainwave_array`` may be a recording file, a loaded
    :class:`~src.eeg_features.Recording`, a band matrix ``(8, samples)`` or a
    raw signal at ``sample_rate``.  Without ``audio_supplier`` the model's
    EEG-only path is used, which is the normal case for reading.  Windows that
    fail the signal-quality check are omitted from the result.
    """

    config = feature_config or DEFAULT_FEATURE_CONFIG
    if sample_rate != config.sample_rate:
        config = replace(config, sample_rate=int(sample_rate))
    recording = _as_recording(brainwave_array, config)

    device = resolve_device(device)
    model = model.to(device).eval()

    window_samples = int(round(window_size * config.sample_rate))
    stride_samples = max(1, int(round(stride * config.sample_rate)))
    if recording.n_samples < window_samples:
        return []
    baseline: Optional[BandBaseline] = (
        recording.baseline() if recording.n_samples >= 2 * window_samples else None
    )

    results: List[EnjoymentWindow] = []
    pending_eeg: List[np.ndarray] = []
    pending_audio: List[Tensor] = []
    pending_bounds: List[Tuple[int, int]] = []

    def flush() -> None:
        if not pending_eeg:
            return
        eeg = torch.from_numpy(np.stack(pending_eeg)).to(device)
        audio = torch.stack(pending_audio).to(device=device, dtype=torch.float32) if pending_audio else None
        probabilities = model.predict_proba(eeg, audio).cpu().tolist()
        for (start, end), probability in zip(pending_bounds, probabilities):
            results.append(
                EnjoymentWindow(
                    start_time=start / config.sample_rate,
                    end_time=end / config.sample_rate,
                    enjoyment_score=float(probability),
                )
            )
        pending_eeg.clear()
        pending_audio.clear()
        pending_bounds.clear()

    for start in range(0, recording.n_samples - window_samples + 1, stride_samples):
        end = start + window_samples
        if not recording.window_quality(start, window_samples).ok:
            continue
        pending_eeg.append(normalise_bands(recording.window(start, window_samples), baseline))
        if audio_supplier is not None:
            segment = audio_supplier(start, end)
            pending_audio.append(segment.squeeze(0) if segment.dim() == 3 else segment)
        pending_bounds.append((start, end))
        if len(pending_eeg) >= batch_size:
            flush()
    flush()
    return results


def split_passages(text: str) -> List[str]:
    """Split an article into passages: paragraphs separated by blank lines."""

    normalised = text.replace("\r\n", "\n").replace("\r", "\n")
    passages = [block.strip() for block in normalised.split("\n\n") if block.strip()]
    if len(passages) <= 1:
        passages = [line.strip() for line in normalised.split("\n") if line.strip()]
    return passages


def passage_intervals(
    passages: Sequence[str],
    duration: float,
    *,
    start: float = 0.0,
    end: Optional[float] = None,
) -> List[Tuple[float, float]]:
    """Estimate when each passage was read, assuming a steady reading speed.

    The reading time between ``start`` and ``end`` (default: the whole
    recording) is divided among the passages in proportion to their length.
    """

    end = duration if end is None else min(float(end), duration)
    start = max(0.0, float(start))
    if end <= start or not passages:
        return []
    lengths = np.array([max(len(passage), 1) for passage in passages], dtype=np.float64)
    edges = start + (end - start) * np.concatenate([[0.0], np.cumsum(lengths) / lengths.sum()])
    return [(float(edges[i]), float(edges[i + 1])) for i in range(len(passages))]


@dataclass(frozen=True)
class PassageScore:
    """How much one passage was enjoyed, averaged over readers."""

    index: int
    score: float
    readers: int
    mean_probability: float
    preview: str = ""


def score_passages(
    reader_windows: Mapping[str, Sequence[EnjoymentWindow]],
    reader_intervals: Mapping[str, Sequence[Tuple[float, float]]],
    passages: Sequence[str],
) -> List[PassageScore]:
    """Combine window scores into one enjoyment score per passage.

    Each reader's window scores are first standardised against that reader's
    own average, so a reader who scores high throughout does not dominate.  A
    passage's score is the mean standardised score of the windows centred in
    it, averaged over the readers who have at least one such window.
    """

    totals = [[] for _ in passages]  # type: List[List[float]]
    probabilities = [[] for _ in passages]  # type: List[List[float]]
    for reader, windows in reader_windows.items():
        intervals = reader_intervals.get(reader)
        if not windows or not intervals:
            continue
        values = np.array([w.enjoyment_score for w in windows], dtype=np.float64)
        centres = np.array([(w.start_time + w.end_time) / 2.0 for w in windows])
        spread = values.std()
        standardised = (values - values.mean()) / spread if spread > 1e-6 else np.zeros_like(values)
        for index, (begin, finish) in enumerate(intervals[: len(passages)]):
            inside = (centres >= begin) & (centres < finish)
            if inside.any():
                totals[index].append(float(standardised[inside].mean()))
                probabilities[index].append(float(values[inside].mean()))

    scores: List[PassageScore] = []
    for index, passage in enumerate(passages):
        if totals[index]:
            scores.append(
                PassageScore(
                    index=index,
                    score=float(np.mean(totals[index])),
                    readers=len(totals[index]),
                    mean_probability=float(np.mean(probabilities[index])),
                    preview=passage[:80],
                )
            )
    return scores


def select_passages(
    scores: Sequence[PassageScore],
    fraction: float = 0.3,
) -> Tuple[List[int], List[int]]:
    """Pick the excellent passages (top ``fraction``) and equally many controls (bottom)."""

    if len(scores) < 2:
        return [score.index for score in scores], []
    ranked = sorted(scores, key=lambda item: item.score, reverse=True)
    count = min(max(1, int(round(len(ranked) * fraction))), len(ranked) // 2)
    return [item.index for item in ranked[:count]], [item.index for item in ranked[-count:]]


# ---------------------------------------------------------------------------
# Step 4 – To creator model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CreatorAnnotation:
    """A stretch of the creator's recording and whether it was a creatable state.

    Stretches longer than five seconds are cut into overlapping windows.
    Annotations sharing a ``group`` stay on one side of the validation split.
    """

    brainwave_path: Path | str
    start_time: float
    end_time: float
    is_creatable: bool
    group: Optional[str] = None


def _keystroke_positions(events: Sequence[Mapping[str, Any]], started_at: Optional[float]) -> List[Tuple[float, int]]:
    """Reduce a keystroke log to ``(seconds into the recording, characters written)``."""

    positions: List[Tuple[float, int]] = []
    count = 0
    for event in events:
        if "t" in event:
            when = float(event["t"])
        elif "timestamp" in event:
            when = float(event["timestamp"])
            if started_at is not None:
                when -= float(started_at)
        else:
            continue
        if "text" in event:
            count = len(str(event["text"]))
        elif "length" in event:
            count = int(event["length"])
        elif "key" in event:
            key = str(event["key"])
            if key in ("Backspace", "BackSpace", "Delete", "\b"):
                count = max(0, count - 1)
            elif len(key) == 1 or key in ("Enter", "Return", "space", "Space", "Tab"):
                count += 1
            else:
                continue
        else:
            continue
        positions.append((when, count))
    positions.sort(key=lambda item: item[0])
    return positions


def timeline_from_keystrokes(
    events: Sequence[Mapping[str, Any]],
    passages: Sequence[str],
    *,
    started_at: Optional[float] = None,
    pause_limit: float = 20.0,
) -> List[Dict[str, float]]:
    """Work out when each passage was being written from a keystroke log.

    ``events`` are dictionaries with a time — ``t`` in seconds from the start
    of the EEG recording, or ``timestamp`` as a Unix time together with
    ``started_at`` — and one of ``text`` (the document so far), ``length``
    (characters written so far) or ``key`` (a single key press).  The text
    length at each event locates the passage of the finished article that was
    being written.  Time spent idle beyond ``pause_limit`` seconds before a
    burst of typing is not attributed to any passage.
    """

    positions = _keystroke_positions(events, started_at)
    if len(positions) < 2 or not passages:
        return []
    boundaries = np.cumsum([len(passage) + 2 for passage in passages])
    final_length = max(position for _, position in positions)
    scale = boundaries[-1] / final_length if final_length > 0 else 1.0

    def passage_at(characters: int) -> int:
        return int(min(np.searchsorted(boundaries, characters * scale, side="left"), len(passages) - 1))

    timeline: List[Dict[str, float]] = []
    for (previous_time, _), (when, characters) in zip(positions[:-1], positions[1:]):
        if when <= previous_time:
            continue
        begin = max(previous_time, when - pause_limit)
        index = passage_at(characters)
        if timeline and timeline[-1]["passage"] == index and begin - timeline[-1]["end_time"] < 1e-6:
            timeline[-1]["end_time"] = when
        else:
            timeline.append({"passage": index, "start_time": begin, "end_time": when})
    return timeline


def read_keystroke_log(path: Path | str) -> List[Dict[str, Any]]:
    """Read a keystroke log stored as JSON lines or as one JSON array."""

    path = Path(path)
    if not path.is_file():
        raise PipelineError(f"Keystroke log not found: {path}")
    text = path.read_text(encoding="utf8").strip()
    if not text:
        return []
    if text.startswith("["):
        content = read_json(path)
        return [event for event in content if isinstance(event, dict)]
    events: List[Dict[str, Any]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("%s: skipping unreadable line %d", path.name, number)
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def annotations_from_timeline(
    brainwave_path: Path | str,
    timeline: Sequence[Mapping[str, Any]],
    excellent: Iterable[int],
    control: Iterable[int],
    *,
    minimum_seconds: float = 2.0,
) -> List[CreatorAnnotation]:
    """Label the creator's recording from when each passage was written.

    Time spent writing an ``excellent`` passage is a creatable state; time
    spent on a ``control`` passage is not.  Other passages are left unlabelled.
    """

    excellent_set: Set[int] = set(int(i) for i in excellent)
    control_set: Set[int] = set(int(i) for i in control)
    annotations: List[CreatorAnnotation] = []
    for entry in timeline:
        index = int(entry["passage"])
        if index not in excellent_set and index not in control_set:
            continue
        start, end = float(entry["start_time"]), float(entry["end_time"])
        if end - start < minimum_seconds:
            continue
        annotations.append(
            CreatorAnnotation(
                brainwave_path=brainwave_path,
                start_time=start,
                end_time=end,
                is_creatable=index in excellent_set,
                group=f"{brainwave_path}:passage:{index}",
            )
        )
    return annotations


def annotations_from_manual_file(
    path: Path | str,
    data_root: Path | str | None = None,
) -> List[CreatorAnnotation]:
    """Read hand-made creator annotations.

    The file describes one session, or several under ``"sessions"``.  A session
    names its ``eeg_file`` and lists ``excellent_passages``, each with
    ``start_time``, ``end_time`` and a ``label`` of ``"creatable"`` or
    ``"not_creatable"``.
    """

    path = Path(path)
    content = read_json(path)
    sessions = content.get("sessions", [content]) if isinstance(content, dict) else content
    bases = (data_root, path.parent, path.parent.parent, Path(data_root).parent if data_root else None)
    annotations: List[CreatorAnnotation] = []
    for session in sessions:
        if not isinstance(session, dict) or not session.get("excellent_passages"):
            continue
        if not session.get("eeg_file"):
            raise PipelineError(f"{path}: a session with annotated passages is missing 'eeg_file'.")
        eeg_path = resolve_path(session["eeg_file"], *bases)
        for index, passage in enumerate(session["excellent_passages"]):
            try:
                label = str(passage.get("label", "creatable")).strip().lower()
                annotations.append(
                    CreatorAnnotation(
                        brainwave_path=eeg_path,
                        start_time=float(passage["start_time"]),
                        end_time=float(passage["end_time"]),
                        is_creatable=label in ("creatable", "excellent", "true", "1", "yes"),
                        group=f"{eeg_path}:manual:{index}",
                    )
                )
            except (KeyError, TypeError, ValueError) as error:
                raise PipelineError(f"{path}: passage {index} is malformed ({error}).") from error
    return annotations


class CreatorStateDataset(Dataset[Tuple[Tensor, Tensor]]):
    """EEG windows annotated with whether the creator was in a creatable state.

    ``segments`` holds ``(eeg, label)`` pairs with ``eeg`` the normalised
    ``(8, 3200)`` band matrix.  Windows failing the signal-quality check are
    left out.
    """

    def __init__(
        self,
        annotations: Sequence[CreatorAnnotation],
        sample_rate: int = 640,
        *,
        feature_config: FeatureConfig = DEFAULT_FEATURE_CONFIG,
        window_hop: float = 2.5,
        baseline_seconds: float = 180.0,
        calibration_seconds: float = 20.0,
        split_block_seconds: float = 15.0,
        reference: Optional[BandBaseline] = None,
    ) -> None:
        if not annotations:
            raise ValueError("CreatorStateDataset requires at least one annotation.")
        if sample_rate != feature_config.sample_rate:
            feature_config = replace(feature_config, sample_rate=int(sample_rate))
        self.annotations = list(annotations)
        self.sample_rate = sample_rate
        self.feature_config = feature_config
        self.segments: List[Tuple[Tensor, float]] = []
        self.groups: List[str] = []
        self.skipped = 0

        # The creator's typical band profile: every window, here and live, is
        # expressed relative to it (see eeg_features.anchored_baseline).
        if reference is None:
            try:
                distinct = sorted({str(Path(a.brainwave_path)) for a in self.annotations})
                reference = reference_profile([load_recording(Path(p), feature_config).baseline() for p in distinct])
            except RecordingError as error:
                raise PipelineError(str(error)) from error
        self.reference = reference

        window = feature_config.segment_seconds
        length = feature_config.segment_samples
        for index, annotation in enumerate(self.annotations):
            try:
                recording = load_recording(Path(annotation.brainwave_path), feature_config)
                if recording.duration < window:
                    raise RecordingError(
                        f"{recording.name} lasts {recording.duration:.1f} s, shorter than one {window:.0f} s window."
                    )
            except RecordingError as error:
                raise PipelineError(str(error)) from error
            begin = max(0.0, float(annotation.start_time))
            finish = min(float(annotation.end_time), recording.duration)
            if finish <= begin:
                continue
            for start_time in _tile(begin, finish, window, window_hop):
                start = recording.clamp_start(int(round(start_time * feature_config.sample_rate)))
                if not recording.window_quality(start).ok:
                    self.skipped += 1
                    continue
                # Normalise exactly as the live loop will: against the preceding minutes.
                baseline = anchored_baseline(
                    recording.running_baseline(
                        start + length, history_seconds=baseline_seconds, min_seconds=calibration_seconds
                    ),
                    reference,
                    feature_config.baseline_adaptation,
                )
                eeg = torch.from_numpy(recording.normalised_window(start, baseline))
                self.segments.append((eeg, 1.0 if annotation.is_creatable else 0.0))
                # Overlapping neighbours stay together in the validation split.
                block = int((start_time - begin) // split_block_seconds)
                self.groups.append(f"{annotation.group or f'{annotation.brainwave_path}:{index}'}:{block}")

        if self.skipped:
            logger.warning("Left out %d creator windows with poor signal quality.", self.skipped)
        if not self.segments:
            raise PipelineError(
                "The creator annotations produced no usable EEG windows. Check that the annotated "
                "times lie inside the recordings and that the signal quality is acceptable."
            )
        self._refresh()

    @classmethod
    def from_segments(cls, segments: Sequence[Tuple[Tensor, float]]) -> "CreatorStateDataset":
        """Build a dataset from ready-made ``(eeg, label)`` pairs."""

        if not segments:
            raise ValueError("CreatorStateDataset requires at least one segment.")
        dataset = cls.__new__(cls)
        dataset.annotations = []
        dataset.sample_rate = DEFAULT_FEATURE_CONFIG.sample_rate
        dataset.feature_config = DEFAULT_FEATURE_CONFIG
        dataset.segments = [(torch.as_tensor(eeg, dtype=torch.float32), float(label)) for eeg, label in segments]
        dataset.groups = [str(index) for index in range(len(segments))]
        dataset.skipped = 0
        dataset.reference = None
        dataset._refresh()
        return dataset

    def _refresh(self) -> None:
        self.eeg = torch.stack([segment for segment, _ in self.segments])
        self.labels = torch.tensor([label for _, label in self.segments], dtype=torch.float32).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        segment, label = self.segments[index]
        return segment, torch.tensor([label], dtype=torch.float32)


def train_creator_model(
    dataset: CreatorStateDataset,
    *,
    val_dataset: Optional[CreatorStateDataset] = None,
    epochs: int = 40,
    batch_size: int = 8,
    learning_rate: float = 5e-4,
    weight_decay: float = 1e-2,
    validation_split: float = 0.2,
    device: str | torch.device | None = None,
    lr: Optional[float] = None,
    hidden_channels: int = 128,
    input_stride: int = 8,
    patience: int = 10,
    seed: int = 0,
) -> Tuple[CreatorStateModel, TrainingHistory]:
    """Train the *to creator* model on annotated creator EEG segments.

    The validation split keeps all windows of one passage together.  The
    model's probability is calibrated on the validation data, which is what
    makes the live ``creatable_threshold`` meaningful.
    """

    if lr is not None:
        learning_rate = lr
    seed_everything(seed)
    device = resolve_device(device)

    eeg, labels = _dataset_tensors(dataset, ("eeg", "labels"))
    labels = labels.reshape(-1, 1).float()
    _require_two_classes(labels, "creator model", "creatable", "not-creatable")
    groups = getattr(dataset, "groups", None) or [str(i) for i in range(len(labels))]

    if val_dataset is not None:
        train_tensors = [eeg, labels]
        v_eeg, v_labels = _dataset_tensors(val_dataset, ("eeg", "labels"))
        val_tensors: Optional[List[Tensor]] = [v_eeg, v_labels.reshape(-1, 1).float()]
    else:
        train_tensors, val_tensors = _split_tensors([eeg, labels], groups, labels, validation_split, seed)
    if val_tensors is None:
        logger.warning("Too little data for a validation split; the creator model is trained without one.")

    model = CreatorStateModel(
        channels=int(eeg.shape[1]),
        samples=int(eeg.shape[2]),
        hidden_channels=hidden_channels,
        input_stride=input_stride,
    ).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_class_weight(train_tensors[1]).to(device))

    def step(batch: Sequence[Tensor], training: bool) -> Tuple[Tensor, Tensor]:
        batch_eeg, batch_labels = batch
        if training:
            batch_eeg = augment_eeg(batch_eeg)
        logits = model(batch_eeg)
        return criterion(logits, batch_labels), logits

    history, (logits, targets) = fit(
        model,
        make_loader(train_tensors, batch_size, shuffle=True, seed=seed),
        make_loader(val_tensors, batch_size, shuffle=False, seed=seed) if val_tensors else None,
        step,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        device=device,
        patience=patience,
        label="creator model",
    )
    if val_tensors is not None:
        model.set_temperature(fit_temperature(logits, targets))
    model.set_reference(getattr(dataset, "reference", None))
    history.metrics = classification_metrics(logits, targets, float(model.temperature))
    history.metrics["temperature"] = float(model.temperature)
    history.metrics["evaluated_on"] = "validation" if val_tensors is not None else "training"  # type: ignore[assignment]
    history.metrics["train_windows"] = int(train_tensors[0].shape[0])
    model.eval()
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
    lr: Optional[float] = None,
    hidden_channels: int = 128,
    num_blocks: int = 4,
    dropout: float = 0.1,
    input_stride: int = 8,
    patience: int = 15,
    seed: int = 0,
    parameter_metadata: Path | str | None = None,
    processing_weights: Path | str | None = None,
    feature_config: FeatureConfig = DEFAULT_FEATURE_CONFIG,
    dataset: Optional[BrainwaveProcessingDataset] = None,
) -> Tuple[BrainwaveToProcessingModel, TrainingHistory, Sequence[str]]:
    """Train the regression model mapping EEG segments to Processing controls.

    Returns the model, its history and the names of the 50 parameters it
    predicts.  ``history.metrics["skill"]`` compares the validation error with
    that of always predicting the average parameter set: 0 means the EEG adds
    nothing, 1 would be a perfect prediction.
    """

    if lr is not None:
        learning_rate = lr
    seed_everything(seed)
    device = resolve_device(device)

    if dataset is None:
        dataset = BrainwaveProcessingDataset(
            dataset_root,
            parameter_metadata=parameter_metadata,
            processing_weights=processing_weights,
            processing_device=device,
            feature_config=feature_config,
        )
    eeg, targets = dataset.eeg, dataset.targets
    train_tensors, val_tensors = _split_tensors([eeg, targets], dataset.groups, None, validation_split, seed)
    if val_tensors is None:
        logger.warning("Too little data for a validation split; the parameter model is trained without one.")

    model = create_model(
        input_shape=(int(eeg.shape[1]), int(eeg.shape[2])),
        hidden_channels=hidden_channels,
        num_blocks=num_blocks,
        dropout=dropout,
        input_stride=input_stride,
        codec=dataset.codec,
    ).to(device)
    # Start from the average parameter set, so an untrained or unsure model shows that.
    with torch.no_grad():
        model.head[-1].bias.copy_(train_tensors[1].mean(dim=0))
        model.head[-1].weight.mul_(0.1)

    criterion = nn.SmoothL1Loss()

    def step(batch: Sequence[Tensor], training: bool) -> Tuple[Tensor, Tensor]:
        batch_eeg, batch_targets = batch
        if training:
            batch_eeg = augment_eeg(batch_eeg)
        outputs = model(batch_eeg)
        return criterion(outputs, batch_targets), outputs

    history, (outputs, references) = fit(
        model,
        make_loader(train_tensors, batch_size, shuffle=True, seed=seed),
        make_loader(val_tensors, batch_size, shuffle=False, seed=seed) if val_tensors else None,
        step,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        device=device,
        patience=patience,
        label="parameter model",
    )

    error = float(criterion(outputs, references))
    mean_prediction = train_tensors[1].mean(dim=0, keepdim=True).expand_as(references)
    reference_error = float(criterion(mean_prediction, references))
    history.metrics = {
        "loss": error,
        "mean_predictor_loss": reference_error,
        "skill": 1.0 - error / reference_error if reference_error > 0 else 0.0,
        "train_windows": int(train_tensors[0].shape[0]),
        "artworks": len({sample.sample_id for sample in dataset.samples}),
    }
    history.metrics["evaluated_on"] = "validation" if val_tensors is not None else "training"  # type: ignore[assignment]
    model.eval()
    return model, history, tuple(dataset.parameter_names)


# ---------------------------------------------------------------------------
# Step 6 – Real-time feedback loop
# ---------------------------------------------------------------------------


def _callback_arity(callback: Callable[..., None]) -> int:
    try:
        parameters = [
            p
            for p in inspect.signature(callback).parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
    except (TypeError, ValueError):
        return 2
    if any(p.kind == p.VAR_POSITIONAL for p in inspect.signature(callback).parameters.values()):
        return 2
    return len(parameters)


def run_feedback_loop(
    eeg_source: Optional[Iterable[np.ndarray | Tensor]] = None,
    *,
    creator_model: CreatorStateModel,
    parameter_model: BrainwaveToProcessingModel,
    parameter_names: Optional[Sequence[str]] = None,
    update_callback: Callable[..., None],
    creatable_threshold: float = 0.6,
    update_interval: float = 5.0,
    device: str | torch.device | None = None,
    eeg_supplier: Optional[Callable[[], Optional[np.ndarray | Tensor]]] = None,
    max_updates: Optional[int] = None,
    probability_smoothing: float = 0.6,
    parameter_smoothing: float = 0.6,
    calibration_windows: int = 4,
) -> List[Dict[str, Any]]:
    """Drive the display from a stream of five-second band matrices.

    Give either ``eeg_source`` — any iterable yielding ``(8, 3200)`` band
    matrices, consumed as fast as it produces them — or ``eeg_supplier``, a
    function called once per ``update_interval`` seconds that returns the
    latest matrix (or ``None`` to stop).

    The first ``calibration_windows`` windows only establish the session's
    baseline (20 seconds by default, as in a live session); after that, for
    every window the creator model is evaluated.  While the creator is in
    a creatable state, new parameters are computed from the same window and
    passed to ``update_callback(parameters, is_creatable)``.  When the state is
    left, the callback receives the *last* parameter set with
    ``is_creatable=False``, so the display keeps showing it.  A callback that
    accepts a single argument is only called with fresh parameter sets.

    This function works on prepared band matrices.  To run from the headset
    itself — decoding, artefact rejection, reconnection, recording — use
    :class:`src.realtime.RealtimeSession` or ``workflow_orchestrator --step 6``.

    Returns one record per processed window.
    """

    from .realtime import FeedbackController, LoopSettings

    if eeg_source is None and eeg_supplier is None:
        raise ValueError("Provide eeg_source or eeg_supplier.")
    controller = FeedbackController(
        creator_model,
        parameter_model,
        LoopSettings(
            creatable_threshold=creatable_threshold,
            probability_smoothing=probability_smoothing,
            parameter_smoothing=parameter_smoothing,
        ),
        device=device,
    )
    if parameter_names is not None and tuple(parameter_names) != tuple(controller.parameter_names):
        logger.warning("parameter_names differ from the names stored with the parameter model; using the model's.")

    arity = _callback_arity(update_callback)
    tracker = BaselineTracker(min_seconds=0.0)
    records: List[Dict[str, Any]] = []

    def windows() -> Iterator[np.ndarray | Tensor]:
        if eeg_source is not None:
            yield from eeg_source
            return
        assert eeg_supplier is not None
        deadline = time.monotonic()
        failures = 0
        while True:
            try:
                window = eeg_supplier()
                failures = 0
            except Exception as error:  # keep the display alive through acquisition hiccups
                failures += 1
                logger.error("Reading EEG failed (%d in a row): %s", failures, error)
                window = False  # type: ignore[assignment]
            if window is None:
                return
            if window is not False:
                yield window  # type: ignore[misc]
            deadline += update_interval
            time.sleep(max(0.0, deadline - time.monotonic()))

    try:
        for window in windows():
            array = window.detach().cpu().numpy() if isinstance(window, Tensor) else np.asarray(window)
            array = np.asarray(array, dtype=np.float32)
            if array.ndim != 2 or not np.all(np.isfinite(array)):
                decision = controller.hold("bad_signal")
            else:
                if array.size and float(array.min()) >= 0.0:
                    tracker.update(array)
                if len(records) < calibration_windows:
                    decision = controller.hold("calibrating")
                else:
                    decision = controller.step(normalise_bands(array, controller.baseline_for(tracker.baseline())))

            if decision.parameters is not None:
                if arity >= 2:
                    update_callback(decision.parameters, True)
                else:
                    update_callback(decision.parameters)
            elif arity >= 2 and decision.held_parameters is not None:
                update_callback(decision.held_parameters, False)
            records.append(decision.as_record())
            if max_updates is not None and len(records) >= max_updates:
                break
    except KeyboardInterrupt:
        logger.info("Feedback loop stopped.")
    return records


__all__ = [
    "PipelineError",
    "decode_hex_eeg_segment",
    "iter_brainwave_segments",
    "convert_recording",
    "compute_log_mel_spectrogram",
    "load_audio",
    "ViewerSample",
    "build_viewer_samples",
    "ViewerEnjoymentDataset",
    "ViewerEnjoymentModel",
    "train_viewer_model",
    "evaluate_enjoyment_windows",
    "EnjoymentWindow",
    "split_passages",
    "passage_intervals",
    "PassageScore",
    "score_passages",
    "select_passages",
    "CreatorAnnotation",
    "annotations_from_manual_file",
    "annotations_from_timeline",
    "timeline_from_keystrokes",
    "read_keystroke_log",
    "CreatorStateDataset",
    "CreatorStateModel",
    "train_creator_model",
    "BrainwaveToProcessingModel",
    "train_processing_parameter_model",
    "run_feedback_loop",
    "TrainingHistory",
    "TrainingError",
]
