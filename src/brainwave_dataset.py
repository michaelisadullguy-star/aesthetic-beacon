"""Dataset for Step 5: EEG recorded while viewing artworks → Processing parameters."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .eeg_features import (
    DEFAULT_FEATURE_CONFIG,
    FeatureConfig,
    Recording,
    RecordingError,
    find_recordings,
    load_recording,
    normalise_bands,
    pool_baselines,
)
from .parameters import PARAMETER_NAMES, ParameterCodec, canonicalise_parameters
from .processing_features import SUPPORTED_IMAGE_SUFFIXES, resolve_artwork_parameters

logger = logging.getLogger(__name__)

#: Separates the artwork name from a participant label in a recording's file name.
PARTICIPANT_SEPARATOR = "__"


@dataclass(frozen=True)
class SampleMetadata:
    """Metadata describing a single training example."""

    sample_id: str
    brainwave_path: Path
    target: torch.Tensor
    image_path: Path
    start_sample: int = 0


def load_parameter_metadata(path: Path | str) -> Dict[str, Dict[str, object]]:
    """Read curated artwork parameters keyed by artwork name (without extension).

    Accepts an object mapping artwork names to parameter objects, or a list of
    ``{"id" | "image": ..., "parameters": {...}}`` entries.
    """

    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Parameter metadata not found at '{path}'.")
    try:
        with path.open("r", encoding="utf8") as handle:
            content = json.load(handle)
    except json.JSONDecodeError as error:
        raise ValueError(f"{path} is not valid JSON: {error}") from error

    entries: Dict[str, Dict[str, object]] = {}
    if isinstance(content, dict):
        items = list(content.items())
    elif isinstance(content, list):
        items = []
        for entry in content:
            if not isinstance(entry, dict):
                raise ValueError(f"{path}: list entries must be objects.")
            sample_id = entry.get("id") or entry.get("image")
            if not sample_id:
                raise ValueError(f"{path}: entries must include an 'id' or 'image' field.")
            items.append((sample_id, entry.get("parameters")))
    else:
        raise ValueError(f"{path} must contain a JSON object or array.")

    for key, parameters in items:
        if not isinstance(parameters, dict):
            raise ValueError(f"{path}: the parameters of '{key}' must be an object.")
        name = str(key)
        if Path(name).suffix.lower() in SUPPORTED_IMAGE_SUFFIXES:
            name = Path(name).stem
        entries[name] = parameters
    return entries


def _match_artwork(stem: str, artwork_ids: Sequence[str]) -> Optional[str]:
    """Artwork a recording belongs to: ``<artwork>`` or ``<artwork>__<participant>``."""

    if stem in artwork_ids:
        return stem
    candidates = [a for a in artwork_ids if stem.startswith(a + PARTICIPANT_SEPARATOR)]
    return max(candidates, key=len) if candidates else None


class BrainwaveProcessingDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """EEG windows paired with the Processing parameters of the artwork being viewed.

    Expected layout::

        dataset_root/
            artworks/
                <artwork>.png | .jpg | .jpeg | .bmp
            brainwaves/
                <artwork>.txt                      one recording per artwork, or
                <artwork>__<participant>.txt       several participants side by side, or
                <participant>/<artwork>.txt        one folder per participant (recommended)
            metadata.json                          optional curated parameters

    Recordings may be headset hex dumps (``.txt`` / ``.hex``) of any length of
    at least five seconds, or ``.npy`` band matrices shaped ``(8, samples)``.
    From longer recordings the middle five seconds are used — the start is
    disturbed by the eyes opening and the end by the participant anticipating
    the stop — plus a few windows shifted slightly around the middle.

    Recordings in the same folder are assumed to come from one sitting and
    share a baseline, which preserves the differences *between* artworks.

    Each item is ``(eeg, target)`` where ``eeg`` is the normalised
    ``(8, 3200)`` band matrix and ``target`` the encoded parameter vector (see
    :class:`~src.parameters.ParameterCodec`).

    Parameters
    ----------
    root:
        Dataset directory.
    transform, target_transform:
        Optional callables applied to the EEG tensor / the target tensor.
    expected_channels, expected_samples:
        Shape every EEG window must have.
    parameter_metadata:
        Curated parameters; defaults to ``root/metadata.json`` when present.
    processing_weights:
        Trained artwork CNN used for artworks without curated parameters.
        Without it those artworks are analysed directly.
    windows_per_recording:
        How many windows to cut around the middle of each recording.
    """

    def __init__(
        self,
        root: Path | str,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        target_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        dtype: torch.dtype = torch.float32,
        expected_channels: int = 8,
        expected_samples: int = 3200,
        *,
        parameter_metadata: Path | str | None = None,
        processing_weights: Path | str | None = None,
        processing_device: str | torch.device | None = None,
        processing_resize_long_edge: int | None = 512,
        feature_config: FeatureConfig = DEFAULT_FEATURE_CONFIG,
        windows_per_recording: int = 5,
        window_jitter_seconds: float = 1.0,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.transform = transform
        self.target_transform = target_transform
        self.dtype = dtype
        self.expected_channels = expected_channels
        self.expected_samples = expected_samples
        self.feature_config = feature_config
        self.codec = ParameterCodec()
        self.parameter_names: Tuple[str, ...] = PARAMETER_NAMES

        if feature_config.num_bands != expected_channels or feature_config.segment_samples != expected_samples:
            raise ValueError(
                f"The feature configuration yields {feature_config.num_bands}×"
                f"{feature_config.segment_samples} windows but {expected_channels}×{expected_samples} "
                "were requested."
            )

        self.brainwave_dir = self.root / "brainwaves"
        if not self.brainwave_dir.is_dir():
            raise FileNotFoundError(f"Expected a 'brainwaves' directory in {self.root}.")
        self.artwork_dir = self.root / "artworks"
        if not self.artwork_dir.is_dir():
            raise FileNotFoundError(f"Expected an 'artworks' directory in {self.root}.")

        artworks: Dict[str, Path] = {}
        for path in sorted(self.artwork_dir.iterdir()):
            if path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES:
                artworks.setdefault(path.stem, path)
        if not artworks:
            raise FileNotFoundError(f"No artwork images found in {self.artwork_dir}.")

        recordings = find_recordings(self.brainwave_dir, recursive=True)
        if not recordings:
            raise ValueError(
                f"No brainwave recordings found in {self.brainwave_dir}. "
                "Provide headset hex dumps (.txt/.hex) or 8×3200 '.npy' band matrices."
            )

        # --- parameter targets ---------------------------------------------------
        metadata_path = Path(parameter_metadata) if parameter_metadata else self.root / "metadata.json"
        curated: Dict[str, Dict[str, object]] = {}
        if parameter_metadata is not None or metadata_path.is_file():
            curated = load_parameter_metadata(metadata_path)
        self.parameter_metadata_path = metadata_path if curated else None
        self.use_metadata = bool(curated)

        weights = None
        if processing_weights is not None:
            weights = Path(processing_weights).expanduser()
            if not weights.is_file():
                raise FileNotFoundError(
                    f"Processing weights not found at '{weights}'. Train the artwork model first, "
                    "or omit the option to analyse the artworks directly."
                )

        self.artwork_parameters: Dict[str, Dict[str, float]] = {}
        self.parameter_sources: Dict[str, str] = {}
        ignored_keys: set = set()
        for artwork_id, image_path in artworks.items():
            entry = curated.get(artwork_id)
            if entry is not None:
                _, ignored = canonicalise_parameters(entry)
                ignored_keys.update(ignored)
            try:
                parameters, source = resolve_artwork_parameters(
                    image_path,
                    curated=entry,
                    weights_path=weights,
                    resize_long_edge=processing_resize_long_edge,
                    device=processing_device,
                )
            except Exception as error:
                raise ValueError(f"Could not derive parameters for artwork '{image_path}': {error}") from error
            self.artwork_parameters[artwork_id] = dict(parameters)
            self.parameter_sources[artwork_id] = source
        if curated:
            missing = sorted(set(artworks) - set(curated))
            if missing:
                logger.warning(
                    "metadata.json has no entry for %s; their parameters were derived automatically.",
                    ", ".join(missing),
                )
        unlearned = sorted(key for key in ignored_keys if key)
        if unlearned:
            logger.info(
                "Ignoring metadata keys that are not EEG-driven parameters: %s", ", ".join(unlearned)
            )

        # --- recordings ---------------------------------------------------------
        unmatched: List[str] = []
        loaded: List[Tuple[Path, str, Recording]] = []
        artwork_ids = list(artworks)
        for path in recordings:
            artwork_id = _match_artwork(path.stem, artwork_ids)
            if artwork_id is None:
                unmatched.append(str(path.relative_to(self.brainwave_dir)))
                continue
            loaded.append((path, artwork_id, load_recording(path, feature_config)))
        if unmatched:
            raise FileNotFoundError(
                "No artwork image matches these recordings (name them '<artwork>' or "
                f"'<artwork>{PARTICIPANT_SEPARATOR}<participant>'): " + ", ".join(unmatched)
            )

        by_folder: Dict[Path, List[Recording]] = {}
        for path, _artwork, recording in loaded:
            by_folder.setdefault(path.parent, []).append(recording)
        baselines = {
            folder: pool_baselines([r.band_matrix() for r in members]) if len(members) > 1 else None
            for folder, members in by_folder.items()
        }

        self.samples: List[SampleMetadata] = []
        self.groups: List[str] = []
        eeg_windows: List[np.ndarray] = []
        skipped: List[str] = []
        segment = feature_config.segment_samples
        jitter = int(round(window_jitter_seconds * feature_config.sample_rate))

        for path, artwork_id, recording in loaded:
            if recording.n_samples < segment:
                raise RecordingError(
                    f"{path} lasts {recording.duration:.1f} s; at least "
                    f"{feature_config.segment_seconds:.0f} s are required."
                )
            centre = (recording.n_samples - segment) // 2
            slack = min(jitter, centre)
            if slack == 0 or windows_per_recording <= 1:
                starts = [centre]
            else:
                starts = sorted(
                    {int(round(centre + offset)) for offset in np.linspace(-slack, slack, windows_per_recording)}
                )
            target = torch.from_numpy(self.codec.encode(self.artwork_parameters[artwork_id]))
            baseline = baselines[path.parent]
            used = 0
            for start in starts:
                if not recording.window_quality(start).ok:
                    continue
                eeg_windows.append(normalise_bands(recording.window(start), baseline))
                self.samples.append(
                    SampleMetadata(
                        sample_id=artwork_id,
                        brainwave_path=path,
                        target=target,
                        image_path=artworks[artwork_id],
                        start_sample=start,
                    )
                )
                self.groups.append(str(path))
                used += 1
            if used == 0:
                skipped.append(path.name)

        if skipped:
            logger.warning(
                "Skipped %d recording(s) with poor signal quality: %s",
                len(skipped),
                ", ".join(skipped[:8]) + (" …" if len(skipped) > 8 else ""),
            )
        if not self.samples:
            raise ValueError(
                "No usable EEG windows remain after the signal-quality check. "
                "Inspect the recordings (headset contact, clipping) and try again."
            )

        self.eeg = torch.from_numpy(np.stack(eeg_windows)).to(dtype)
        self.targets = torch.stack([sample.target for sample in self.samples])

        sources = sorted(set(self.parameter_sources.values()))
        logger.info(
            "Loaded %d EEG windows from %d recordings of %d artworks (parameters from: %s).",
            len(self.samples),
            len(loaded) - len(skipped),
            len({sample.sample_id for sample in self.samples}),
            ", ".join(sources),
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        tensor = self.eeg[index]
        if self.transform is not None:
            tensor = self.transform(tensor)
        target = self.targets[index].clone()
        if self.target_transform is not None:
            target = self.target_transform(target)
        return tensor, target

    def parameter_size(self) -> int:
        """Return the dimensionality of the encoded parameter vector."""

        return self.codec.dim

    def describe_parameters(self) -> str:
        """Return a human-readable description of the parameter ordering."""

        return ", ".join(self.parameter_names)


def load_brainwave_array(
    path: Path,
    expected_channels: int = 8,
    expected_samples: int = 3200,
) -> np.ndarray:
    """Load a band matrix stored as a NumPy ``.npy`` tensor."""

    path = Path(path)
    if path.suffix.lower() != ".npy":
        raise ValueError(
            f"Unsupported brainwave file type: {path.suffix}. Expected a NumPy '.npy' tensor."
        )

    array = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("Brainwave tensors must be 2-dimensional with shape [channels, samples].")

    if expected_channels > 0:
        if array.shape[0] == expected_channels:
            pass
        elif array.shape[1] == expected_channels:
            array = np.swapaxes(array, 0, 1)
        else:
            raise ValueError(
                f"Brainwave tensor has {array.shape[0]} channels, expected {expected_channels}."
            )

    if expected_samples > 0 and array.shape[1] != expected_samples:
        raise ValueError(
            f"Brainwave tensor has {array.shape[1]} samples per channel, expected {expected_samples}."
        )

    return array


def z_score_normalise(features: torch.Tensor) -> torch.Tensor:
    """Apply per-channel z-score normalisation."""

    mean = features.mean(dim=-1, keepdim=True)
    std = features.std(dim=-1, keepdim=True)
    std = std.masked_fill(std == 0, 1.0)
    return (features - mean) / std


class RandomTimeCrop:
    """Randomly crop the temporal dimension to a fixed length."""

    def __init__(self, target_length: int) -> None:
        if target_length <= 0:
            raise ValueError("target_length must be positive")
        self.target_length = target_length

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        samples = tensor.shape[-1]
        if samples <= self.target_length:
            return tensor
        start = int(torch.randint(0, samples - self.target_length + 1, (1,)).item())
        return tensor[..., start : start + self.target_length]


def collate_pad_batch(
    batch: Sequence[Tuple[torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collate function that pads variable-length sequences at the end."""

    features, targets = zip(*batch)
    max_length = max(feature.shape[1] for feature in features)
    padded = [
        torch.nn.functional.pad(feature, (0, max_length - feature.shape[1]))
        if feature.shape[1] < max_length
        else feature
        for feature in features
    ]
    return torch.stack(padded), torch.stack(targets)


__all__ = [
    "BrainwaveProcessingDataset",
    "PARTICIPANT_SEPARATOR",
    "RandomTimeCrop",
    "SampleMetadata",
    "collate_pad_batch",
    "load_brainwave_array",
    "load_parameter_metadata",
    "z_score_normalise",
]
