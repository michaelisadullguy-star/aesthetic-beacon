"""Dataset utilities for brainwave-to-Processing parameter mapping."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .processing_features import compute_processing_parameters


@dataclass(frozen=True)
class SampleMetadata:
    """Metadata describing a single training example."""

    sample_id: str
    brainwave_path: Path
    target: torch.Tensor
    image_path: Path


class BrainwaveProcessingDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """Dataset backed by files on disk.

    The dataset expects the following layout::

        dataset_root/
            brainwaves/
                <sample_id>.npy
            artworks/
                <sample_id>.<ext>

    ``artworks`` should contain an RGB image for each brainwave segment. The file name
    (without the extension) must match the ``sample_id`` of the corresponding
    brainwave recording. Supported artwork extensions are ``.png``, ``.jpg``,
    ``.jpeg`` and ``.bmp``. Brainwave recordings must be stored as NumPy ``.npy``
    tensors with shape ``(8, 3200)`` representing five seconds of activity sampled
    across eight bands. Processing parameters are derived automatically from the artwork using
    :func:`~src.processing_features.compute_processing_parameters` when the dataset is
    instantiated.

    Parameters
    ----------
    root : str or Path
        Directory containing the dataset.
    transform : callable, optional
        Optional transform applied to the raw brainwave tensor after it has been
        converted to ``torch.float32``. Use this to implement normalisation or data
        augmentation strategies.
    target_transform : callable, optional
        Optional transform applied to the target tensor.
    dtype : torch.dtype
        The floating-point data type to use for the inputs.
    expected_channels : int, default 8
        Number of channels (frequency bands) expected per recording.
    expected_samples : int, default 3200
        Number of temporal samples expected per channel. Set to ``0`` to disable the
        length check while keeping channel validation.
    """

    def __init__(
        self,
        root: Path | str,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        target_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        dtype: torch.dtype = torch.float32,
        expected_channels: int = 8,
        expected_samples: int = 3200,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.transform = transform
        self.target_transform = target_transform
        self.dtype = dtype
        self.expected_channels = expected_channels
        self.expected_samples = expected_samples

        self.brainwave_dir = self.root / "brainwaves"
        if not self.brainwave_dir.exists():
            raise FileNotFoundError("Expected a 'brainwaves' directory in the dataset root.")

        self.artwork_dir = self.root / "artworks"
        if not self.artwork_dir.exists():
            raise FileNotFoundError("Expected an 'artworks' directory in the dataset root.")

        supported_artwork_exts = (".png", ".jpg", ".jpeg", ".bmp")
        supported_brainwave_exts = (".npy",)
        self.samples: List[SampleMetadata] = []
        parameter_names: Optional[Sequence[str]] = None

        brainwave_paths: dict[str, Path] = {}
        for ext in supported_brainwave_exts:
            for path in sorted(self.brainwave_dir.glob(f"*{ext}")):
                sample_id = path.stem
                if sample_id in brainwave_paths:
                    raise ValueError(
                        "Duplicate brainwave recordings detected. Ensure only one file per sample."
                    )
                brainwave_paths[sample_id] = path

        if not brainwave_paths:
            raise ValueError(
                "No brainwave files found. Provide NumPy '.npy' tensors shaped as 8x3200 recordings."
            )

        missing_artworks: List[str] = []
        for sample_id in sorted(brainwave_paths.keys()):
            brainwave_path = brainwave_paths[sample_id]
            candidates = sorted(
                path
                for path in self.artwork_dir.glob(f"{sample_id}.*")
                if path.suffix.lower() in supported_artwork_exts
            )
            image_path = candidates[0] if candidates else None

            if image_path is None:
                missing_artworks.append(sample_id)
                continue

            parameter_dict = compute_processing_parameters(image_path)
            if parameter_names is None:
                parameter_names = tuple(parameter_dict.keys())
            else:
                current_names = tuple(parameter_dict.keys())
                if current_names != tuple(parameter_names):
                    raise ValueError(
                        "Inconsistent parameter ordering encountered when analysing artworks."
                    )

            assert parameter_names is not None
            target = torch.tensor(
                [float(parameter_dict[name]) for name in parameter_names], dtype=torch.float32
            )
            self.samples.append(
                SampleMetadata(
                    sample_id=sample_id,
                    brainwave_path=brainwave_path,
                    target=target,
                    image_path=image_path,
                )
            )

        if missing_artworks:
            missing_formatted = ", ".join(missing_artworks)
            raise FileNotFoundError(
                f"Missing artwork images for the following samples: {missing_formatted}"
            )

        if not self.samples:
            raise ValueError(
                "The dataset is empty after processing artworks. Ensure each brainwave file has a matching image."
            )

        self.parameter_names = parameter_names or ()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        tensor = torch.from_numpy(
            load_brainwave_array(
                sample.brainwave_path,
                expected_channels=self.expected_channels,
                expected_samples=self.expected_samples,
            )
        ).to(self.dtype)

        if self.transform is not None:
            tensor = self.transform(tensor)

        target = sample.target.clone()
        if self.target_transform is not None:
            target = self.target_transform(target)

        return tensor, target

    def parameter_size(self) -> int:
        """Return the dimensionality of the Processing parameter vector."""

        return len(self.parameter_names)

    def describe_parameters(self) -> str:
        """Return a human-readable description of the parameter ordering."""

        return ", ".join(self.parameter_names)


def load_brainwave_array(
    path: Path,
    expected_channels: int = 8,
    expected_samples: int = 3200,
) -> np.ndarray:
    """Load a brainwave recording stored as a NumPy ``.npy`` tensor."""

    if path.suffix.lower() != ".npy":
        raise ValueError(
            f"Unsupported brainwave file type: {path.suffix}. Expected a NumPy '.npy' tensor."
        )

    array = np.load(path, allow_pickle=False)
    array = np.asarray(array, dtype=np.float32)

    if array.ndim != 2:
        raise ValueError(
            "Brainwave tensors must be 2-dimensional with shape [channels, samples]."
        )

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

    mean = features.mean(dim=1, keepdim=True)
    std = features.std(dim=1, keepdim=True)
    std = std.masked_fill(std == 0, 1.0)
    return (features - mean) / std


class RandomTimeCrop:
    """Randomly crop the temporal dimension to a fixed length."""

    def __init__(self, target_length: int) -> None:
        if target_length <= 0:
            raise ValueError("target_length must be positive")
        self.target_length = target_length

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        channels, samples = tensor.shape
        if samples <= self.target_length:
            return tensor

        start = np.random.randint(0, samples - self.target_length + 1)
        return tensor[:, start : start + self.target_length]


def collate_pad_batch(batch: Sequence[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collate function that pads variable-length sequences.

    This allows for variable-length temporal recordings when using ``DataLoader``.
    The padding is applied to the end of the sequence.
    """

    features, targets = zip(*batch)
    max_length = max(feature.shape[1] for feature in features)
    padded = []
    for feature in features:
        length = feature.shape[1]
        if length < max_length:
            pad_width = (0, max_length - length)
            padded_feature = torch.nn.functional.pad(feature, pad_width)
        else:
            padded_feature = feature
        padded.append(padded_feature)

    stacked_features = torch.stack(padded)
    stacked_targets = torch.stack(targets)
    return stacked_features, stacked_targets

