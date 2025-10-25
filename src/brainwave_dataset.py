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
    ``.npy`` file. Supported extensions are ``.png``, ``.jpg``, ``.jpeg`` and ``.bmp``.
    Processing parameters are derived automatically from the artwork using
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
    """

    def __init__(
        self,
        root: Path | str,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        target_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.transform = transform
        self.target_transform = target_transform
        self.dtype = dtype

        self.brainwave_dir = self.root / "brainwaves"
        if not self.brainwave_dir.exists():
            raise FileNotFoundError("Expected a 'brainwaves' directory in the dataset root.")

        self.artwork_dir = self.root / "artworks"
        if not self.artwork_dir.exists():
            raise FileNotFoundError("Expected an 'artworks' directory in the dataset root.")

        supported_exts = (".png", ".jpg", ".jpeg", ".bmp")
        self.samples = []
        parameter_names: Optional[Sequence[str]] = None

        brainwave_files = sorted(self.brainwave_dir.glob("*.npy"))
        if not brainwave_files:
            raise ValueError("No brainwave files found. Ensure the dataset contains *.npy files.")

        missing_artworks: List[str] = []
        for brainwave_path in brainwave_files:
            sample_id = brainwave_path.stem
            candidates = sorted(
                path
                for path in self.artwork_dir.glob(f"{sample_id}.*")
                if path.suffix.lower() in supported_exts
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
                SampleMetadata(sample_id=sample_id, target=target, image_path=image_path)
            )

        if missing_artworks:
            missing_formatted = ", ".join(missing_artworks)
            raise FileNotFoundError(
                f"Missing artwork images for the following samples: {missing_formatted}"
            )

        if not self.samples:
            raise ValueError(
                "The dataset is empty after processing artworks. Ensure images match *.npy files."
            )

        self.parameter_names = parameter_names or ()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        brainwave_path = self.brainwave_dir / f"{sample.sample_id}.npy"
        if not brainwave_path.exists():
            raise FileNotFoundError(
                f"Brainwave file {brainwave_path!s} is missing for sample "
                f"{sample.sample_id}."
            )

        array = np.load(brainwave_path)
        tensor = torch.from_numpy(array).to(self.dtype)
        if tensor.ndim != 2:
            raise ValueError(
                "Brainwave tensors must be 2-dimensional with shape [channels, samples]."
            )

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

