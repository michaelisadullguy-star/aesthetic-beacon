"""Dataset utilities for brainwave-to-Processing parameter mapping."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SampleMetadata:
    """Metadata describing a single training example."""

    sample_id: str
    target: torch.Tensor


class BrainwaveProcessingDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """Dataset backed by files on disk.

    The dataset expects the following layout::

        dataset_root/
            metadata.json
            brainwaves/
                <sample_id>.npy

    ``metadata.json`` should contain a mapping from ``sample_id`` to a dictionary of
    Processing parameters. Each entry in the dictionary must be numeric. The order of
    the parameters will be inferred from the keys in the first sample, so it is
    recommended to keep the keys consistent across all samples.

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

        metadata_path = self.root / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Could not find metadata.json at {metadata_path!s}. "
                "Ensure the dataset is prepared correctly."
            )

        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata_raw: Dict[str, Dict[str, float]] = json.load(handle)

        if not metadata_raw:
            raise ValueError("metadata.json is empty; at least one sample is required.")

        # Determine the canonical order of Processing parameters using the first entry.
        first_key = next(iter(metadata_raw))
        self.parameter_names: Sequence[str] = tuple(metadata_raw[first_key].keys())

        missing_parameters: Dict[str, List[str]] = {}
        self.samples: List[SampleMetadata] = []
        for sample_id, param_dict in metadata_raw.items():
            missing = [p for p in self.parameter_names if p not in param_dict]
            if missing:
                missing_parameters[sample_id] = missing
                continue

            targets = torch.tensor(
                [float(param_dict[p]) for p in self.parameter_names], dtype=torch.float32
            )
            self.samples.append(SampleMetadata(sample_id=sample_id, target=targets))

        if missing_parameters:
            formatted = ", ".join(
                f"{sample_id}: missing {', '.join(params)}"
                for sample_id, params in missing_parameters.items()
            )
            raise ValueError(
                "Inconsistent parameter definitions detected in metadata.json: "
                f"{formatted}"
            )

        self.brainwave_dir = self.root / "brainwaves"
        if not self.brainwave_dir.exists():
            raise FileNotFoundError(
                "Expected a 'brainwaves' directory alongside metadata.json."
            )

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

