"""Dataset utilities for brainwave-to-Processing parameter mapping."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

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
            metadata.json  (optional)

    ``artworks`` should contain an RGB image for each brainwave segment. The file name
    (without the extension) must match the ``sample_id`` of the corresponding
    brainwave recording. Supported artwork extensions are ``.png``, ``.jpg``,
    ``.jpeg`` and ``.bmp``. Brainwave recordings must be stored as NumPy ``.npy``
    tensors with shape ``(8, 3200)`` representing five seconds of activity sampled
    across eight bands.

    Processing parameters can be provided in two ways:

    1. **Ground Truth (Recommended)**: Provide a ``metadata.json`` file with ground truth
       parameter annotations for each artwork. This creates a direct static mapping from
       images to parameters, avoiding dynamic CNN-based generation.
    2. **CNN-based (Legacy)**: If no metadata is provided, parameters are derived
       automatically from artworks using the trained CNN model via
       :func:`~src.processing_features.compute_processing_parameters`.

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
    parameter_metadata : Path or str, optional
        Path to a JSON file containing ground truth parameter annotations for each
        artwork. If provided, parameters are loaded from this file instead of being
        computed via the CNN model. The JSON should map sample IDs to parameter dicts.
    processing_weights : Path or str, optional
        Path to the trained artwork-to-Processing parameter model checkpoint used
        to derive targets from the artwork images. Only required if parameter_metadata
        is not provided.
    processing_device : str or torch.device, optional
        Device on which the preprocessing model is executed. Defaults to CPU.
        Only used when parameter_metadata is not provided.
    processing_resize_long_edge : int, optional
        Optional resize applied before preprocessing. ``None`` disables resizing.
        Only used when parameter_metadata is not provided.
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
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.transform = transform
        self.target_transform = target_transform
        self.dtype = dtype
        self.expected_channels = expected_channels
        self.expected_samples = expected_samples

        # Determine parameter source: ground truth metadata or CNN inference
        self.parameter_metadata_path = None
        self.use_metadata = False
        self.ground_truth_parameters: Dict[str, Dict[str, float]] = {}

        if parameter_metadata is not None:
            self.parameter_metadata_path = Path(parameter_metadata).expanduser().resolve()
            if not self.parameter_metadata_path.exists():
                raise FileNotFoundError(
                    f"Parameter metadata not found at '{self.parameter_metadata_path}'."
                )
            self.use_metadata = True
            # Load ground truth parameters
            with self.parameter_metadata_path.open("r", encoding="utf8") as f:
                metadata_content = json.load(f)

            # Parse metadata format (support both dict and list formats)
            if isinstance(metadata_content, dict):
                self.ground_truth_parameters = {
                    str(k): v for k, v in metadata_content.items()
                }
            elif isinstance(metadata_content, list):
                for entry in metadata_content:
                    if not isinstance(entry, dict):
                        raise ValueError("List metadata entries must be objects.")
                    sample_id = entry.get("id") or entry.get("image")
                    if not sample_id:
                        raise ValueError("Metadata entries must include 'id' or 'image' field.")
                    params = entry.get("parameters")
                    if not isinstance(params, dict):
                        raise ValueError("Each entry must have a 'parameters' dict.")
                    self.ground_truth_parameters[str(sample_id)] = params
            else:
                raise ValueError("Metadata must be a JSON object or array.")
        else:
            # Fall back to CNN-based parameter generation
            if processing_weights is None:
                raise ValueError(
                    "Either parameter_metadata or processing_weights must be provided."
                )
            self.processing_weights = Path(processing_weights).expanduser().resolve()
            if not self.processing_weights.exists():
                raise FileNotFoundError(
                    f"Processing weights not found at '{self.processing_weights}'. "
                    "Train the artwork model first or provide parameter_metadata."
                )
            self.processing_device = processing_device
            self.processing_resize_long_edge = processing_resize_long_edge

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
        missing_parameters: List[str] = []

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

            # Get parameters: from ground truth metadata or CNN inference
            if self.use_metadata:
                # Use ground truth parameters from metadata
                if sample_id not in self.ground_truth_parameters:
                    missing_parameters.append(sample_id)
                    continue
                parameter_dict = self.ground_truth_parameters[sample_id]
            else:
                # Fall back to CNN-based inference
                parameter_dict = compute_processing_parameters(
                    image_path,
                    weights_path=self.processing_weights,
                    resize_long_edge=self.processing_resize_long_edge,
                    device=self.processing_device,
                )

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

        if missing_parameters:
            missing_formatted = ", ".join(missing_parameters)
            raise FileNotFoundError(
                f"Missing ground truth parameters in metadata for the following samples: {missing_formatted}"
            )

        if not self.samples:
            raise ValueError(
                "The dataset is empty after processing artworks. Ensure each brainwave file has a matching image."
            )

        self.parameter_names = parameter_names or ()

        # Log which parameter source was used
        if self.use_metadata:
            print(f"Loaded {len(self.samples)} samples using ground truth parameters from metadata.")
        else:
            print(f"Loaded {len(self.samples)} samples using CNN-generated parameters.")

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

    array = np.load(path)
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

