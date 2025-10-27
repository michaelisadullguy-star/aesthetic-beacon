"""Image-to-Processing parameter inference utilities.

This module provides the neural network used to translate artwork images into
Processing sketch parameters. The weights are trained separately via
``train_processing_features.py`` and serialized to disk. During dataset
construction the :func:`compute_processing_parameters` helper loads the trained
predictor and performs a forward pass to obtain the parameter values.
"""
from __future__ import annotations

from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from typing import OrderedDict as OrderedDictType, Sequence

import numpy as np
import torch
from PIL import Image
from torch import nn
import torch.nn.functional as F


def _ensure_image(image: Image.Image | np.ndarray | Path | str) -> Image.Image:
    """Load an image input and ensure it is in RGB mode."""

    if isinstance(image, Image.Image):
        return image.convert("RGB")

    if isinstance(image, (str, Path)):
        with Image.open(image) as handle:
            return handle.convert("RGB")

    if isinstance(image, np.ndarray):
        array = np.asarray(image)
        if array.ndim == 2:
            array = np.stack([array] * 3, axis=-1)
        return Image.fromarray(array.astype(np.uint8), mode="RGB")

    raise TypeError(f"Unsupported image type: {type(image)!r}")


def prepare_image_tensor(
    image: Image.Image | np.ndarray | Path | str,
    *,
    resize_long_edge: int | None = 512,
) -> torch.Tensor:
    """Convert an input image into a normalised ``torch.Tensor``.

    The returned tensor is shaped ``(3, H, W)`` with floating point values
    normalised to the range ``[0, 1]``. Optionally the image is resized such
    that its longest edge matches ``resize_long_edge`` while preserving aspect
    ratio.
    """

    pil_image = _ensure_image(image)
    if resize_long_edge is not None:
        width, height = pil_image.size
        long_edge = max(width, height)
        if long_edge > resize_long_edge:
            scale = resize_long_edge / long_edge
            new_size = (int(round(width * scale)), int(round(height * scale)))
            pil_image = pil_image.resize(new_size, Image.Resampling.BICUBIC)

    array = np.asarray(pil_image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return tensor


class ProcessingParameterPredictor(nn.Module):
    """CNN regressor that maps artwork images to Processing parameters."""

    def __init__(
        self,
        parameter_count: int,
        *,
        base_channels: int = 32,
        mlp_hidden: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if parameter_count <= 0:
            raise ValueError("parameter_count must be positive")

        self.parameter_count = parameter_count
        self.base_channels = base_channels
        self.mlp_hidden = mlp_hidden
        self.dropout_prob = dropout

        layers: list[nn.Module] = []
        in_channels = 3
        channels = base_channels
        max_channels = base_channels * 8
        for _ in range(3):
            layers.append(
                nn.Conv2d(in_channels, channels, kernel_size=3, stride=2, padding=1)
            )
            layers.append(nn.BatchNorm2d(channels))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Conv2d(channels, channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = channels
            channels = min(channels * 2, max_channels)

        self.encoder = nn.Sequential(*layers)
        self.feature_dim = in_channels

        # RGB means, RGB standard deviations, RGB minima, RGB maxima, luminance
        # mean and luminance standard deviation.
        self._stats_dim = 3 + 3 + 3 + 3 + 1 + 1

        self.head = nn.Sequential(
            nn.Linear(self.feature_dim + self._stats_dim, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, parameter_count),
        )

    def _global_stats(self, images: torch.Tensor) -> torch.Tensor:
        means = images.mean(dim=(2, 3))
        stds = images.std(dim=(2, 3))
        minima = images.amin(dim=(2, 3))
        maxima = images.amax(dim=(2, 3))
        luminance = 0.2989 * images[:, 0] + 0.5870 * images[:, 1] + 0.1140 * images[:, 2]
        luminance_mean = luminance.mean(dim=(1, 2), keepdim=True)
        luminance_std = luminance.std(dim=(1, 2), keepdim=True)
        return torch.cat(
            [
                means,
                stds,
                minima,
                maxima,
                luminance_mean,
                luminance_std,
            ],
            dim=1,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError("images must be a 4D tensor shaped as (N, 3, H, W)")

        features = self.encoder(images)
        pooled = F.adaptive_avg_pool2d(features, 1).flatten(1)
        stats = self._global_stats(images)
        combined = torch.cat([pooled, stats], dim=1)
        return self.head(combined)

    def config(self) -> dict[str, float | int]:
        """Return the configuration needed to recreate the module."""

        return {
            "base_channels": self.base_channels,
            "mlp_hidden": self.mlp_hidden,
            "dropout": self.dropout_prob,
        }


@lru_cache(maxsize=8)
def _load_predictor(weights_path: str) -> tuple[
    ProcessingParameterPredictor,
    Sequence[str],
    dict[str, Sequence[float]] | None,
    dict[str, float | int],
]:
    checkpoint = torch.load(weights_path, map_location="cpu")
    parameter_names = tuple(checkpoint.get("parameter_names", ()))
    if not parameter_names:
        raise ValueError(
            "Checkpoint is missing 'parameter_names'. Train using train_processing_features.py."
        )

    model_config = checkpoint.get("model_config", {})
    model = ProcessingParameterPredictor(
        parameter_count=len(parameter_names),
        base_channels=model_config.get("base_channels", 32),
        mlp_hidden=model_config.get("mlp_hidden", 128),
        dropout=model_config.get("dropout", 0.1),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    normaliser = checkpoint.get("target_normaliser")
    return model, parameter_names, normaliser, model_config


def compute_processing_parameters(
    image: Image.Image | np.ndarray | Path | str,
    *,
    weights_path: Path | str,
    resize_long_edge: int | None = 512,
    device: str | torch.device | None = None,
) -> OrderedDictType[str, float]:
    """Predict Processing parameters for ``image`` using a trained model.

    Parameters
    ----------
    image:
        Artwork to analyse. ``str`` and :class:`pathlib.Path` inputs will be
        opened with Pillow.
    weights_path:
        Path to a checkpoint produced by ``train_processing_features.py``.
    resize_long_edge:
        Optionally resize the image before inference. ``None`` preserves the
        original resolution.
    device:
        Device identifier used for inference. Defaults to CPU if not provided.
    """

    if weights_path is None:
        raise ValueError("weights_path must point to trained Processing feature weights.")

    resolved_path = Path(weights_path).expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Processing feature weights not found at '{resolved_path}'. Run train_processing_features.py first."
        )

    model, parameter_names, normaliser, model_config = _load_predictor(str(resolved_path))

    tensor = prepare_image_tensor(image, resize_long_edge=resize_long_edge).unsqueeze(0)
    inference_device = torch.device(device) if device is not None else torch.device("cpu")
    if inference_device.type == "cpu":
        working_model = model.to(inference_device)
    else:
        working_model = ProcessingParameterPredictor(
            parameter_count=len(parameter_names),
            base_channels=int(model_config.get("base_channels", 32)),
            mlp_hidden=int(model_config.get("mlp_hidden", 128)),
            dropout=float(model_config.get("dropout", 0.1)),
        ).to(inference_device)
        working_model.load_state_dict(model.state_dict())
    tensor = tensor.to(inference_device)

    with torch.no_grad():
        prediction = working_model(tensor.to(inference_device)).squeeze(0).cpu()

    if normaliser:
        mean = torch.tensor(normaliser.get("mean", []), dtype=torch.float32)
        std = torch.tensor(normaliser.get("std", []), dtype=torch.float32)
        if mean.numel() != prediction.numel() or std.numel() != prediction.numel():
            raise ValueError("Normaliser statistics are incompatible with the prediction size.")
        prediction = prediction * std + mean

    parameters = OrderedDict((name, float(value)) for name, value in zip(parameter_names, prediction))
    return parameters  # type: ignore[return-value]


__all__ = [
    "ProcessingParameterPredictor",
    "compute_processing_parameters",
    "prepare_image_tensor",
]
