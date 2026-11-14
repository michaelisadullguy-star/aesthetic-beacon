"""Turning an artwork into Processing parameters.

Step 5 needs, for every image a participant looked at, the parameter set that
makes the Processing sketch echo that image's colours and texture.  There are
three ways to obtain it, tried in this order by :func:`resolve_artwork_parameters`:

1. **Curated values** from a ``metadata.json`` next to the artworks.
2. **A trained CNN** (:class:`ProcessingParameterPredictor`, trained with
   ``train_processing_features.py`` on images with curated parameters).
3. **Direct image analysis** (:func:`analyse_image`) — always available, needs
   no training data.  The palette comes from clustering the image's colours;
   texture parameters come from measured contrast, edge density, feature size
   and edge orientation.  The individual mappings are design choices, kept
   monotonic and bounded so that visibly different images yield clearly
   different parameter sets.
"""
from __future__ import annotations

import math
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from typing import Dict, Mapping, Optional, OrderedDict as OrderedDictType, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from .parameters import (
    PALETTE_SIZE,
    PARAMETER_NAMES,
    SPEC_BY_NAME,
    ParameterCodec,
    canonicalise_parameters,
)

SUPPORTED_IMAGE_SUFFIXES: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp")


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
        if array.dtype != np.uint8:
            scale = 255.0 if float(array.max(initial=0.0)) <= 1.0 else 1.0
            array = np.clip(array * scale, 0, 255).astype(np.uint8)
        return Image.fromarray(array[..., :3], mode="RGB")

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
            new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
            pil_image = pil_image.resize(new_size, Image.Resampling.BICUBIC)

    array = np.asarray(pil_image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


# ---------------------------------------------------------------------------
# Direct image analysis
# ---------------------------------------------------------------------------


def _rgb_to_hsb(rgb: np.ndarray) -> np.ndarray:
    """``(..., 3)`` RGB in [0, 1] → hue [0, 360), saturation and brightness in [0, 1]."""

    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maximum = rgb.max(axis=-1)
    minimum = rgb.min(axis=-1)
    delta = maximum - minimum
    safe = np.where(delta > 0, delta, 1.0)
    hue = np.where(
        maximum == r,
        ((g - b) / safe) % 6.0,
        np.where(maximum == g, (b - r) / safe + 2.0, (r - g) / safe + 4.0),
    )
    hue = np.where(delta > 0, hue * 60.0, 0.0) % 360.0
    saturation = np.where(maximum > 0, delta / np.where(maximum > 0, maximum, 1.0), 0.0)
    return np.stack([hue, saturation, maximum], axis=-1)


def _cluster_colours(hsb: np.ndarray, clusters: int, iterations: int = 15) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministic k-means over colours; returns centres (HSB) and their pixel shares."""

    angle = np.deg2rad(hsb[:, 0])
    points = np.stack(
        [hsb[:, 1] * np.cos(angle), hsb[:, 1] * np.sin(angle), hsb[:, 2]], axis=1
    ).astype(np.float64)

    # Farthest-point initialisation from the mean colour: no randomness involved.
    centres = [points.mean(axis=0)]
    distance = np.sum((points - centres[0]) ** 2, axis=1)
    for _ in range(1, clusters):
        centres.append(points[int(np.argmax(distance))])
        distance = np.minimum(distance, np.sum((points - centres[-1]) ** 2, axis=1))
    centre_array = np.stack(centres)

    assignment = np.zeros(points.shape[0], dtype=np.int64)
    for _ in range(iterations):
        distances = ((points[:, None, :] - centre_array[None, :, :]) ** 2).sum(axis=2)
        assignment = distances.argmin(axis=1)
        for index in range(clusters):
            members = points[assignment == index]
            if len(members):
                centre_array[index] = members.mean(axis=0)

    shares = np.bincount(assignment, minlength=clusters) / float(points.shape[0])
    hue = np.rad2deg(np.arctan2(centre_array[:, 1], centre_array[:, 0])) % 360.0
    saturation = np.clip(np.hypot(centre_array[:, 0], centre_array[:, 1]), 0.0, 1.0)
    brightness = np.clip(centre_array[:, 2], 0.0, 1.0)
    return np.stack([hue, saturation, brightness], axis=1), shares


def _correlation_length(luminance: np.ndarray) -> float:
    """Typical feature size as a fraction of the image width (0 = fine grain)."""

    centred = luminance - luminance.mean()
    power = np.abs(np.fft.rfft2(centred)) ** 2
    autocorrelation = np.fft.irfft2(power, s=centred.shape)
    peak = autocorrelation[0, 0]
    if peak <= 0:
        return 0.25
    autocorrelation = autocorrelation / peak

    def first_below(values: np.ndarray, size: int) -> float:
        below = np.nonzero(values[: size // 2] < 0.5)[0]
        return (float(below[0]) if below.size else size / 2.0) / size

    horizontal = first_below(autocorrelation[0, :], centred.shape[1])
    vertical = first_below(autocorrelation[:, 0], centred.shape[0])
    return float((horizontal + vertical) / 2.0)


def image_statistics(
    image: Image.Image | np.ndarray | Path | str,
    *,
    analysis_size: int = 192,
) -> Dict[str, object]:
    """Measurements of colour and texture that :func:`analyse_image` builds on.

    All scalar statistics lie in ``[0, 1]``.
    """

    pil_image = _ensure_image(image)
    width, height = pil_image.size
    scale = analysis_size / max(width, height)
    size = (max(8, int(round(width * scale))), max(8, int(round(height * scale))))
    rgb = np.asarray(pil_image.resize(size, Image.Resampling.BILINEAR), dtype=np.float64) / 255.0

    hsb = _rgb_to_hsb(rgb).reshape(-1, 3)
    centres, shares = _cluster_colours(hsb, PALETTE_SIZE + 1)
    order = np.argsort(-shares, kind="stable")
    centres, shares = centres[order], shares[order]

    saturation = hsb[:, 1]
    brightness = hsb[:, 2]
    weight = saturation.sum()
    if weight > 1e-6:
        resultant = np.abs(np.sum(saturation * np.exp(1j * np.deg2rad(hsb[:, 0])))) / weight
    else:
        resultant = 1.0

    luminance = 0.2989 * rgb[..., 0] + 0.5870 * rgb[..., 1] + 0.1140 * rgb[..., 2]
    grad_y, grad_x = np.gradient(luminance)
    magnitude = np.hypot(grad_x, grad_y)
    total_magnitude = magnitude.sum()

    if total_magnitude > 1e-9:
        theta = np.arctan2(grad_y, grad_x)
        alignment = float(np.sum(magnitude * np.cos(4.0 * theta)) / total_magnitude)
        bins = np.floor(((theta % math.pi) / math.pi) * 12).astype(int).clip(0, 11)
        histogram = np.bincount(bins.ravel(), weights=magnitude.ravel(), minlength=12)
        histogram = histogram / histogram.sum()
        entropy = float(-(histogram[histogram > 0] * np.log(histogram[histogram > 0])).sum() / math.log(12))
        horizontal = float((grad_y**2).sum() / ((grad_x**2).sum() + (grad_y**2).sum()))
    else:
        alignment, entropy, horizontal = 0.0, 0.0, 0.5

    rows = np.array_split(np.arange(luminance.shape[0]), 6)
    columns = np.array_split(np.arange(luminance.shape[1]), 6)
    block_contrast = np.array(
        [luminance[np.ix_(r, c)].std() for r in rows for c in columns], dtype=np.float64
    )
    mean_block = block_contrast.mean()
    irregularity = float(block_contrast.std() / mean_block) if mean_block > 1e-6 else 0.0

    contrast = float(np.clip(luminance.std() / 0.3, 0.0, 1.0))
    edges = float(np.clip(magnitude.mean() / 0.06, 0.0, 1.0))
    sharpness = float(np.clip(np.percentile(magnitude, 98) / 0.4, 0.0, 1.0))
    length = _correlation_length(luminance)
    coarseness = float(
        np.clip((math.log(max(length, 1e-3)) - math.log(0.01)) / (math.log(0.25) - math.log(0.01)), 0.0, 1.0)
    )
    mean_saturation = float(saturation.mean())
    energy = float(np.clip(0.4 * edges + 0.3 * mean_saturation + 0.3 * contrast, 0.0, 1.0))

    return {
        "palette": centres,  # (PALETTE_SIZE + 1, 3): hue°, saturation, brightness; largest area first
        "shares": shares,
        "contrast": contrast,
        "edges": edges,
        "sharpness": sharpness,
        "coarseness": coarseness,
        "rotation": float(np.clip(0.5 - 0.5 * alignment, 0.0, 1.0)),
        "roundness": float(np.clip(entropy, 0.0, 1.0)),
        "horizontal": float(np.clip((horizontal - 0.3) / 0.4, 0.0, 1.0)),
        "irregularity": float(np.clip(irregularity, 0.0, 1.0)),
        "hue_spread": float(np.clip(1.0 - resultant, 0.0, 1.0)),
        "saturation": mean_saturation,
        "saturation_spread": float(np.clip(saturation.std() / 0.35, 0.0, 1.0)),
        "brightness_spread": float(np.clip(brightness.std() / 0.35, 0.0, 1.0)),
        "energy": energy,
    }


def _scaled(name: str, unit: float) -> float:
    """Value ``unit`` of the way through a parameter's range (log-aware)."""

    spec = SPEC_BY_NAME[name]
    unit = min(max(float(unit), 0.0), 1.0)
    return spec.clamp(ParameterCodec._from_unit(spec, unit))


def analyse_image(
    image: Image.Image | np.ndarray | Path | str,
    *,
    analysis_size: int = 192,
) -> OrderedDictType[str, float]:
    """Derive the 50 Processing parameters directly from an image.

    Deterministic and training-free.  The largest colour region becomes the
    background, the next four the palette; the remaining parameters follow the
    measured texture statistics (see :func:`image_statistics`).
    """

    stats = image_statistics(image, analysis_size=analysis_size)
    palette = np.asarray(stats["palette"])
    s = {key: float(value) for key, value in stats.items() if key not in ("palette", "shares")}

    values: Dict[str, float] = {}
    background, colours = palette[0], palette[1:]
    for index in range(PALETTE_SIZE):
        hue, saturation, brightness = colours[index % len(colours)]
        values[f"palette_{index + 1}_h"] = float(hue)
        values[f"palette_{index + 1}_s"] = float(saturation * 100.0)
        values[f"palette_{index + 1}_b"] = float(brightness * 100.0)
    values["backgroundHue"] = float(background[0])
    values["backgroundSaturation"] = float(background[1] * 100.0)
    values["backgroundBrightness"] = float(background[2] * 100.0)

    units = {
        # Grain: coarse images get large cells and a slowly varying noise field.
        "cellSize": s["coarseness"],
        "noiseScale": 1.0 - s["coarseness"],
        "overlayNoiseScale": 1.0 - s["coarseness"],
        "overlayStep": s["coarseness"],
        "accentScale": s["coarseness"],
        "noiseOctaves": s["edges"],
        "noiseFalloff": s["edges"],
        # Contrast drives depth and opacity.
        "textureDepth": s["contrast"],
        "fillAlpha": 0.3 + 0.6 * s["contrast"],
        "accentAlpha": 0.2 + 0.7 * s["contrast"],
        "overlayAmplitude": s["contrast"],
        # Edges become accent rings and overlay strokes.
        "accentThreshold": 1.0 - s["edges"],
        "accentLayers": s["edges"],
        "accentStrokeWeight": s["sharpness"],
        "overlayStrokeScale": s["sharpness"],
        "overlayAlpha": 0.2 + 0.5 * s["edges"],
        "overlayBands": s["horizontal"],
        # Shape: isotropic edge directions mean round forms; skewed ones mean rotation.
        "cornerRadius": 0.8 * s["roundness"],
        "cellRotationJitter": s["rotation"],
        # Unevenness across the image becomes jitter.
        "cellScaleJitter": s["irregularity"],
        "cellOffsetJitter": 0.6 * s["irregularity"],
        "cornerRadiusJitter": s["irregularity"],
        "accentJitter": s["irregularity"],
        "accentPhaseJitter": s["irregularity"],
        "accentRadiusJitter": s["irregularity"],
        "overlayAlphaJitter": s["irregularity"],
        # Colour variety.
        "paletteSpread": 0.15 + 0.5 * s["hue_spread"],
        "palettePhaseJitter": s["hue_spread"],
        "saturationJitter": s["saturation_spread"],
        "brightnessJitter": s["brightness_spread"],
        # Busy, saturated images move faster.
        "noiseSpeed": s["energy"],
        "overlaySpeed": s["energy"],
        "accentDrift": s["energy"],
        "colorCycleSpeed": 0.5 * s["energy"] + 0.5 * s["hue_spread"],
        "backgroundAlpha": 1.0 - 0.6 * s["energy"],
    }
    for name, unit in units.items():
        values[name] = _scaled(name, unit)

    canonical, _ = canonicalise_parameters(values)
    return OrderedDict((name, canonical[name]) for name in PARAMETER_NAMES)


# ---------------------------------------------------------------------------
# CNN predictor
# ---------------------------------------------------------------------------


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
        max_channels = base_channels * 4
        for _ in range(3):
            layers.append(nn.Conv2d(in_channels, channels, kernel_size=3, stride=2, padding=1))
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
        luminance_mean = luminance.mean(dim=(1, 2)).unsqueeze(1)
        luminance_std = luminance.std(dim=(1, 2)).unsqueeze(1)
        return torch.cat([means, stds, minima, maxima, luminance_mean, luminance_std], dim=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError("images must be a 4D tensor shaped as (N, 3, H, W)")

        features = self.encoder(images)
        pooled = F.adaptive_avg_pool2d(features, 1).flatten(1)
        stats = self._global_stats(images)
        return self.head(torch.cat([pooled, stats], dim=1))

    def config(self) -> dict[str, float | int]:
        """Return the configuration needed to recreate the module."""

        return {
            "base_channels": self.base_channels,
            "mlp_hidden": self.mlp_hidden,
            "dropout": self.dropout_prob,
        }


@lru_cache(maxsize=8)
def _load_predictor(weights_path: str, modified: int) -> Tuple[ProcessingParameterPredictor, ParameterCodec]:
    del modified  # part of the cache key only: a retrained file is reloaded
    try:
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError(f"Could not read artwork model '{weights_path}': {error}") from error
    if not isinstance(checkpoint, dict) or "codec" not in checkpoint:
        raise ValueError(
            f"'{weights_path}' is not an artwork model written by train_processing_features.py "
            "of this version. Re-train it."
        )
    codec = ParameterCodec.from_dict(checkpoint["codec"])
    model_config = checkpoint.get("model_config", {})
    model = ProcessingParameterPredictor(
        parameter_count=codec.dim,
        base_channels=int(model_config.get("base_channels", 32)),
        mlp_hidden=int(model_config.get("mlp_hidden", 128)),
        dropout=float(model_config.get("dropout", 0.1)),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, codec


def compute_processing_parameters(
    image: Image.Image | np.ndarray | Path | str,
    *,
    weights_path: Path | str,
    resize_long_edge: int | None = 512,
    device: str | torch.device | None = None,
) -> OrderedDictType[str, float]:
    """Predict Processing parameters for ``image`` using a trained CNN.

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
            f"Processing feature weights not found at '{resolved_path}'. "
            "Run train_processing_features.py first."
        )

    model, codec = _load_predictor(str(resolved_path), resolved_path.stat().st_mtime_ns)
    inference_device = torch.device(device) if device is not None else torch.device("cpu")
    if inference_device.type == "cuda" and not torch.cuda.is_available():
        inference_device = torch.device("cpu")
    model = model.to(inference_device)

    tensor = prepare_image_tensor(image, resize_long_edge=resize_long_edge).unsqueeze(0)
    with torch.no_grad():
        prediction = model(tensor.to(inference_device)).squeeze(0).cpu().numpy()

    decoded = codec.decode(prediction)
    canonical, _ = canonicalise_parameters(decoded)
    return OrderedDict((name, canonical[name]) for name in PARAMETER_NAMES)


def resolve_artwork_parameters(
    image_path: Path | str,
    *,
    curated: Optional[Mapping[str, object]] = None,
    weights_path: Path | str | None = None,
    resize_long_edge: int | None = 512,
    device: str | torch.device | None = None,
) -> Tuple[OrderedDictType[str, float], str]:
    """Parameters for one artwork and where they came from.

    Curated values win over the CNN, which wins over direct image analysis.
    Returns ``(parameters, source)`` with ``source`` one of ``"metadata"``,
    ``"cnn"`` or ``"analysis"``.
    """

    if curated is not None:
        canonical, _ = canonicalise_parameters(curated)
        return OrderedDict((name, canonical[name]) for name in PARAMETER_NAMES), "metadata"
    if weights_path is not None:
        parameters = compute_processing_parameters(
            image_path, weights_path=weights_path, resize_long_edge=resize_long_edge, device=device
        )
        return parameters, "cnn"
    return analyse_image(image_path), "analysis"


__all__ = [
    "ProcessingParameterPredictor",
    "SUPPORTED_IMAGE_SUFFIXES",
    "analyse_image",
    "compute_processing_parameters",
    "image_statistics",
    "prepare_image_tensor",
    "resolve_artwork_parameters",
]
