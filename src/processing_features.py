"""Feature extraction utilities for deriving Processing parameters from images."""
from __future__ import annotations

from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from typing import OrderedDict as OrderedDictType

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


def _ensure_image(image: Image.Image | np.ndarray | Path | str) -> Image.Image:
    """Load an image and ensure it is in RGB mode."""

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


def _rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """Convert an RGB array in range [0, 1] to HSV."""

    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.max(rgb, axis=-1)
    minc = np.min(rgb, axis=-1)
    delta = maxc - minc

    hue = np.zeros_like(maxc)
    mask = delta > 1e-6

    # Hue calculation branches based on which channel is the maximum.
    r_is_max = np.logical_and(mask, maxc == r)
    g_is_max = np.logical_and(mask, maxc == g)
    b_is_max = np.logical_and(mask, maxc == b)

    hue[r_is_max] = (g[r_is_max] - b[r_is_max]) / delta[r_is_max]
    hue[g_is_max] = 2.0 + (b[g_is_max] - r[g_is_max]) / delta[g_is_max]
    hue[b_is_max] = 4.0 + (r[b_is_max] - g[b_is_max]) / delta[b_is_max]
    hue = (hue / 6.0) % 1.0

    saturation = np.zeros_like(maxc)
    saturation[mask] = delta[mask] / maxc[mask]

    value = maxc
    return np.stack([hue, saturation, value], axis=-1)


def _gabor_kernel(
    theta: float, *, sigma: float = 1.5, lambd: float = 3.0, gamma: float = 0.5, size: int = 7
) -> torch.Tensor:
    """Create a Gabor kernel for the provided orientation."""

    # Create coordinate grid centred at zero.
    coords = torch.linspace(-(size // 2), size // 2, size)
    y, x = torch.meshgrid(coords, coords, indexing="ij")

    # Rotate coordinates.
    rotx = x * torch.cos(theta) + y * torch.sin(theta)
    roty = -x * torch.sin(theta) + y * torch.cos(theta)

    gaussian = torch.exp(-0.5 * ((rotx ** 2 + (gamma ** 2) * roty ** 2) / (sigma ** 2)))
    sinusoid = torch.cos((2.0 * np.pi * rotx) / lambd)
    kernel = gaussian * sinusoid
    kernel -= kernel.mean()
    kernel /= kernel.abs().sum().clamp_min(1e-6)
    return kernel


class ProcessingFeatureCNN(nn.Module):
    """Small CNN used to derive texture descriptors from artwork images."""

    def __init__(self) -> None:
        super().__init__()

        sobel_x = torch.tensor(
            [[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]], dtype=torch.float32
        )
        sobel_y = sobel_x.t().contiguous()
        laplacian = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float32
        )
        smooth = torch.full((3, 3), 1.0 / 9.0, dtype=torch.float32)

        edge_kernels = torch.stack(
            [
                sobel_x / sobel_x.abs().sum(),
                sobel_y / sobel_y.abs().sum(),
                laplacian / laplacian.abs().sum().clamp_min(1.0),
                smooth,
            ]
        ).unsqueeze(1)
        self.register_buffer("edge_kernels", edge_kernels)

        orientations = [0.0, np.pi / 4.0, np.pi / 2.0, 3.0 * np.pi / 4.0]
        texture_kernels = []
        for angle in orientations:
            kernel = _gabor_kernel(angle)
            # Blend gradient channels with orientation-sensitive Gabor filters and
            # inject a small contribution from the Laplacian response so the
            # activations remain sensitive to fine texture variations.
            texture_kernels.append(torch.stack([kernel * 0.5, kernel * 0.5, kernel * 0.25, kernel * 0.1]))
        texture_kernel_tensor = torch.stack(texture_kernels)
        self.register_buffer("texture_kernels", texture_kernel_tensor)

        orientation_kernels = torch.zeros((8, 4, 1, 1), dtype=torch.float32)
        for index in range(4):
            orientation_kernels[index, index, 0, 0] = 1.0
            orientation_kernels[index + 4, index, 0, 0] = -1.0
        self.register_buffer("orientation_kernels", orientation_kernels)

    def forward(self, luminance: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run the CNN on a luminance map and return intermediate features."""

        conv1 = F.conv2d(luminance, self.edge_kernels, padding=1)
        edge_activation = F.relu(conv1)
        conv2 = F.conv2d(edge_activation, self.texture_kernels, padding=3)
        texture_activation = F.relu(conv2)
        conv3 = F.conv2d(texture_activation, self.orientation_kernels)
        orientation_activation = F.relu(conv3)
        return {
            "conv1": conv1,
            "edge": edge_activation,
            "texture": texture_activation,
            "orientation": orientation_activation,
        }


@lru_cache(maxsize=1)
def _get_feature_extractor() -> ProcessingFeatureCNN:
    """Singleton accessor so weights are created only once per process."""

    return ProcessingFeatureCNN()


def compute_processing_parameters(
    image: Image.Image | np.ndarray | Path | str,
    *,
    resize_long_edge: int | None = 512,
) -> OrderedDictType[str, float]:
    """Analyse an artwork image and derive Processing-friendly parameters.

    The implementation first extracts global colour statistics directly in the RGB
    domain to recover palette information. Texture-oriented statistics are produced by
    forwarding the image through :class:`ProcessingFeatureCNN`, a lightweight
    convolutional network whose hand-crafted kernels emulate classical edge, Gabor, and
    orientation detectors. The resulting activations are aggregated into parameters that
    directly map to Processing sketch controls.

    Parameters
    ----------
    image:
        Image to analyse. Paths will be loaded via Pillow. Arrays are assumed to be in
        RGB order with values in the range ``[0, 255]``.
    resize_long_edge:
        Optionally resize the image so that its longest edge matches ``resize_long_edge``.
        ``None`` disables resizing.

    Returns
    -------
    OrderedDict[str, float]
        Ordered mapping of parameter names to values normalised to roughly
        Processing-friendly ranges.
    """

    pil_image = _ensure_image(image)
    if resize_long_edge is not None:
        width, height = pil_image.size
        long_edge = max(width, height)
        if long_edge > resize_long_edge:
            scale = resize_long_edge / long_edge
            new_size = (int(round(width * scale)), int(round(height * scale)))
            pil_image = pil_image.resize(new_size, Image.Resampling.BICUBIC)

    rgb = np.asarray(pil_image, dtype=np.float32) / 255.0
    gray = 0.2989 * rgb[..., 0] + 0.5870 * rgb[..., 1] + 0.1140 * rgb[..., 2]

    # Basic colour statistics.
    std_color = rgb.std(axis=(0, 1))

    flattened = rgb.reshape(-1, 3)
    luminance_flat = gray.reshape(-1)
    percentile_count = max(int(len(luminance_flat) * 0.1), 1)
    dark_indices = np.argpartition(luminance_flat, percentile_count)[:percentile_count]
    light_indices = np.argpartition(
        luminance_flat, len(luminance_flat) - percentile_count
    )[-percentile_count:]
    background_color = flattened[dark_indices].mean(axis=0)
    accent_color = flattened[light_indices].mean(axis=0)

    hsv = _rgb_to_hsv(rgb)
    hue_mean = float(np.mean(hsv[..., 0]))
    hue_variance = float(np.var(hsv[..., 0]))
    saturation_mean = float(np.mean(hsv[..., 1]))
    value_mean = float(np.mean(hsv[..., 2]))

    # CNN-based texture descriptors.
    luminance_tensor = torch.from_numpy(gray).to(torch.float32).unsqueeze(0).unsqueeze(0)
    extractor = _get_feature_extractor()
    with torch.no_grad():
        feature_maps = extractor(luminance_tensor)

    grad_x = feature_maps["conv1"][0, 0]
    grad_y = feature_maps["conv1"][0, 1]
    laplacian_map = feature_maps["conv1"][0, 2]
    gradient_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)
    gradient_strength = float(gradient_magnitude.mean().item())
    edge_threshold = gradient_magnitude.mean() + gradient_magnitude.std()
    edge_density = float((gradient_magnitude > edge_threshold).float().mean().item())

    texture_maps = feature_maps["texture"].abs()
    texture_complexity = float(texture_maps.mean().item())

    orientation_maps = feature_maps["orientation"]
    pattern_variability = float(orientation_maps.std(dim=(-1, -2)).mean().item())

    # Map statistics into Processing-oriented scales.
    stroke_weight = float(np.clip(1.0 + edge_density * 10.0, 1.0, 12.0))
    noise_scale = float(
        np.clip(0.5 + laplacian_map.abs().mean().item() * 10.0, 0.5, 12.0)
    )
    pattern_scale = float(np.clip(1.0 + pattern_variability * 25.0, 1.0, 25.0))
    colour_variation = float(np.mean(std_color))

    parameters = OrderedDict(
        [
            ("background_r", float(background_color[0])),
            ("background_g", float(background_color[1])),
            ("background_b", float(background_color[2])),
            ("accent_r", float(accent_color[0])),
            ("accent_g", float(accent_color[1])),
            ("accent_b", float(accent_color[2])),
            ("mean_hue", hue_mean),
            ("hue_variance", hue_variance),
            ("mean_saturation", saturation_mean),
            ("mean_value", value_mean),
            ("gradient_strength", gradient_strength),
            ("texture_complexity", texture_complexity),
            ("pattern_scale", pattern_scale),
            ("stroke_weight", stroke_weight),
            ("noise_scale", noise_scale),
            ("colour_variation", colour_variation),
        ]
    )

    return parameters  # type: ignore[return-value]


__all__ = ["compute_processing_parameters"]
