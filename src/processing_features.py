"""Feature extraction utilities for deriving Processing parameters from images."""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Iterable, OrderedDict as OrderedDictType

import numpy as np
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


def _sliding_window_view(array: np.ndarray, window_shape: Iterable[int]) -> np.ndarray:
    """Compatibility wrapper for ``numpy.lib.stride_tricks.sliding_window_view``."""

    from numpy.lib.stride_tricks import sliding_window_view

    return sliding_window_view(array, window_shape)


def compute_processing_parameters(
    image: Image.Image | np.ndarray | Path | str,
    *,
    resize_long_edge: int | None = 512,
) -> OrderedDictType[str, float]:
    """Analyse an artwork image and derive Processing-friendly parameters.

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
    luminance = gray.reshape(-1)
    percentile_count = max(int(len(luminance) * 0.1), 1)
    dark_indices = np.argpartition(luminance, percentile_count)[:percentile_count]
    light_indices = np.argpartition(luminance, len(luminance) - percentile_count)[-percentile_count:]
    background_color = flattened[dark_indices].mean(axis=0)
    accent_color = flattened[light_indices].mean(axis=0)

    # HSV statistics capture hue and saturation trends.
    hsv = _rgb_to_hsv(rgb)
    hue_mean = float(np.mean(hsv[..., 0]))
    hue_variance = float(np.var(hsv[..., 0]))
    saturation_mean = float(np.mean(hsv[..., 1]))
    value_mean = float(np.mean(hsv[..., 2]))

    # Texture descriptors derived from local gradients.
    gx = np.pad(np.diff(gray, axis=1), ((0, 0), (0, 1)), mode="edge")
    gy = np.pad(np.diff(gray, axis=0), ((0, 1), (0, 0)), mode="edge")
    gradient_magnitude = np.sqrt(gx ** 2 + gy ** 2)
    gradient_strength = float(np.mean(gradient_magnitude))
    edge_threshold = float(np.mean(gradient_magnitude) + np.std(gradient_magnitude))
    edge_density = float(np.mean(gradient_magnitude > edge_threshold))

    local_windows = _sliding_window_view(gray, (5, 5))
    local_variance = np.var(local_windows, axis=(-2, -1))
    texture_complexity = float(np.mean(local_variance))

    # Map statistics into Processing-oriented scales.
    stroke_weight = float(np.clip(1.0 + edge_density * 12.0, 1.0, 12.0))
    noise_scale = float(np.clip(gray.std() * 12.0, 0.5, 12.0))
    pattern_scale = float(np.clip(1.0 + texture_complexity * 80.0, 1.0, 25.0))
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
