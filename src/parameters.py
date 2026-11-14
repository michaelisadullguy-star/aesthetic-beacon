"""The 50 Processing parameters driven by the EEG, and how they are encoded.

This module is the single source of truth shared by training, the live loop and
the Processing sketch.  It defines

* which parameters the models control (38 scalars + 4 palette colours × h/s/b),
* the range each one may take, so nothing that reaches the sketch can stall or
  break it (a cell size of zero, for instance, would loop forever),
* the display constants that are *not* learned (canvas size, frame rate),
* :class:`ParameterCodec`, which converts between parameter dictionaries and the
  vectors the networks regress.  Hues are encoded as points on a circle so that
  averaging red (350°) and red (10°) does not come out cyan.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

PALETTE_SIZE = 4

#: Display settings fixed by the hardware; read by the sketch at start-up, never learned.
FIXED_DISPLAY_PARAMETERS: Dict[str, int] = {
    "canvasWidth": 1080,
    "canvasHeight": 640,
    "targetFrameRate": 60,
    "smoothingSamples": 4,
}


@dataclass(frozen=True)
class ParameterSpec:
    """Definition of one controllable parameter."""

    name: str
    low: float
    high: float
    default: float
    integer: bool = False
    #: Angle in degrees that wraps around at ``high``.
    circular: bool = False
    #: Interpolate on a logarithmic scale (for scale- and speed-like quantities).
    log: bool = False

    def clamp(self, value: float) -> float:
        if value is None or not math.isfinite(value):
            value = self.default
        if self.circular:
            value = value % self.high
        else:
            value = min(max(value, self.low), self.high)
        if self.integer:
            value = float(int(round(value)))
        return float(value)


def _spec(name: str, low: float, high: float, default: float, **kwargs) -> ParameterSpec:
    return ParameterSpec(name, float(low), float(high), float(default), **kwargs)


_SCALAR_SPECS: Tuple[ParameterSpec, ...] = (
    # --- cell grid -----------------------------------------------------------
    _spec("cellSize", 12, 72, 24, integer=True),
    _spec("cellScaleJitter", 0.0, 1.0, 0.38),
    _spec("cellRotationJitter", 0.0, 1.5, 0.55),
    _spec("cellOffsetJitter", 0.0, 1.0, 0.30),
    _spec("cornerRadiusJitter", 0.0, 8.0, 2.8),
    _spec("palettePhaseJitter", 0.0, 1.0, 0.28),
    # --- noise field ---------------------------------------------------------
    _spec("noiseScale", 0.001, 0.03, 0.006, log=True),
    _spec("noiseSpeed", 0.02, 1.0, 0.18, log=True),
    _spec("noiseOctaves", 1, 6, 5, integer=True),
    _spec("noiseFalloff", 0.3, 0.7, 0.52),
    # --- colour dynamics -----------------------------------------------------
    _spec("colorCycleSpeed", 0.02, 1.5, 0.45, log=True),
    _spec("paletteSpread", 0.0, 180.0, 85.0),
    _spec("saturationJitter", 0.0, 50.0, 18.0),
    _spec("brightnessJitter", 0.0, 50.0, 26.0),
    # --- texture -------------------------------------------------------------
    _spec("textureDepth", 0.1, 2.0, 0.7),
    _spec("fillAlpha", 15.0, 100.0, 60.0),
    _spec("cornerRadius", 0.0, 24.0, 5.0),
    # --- accent rings --------------------------------------------------------
    _spec("accentThreshold", 0.45, 0.95, 0.72),
    _spec("accentLayers", 0, 6, 3, integer=True),
    _spec("accentScale", 0.5, 4.0, 2.2),
    _spec("accentStrokeWeight", 0.5, 5.0, 1.6),
    _spec("accentAlpha", 0.0, 90.0, 45.0),
    _spec("accentJitter", 0.0, 1.0, 0.35),
    _spec("accentDrift", 0.0, 1.5, 0.5),
    _spec("accentPhaseJitter", 0.0, 1.0, 0.4),
    _spec("accentRadiusJitter", 0.0, 1.0, 0.4),
    # --- flowing overlay bands -----------------------------------------------
    _spec("overlayBands", 0, 12, 6, integer=True),
    _spec("overlayNoiseScale", 0.001, 0.02, 0.0035, log=True),
    _spec("overlayAmplitude", 0.0, 120.0, 42.0),
    _spec("overlayAlpha", 0.0, 70.0, 32.0),
    _spec("overlaySpeed", 0.02, 1.2, 0.4, log=True),
    _spec("overlayStep", 6, 48, 14, integer=True),
    _spec("overlayStrokeScale", 0.5, 6.0, 3.4),
    _spec("overlayAlphaJitter", 0.0, 1.0, 0.4),
    # --- background ----------------------------------------------------------
    _spec("backgroundHue", 0.0, 360.0, 210.0, circular=True),
    _spec("backgroundSaturation", 0.0, 100.0, 18.0),
    _spec("backgroundBrightness", 0.0, 100.0, 8.0),
    _spec("backgroundAlpha", 20.0, 100.0, 100.0),
)

_DEFAULT_PALETTE: Tuple[Tuple[float, float, float], ...] = (
    (210.0, 65.0, 95.0),
    (310.0, 70.0, 95.0),
    (35.0, 85.0, 96.0),
    (160.0, 55.0, 90.0),
)


def _palette_specs() -> Tuple[ParameterSpec, ...]:
    specs: List[ParameterSpec] = []
    for index, (hue, saturation, brightness) in enumerate(_DEFAULT_PALETTE, start=1):
        specs.append(_spec(f"palette_{index}_h", 0.0, 360.0, hue, circular=True))
        specs.append(_spec(f"palette_{index}_s", 0.0, 100.0, saturation))
        specs.append(_spec(f"palette_{index}_b", 0.0, 100.0, brightness))
    return tuple(specs)


#: Palette colours first, then scalars — the order used for model outputs.
PARAMETER_SPECS: Tuple[ParameterSpec, ...] = _palette_specs() + _SCALAR_SPECS
PARAMETER_NAMES: Tuple[str, ...] = tuple(spec.name for spec in PARAMETER_SPECS)
SPEC_BY_NAME: Dict[str, ParameterSpec] = {spec.name: spec for spec in PARAMETER_SPECS}

assert len(PARAMETER_SPECS) == 50, "The display is driven by exactly 50 parameters."

_PALETTE_KEY = re.compile(r"^palette[\s_\-]*(\[?)(\d+)\]?[\s_\-.]*([hsbHSB])$")


def default_parameters() -> Dict[str, float]:
    """Parameter set shown before any EEG-driven update has arrived."""

    return {spec.name: spec.clamp(spec.default) for spec in PARAMETER_SPECS}


def canonicalise_parameters(
    parameters: Mapping[str, object],
    *,
    fill_defaults: bool = True,
) -> Tuple[Dict[str, float], List[str]]:
    """Normalise a loosely formatted parameter mapping to the 50 canonical names.

    Understands nested palettes (``"palette": [{"h":..,"s":..,"b":..}, ...]``),
    flattened palette keys in several spellings and a single ``palette_h/s/b``
    base colour.  Out-of-range values are clamped.  Returns the canonical
    dictionary and the list of keys that were not recognised or are fixed
    display settings.
    """

    result: Dict[str, float] = {}
    ignored: List[str] = []
    base_colour: Dict[str, float] = {}

    for key, value in parameters.items():
        key = str(key)
        if key == "palette":
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                colours = [entry for entry in value if isinstance(entry, Mapping)]
                for index in range(PALETTE_SIZE):
                    if not colours:
                        break
                    entry = colours[index % len(colours)]
                    shift = 0.0 if index < len(colours) else 360.0 / PALETTE_SIZE * (index // len(colours))
                    for component in "hsb":
                        if component in entry:
                            number = _to_float(entry[component])
                            if number is None:
                                continue
                            if component == "h":
                                number += shift
                            result[f"palette_{index + 1}_{component}"] = number
            else:
                ignored.append(key)
            continue

        match = _PALETTE_KEY.match(key)
        if match:
            index = int(match.group(2)) + (1 if match.group(1) else 0)
            if 1 <= index <= PALETTE_SIZE:
                number = _to_float(value)
                if number is not None:
                    result[f"palette_{index}_{match.group(3).lower()}"] = number
            else:
                ignored.append(key)
            continue

        if key in ("palette_h", "palette_s", "palette_b"):
            number = _to_float(value)
            if number is not None:
                base_colour[key[-1]] = number
            continue

        if key in SPEC_BY_NAME:
            number = _to_float(value)
            if number is not None:
                result[key] = number
            continue

        ignored.append(key)

    if base_colour:
        for index in range(PALETTE_SIZE):
            for component, number in base_colour.items():
                name = f"palette_{index + 1}_{component}"
                if name not in result:
                    result[name] = number + (index * 360.0 / PALETTE_SIZE if component == "h" else 0.0)

    canonical: Dict[str, float] = {}
    for spec in PARAMETER_SPECS:
        if spec.name in result:
            canonical[spec.name] = spec.clamp(result[spec.name])
        elif fill_defaults:
            canonical[spec.name] = spec.clamp(spec.default)
    return canonical, ignored


def _to_float(value: object) -> Optional[float]:
    if isinstance(value, bool):
        return float(value)
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def constrain_parameters(parameters: Mapping[str, float]) -> Dict[str, float]:
    """Clamp every known parameter to its safe range; unknown keys are dropped."""

    return {
        spec.name: spec.clamp(float(parameters[spec.name]))
        for spec in PARAMETER_SPECS
        if spec.name in parameters
    }


def blend_parameters(
    previous: Optional[Mapping[str, float]],
    new: Mapping[str, float],
    weight: float,
) -> Dict[str, float]:
    """Move from ``previous`` towards ``new`` by ``weight`` (1.0 = jump straight to new).

    Hues take the short way around the colour wheel.
    """

    if previous is None or weight >= 1.0:
        return constrain_parameters(new)
    blended: Dict[str, float] = {}
    for spec in PARAMETER_SPECS:
        if spec.name not in new:
            continue
        target = float(new[spec.name])
        start = float(previous.get(spec.name, target))
        if spec.circular:
            delta = ((target - start + spec.high / 2.0) % spec.high) - spec.high / 2.0
            value = start + weight * delta
        else:
            value = start + weight * (target - start)
        blended[spec.name] = spec.clamp(value)
    return blended


def typed_items(parameters: Mapping[str, float]) -> List[Tuple[str, float | int]]:
    """``(name, value)`` pairs in canonical order with integers as ``int``."""

    items: List[Tuple[str, float | int]] = []
    for spec in PARAMETER_SPECS:
        if spec.name in parameters:
            value = spec.clamp(float(parameters[spec.name]))
            items.append((spec.name, int(value) if spec.integer else value))
    return items


def to_sketch_json(parameters: Mapping[str, float]) -> Dict[str, object]:
    """Parameter set in the JSON layout the Processing sketch loads from disk."""

    document: Dict[str, object] = dict(FIXED_DISPLAY_PARAMETERS)
    palette: List[Dict[str, float]] = [dict() for _ in range(PALETTE_SIZE)]
    for name, value in typed_items(parameters):
        match = _PALETTE_KEY.match(name)
        if match:
            palette[int(match.group(2)) - 1][match.group(3)] = round(float(value), 4)
        else:
            document[name] = value if isinstance(value, int) else round(float(value), 6)
    if all(len(colour) == 3 for colour in palette):
        document["palette"] = palette
    return document


class ParameterCodec:
    """Converts parameter dictionaries to regression targets and back.

    Ordinary parameters map linearly (or log-linearly) onto ``[-1, 1]``; hues
    become ``(cos, sin)`` pairs.  Decoding always clamps to the safe range, so
    whatever a network outputs, the result is a valid parameter set.
    """

    def __init__(self, specs: Sequence[ParameterSpec] = PARAMETER_SPECS) -> None:
        self.specs: Tuple[ParameterSpec, ...] = tuple(specs)
        self.names: Tuple[str, ...] = tuple(spec.name for spec in self.specs)
        self._slices: List[Tuple[int, int]] = []
        offset = 0
        for spec in self.specs:
            width = 2 if spec.circular else 1
            self._slices.append((offset, offset + width))
            offset += width
        self.dim = offset

    # -- encoding ---------------------------------------------------------------
    def encode(self, parameters: Mapping[str, float]) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        for spec, (start, _end) in zip(self.specs, self._slices):
            value = spec.clamp(float(parameters.get(spec.name, spec.default)))
            if spec.circular:
                angle = 2.0 * math.pi * value / spec.high
                vector[start] = math.cos(angle)
                vector[start + 1] = math.sin(angle)
            else:
                vector[start] = 2.0 * self._unit(spec, value) - 1.0
        return vector

    def decode(self, vector: Sequence[float] | np.ndarray) -> Dict[str, float]:
        values = np.asarray(vector, dtype=np.float64).ravel()
        if values.shape[0] != self.dim:
            raise ValueError(f"Expected a vector of length {self.dim}, got {values.shape[0]}.")
        decoded: Dict[str, float] = {}
        for spec, (start, _end) in zip(self.specs, self._slices):
            if spec.circular:
                x, y = values[start], values[start + 1]
                if not (math.isfinite(x) and math.isfinite(y)) or math.hypot(x, y) < 1e-6:
                    decoded[spec.name] = spec.clamp(spec.default)
                else:
                    decoded[spec.name] = spec.clamp(math.degrees(math.atan2(y, x)) % 360.0 * spec.high / 360.0)
            else:
                z = values[start]
                if not math.isfinite(z):
                    decoded[spec.name] = spec.clamp(spec.default)
                else:
                    unit = min(max((z + 1.0) / 2.0, 0.0), 1.0)
                    decoded[spec.name] = spec.clamp(self._from_unit(spec, unit))
        return decoded

    @staticmethod
    def _unit(spec: ParameterSpec, value: float) -> float:
        if spec.high == spec.low:
            return 0.5
        if spec.log:
            return (math.log(value) - math.log(spec.low)) / (math.log(spec.high) - math.log(spec.low))
        return (value - spec.low) / (spec.high - spec.low)

    @staticmethod
    def _from_unit(spec: ParameterSpec, unit: float) -> float:
        if spec.log:
            return math.exp(math.log(spec.low) + unit * (math.log(spec.high) - math.log(spec.low)))
        return spec.low + unit * (spec.high - spec.low)

    # -- persistence ------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "specs": [
                [s.name, s.low, s.high, s.default, bool(s.integer), bool(s.circular), bool(s.log)]
                for s in self.specs
            ]
        }

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "ParameterCodec":
        if not data or "specs" not in data:
            return cls()
        specs = [
            ParameterSpec(str(n), float(lo), float(hi), float(d), bool(i), bool(c), bool(g))
            for n, lo, hi, d, i, c, g in data["specs"]
        ]
        return cls(specs)


def schema_document() -> List[Dict[str, object]]:
    """The parameter definitions as plain data, read by the Processing sketch."""

    return [
        {
            "name": spec.name,
            "min": spec.low,
            "max": spec.high,
            "default": spec.default,
            "integer": spec.integer,
            "circular": spec.circular,
        }
        for spec in PARAMETER_SPECS
    ]


def write_sketch_data(directory) -> None:
    """Write the schema and the default parameter file into a sketch's ``data`` folder.

    Run ``python -m src.parameters`` after changing the parameter definitions,
    so the sketch and the Python side stay in step.
    """

    import json
    from pathlib import Path

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / "parameter_schema.json", "w", encoding="utf8") as handle:
        json.dump(schema_document(), handle, indent=2)
        handle.write("\n")
    with open(directory / "dynamic_texture_params.json", "w", encoding="utf8") as handle:
        json.dump(to_sketch_json(default_parameters()), handle, indent=2)
        handle.write("\n")


__all__ = [
    "FIXED_DISPLAY_PARAMETERS",
    "PALETTE_SIZE",
    "PARAMETER_NAMES",
    "PARAMETER_SPECS",
    "ParameterCodec",
    "ParameterSpec",
    "SPEC_BY_NAME",
    "blend_parameters",
    "canonicalise_parameters",
    "constrain_parameters",
    "default_parameters",
    "schema_document",
    "to_sketch_json",
    "typed_items",
    "write_sketch_data",
]


if __name__ == "__main__":
    from pathlib import Path

    _templates = Path(__file__).resolve().parent.parent / "scripts" / "processing_templates"
    write_sketch_data(_templates / "dynamic_texture_osc" / "data")
    print(f"wrote {_templates / 'dynamic_texture_osc' / 'data'}")
