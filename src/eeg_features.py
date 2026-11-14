"""From raw headset samples to the 8×3200 band matrix the models consume.

The headset delivers one raw channel.  Every model in the pipeline works on the
same derived representation: the amplitude envelope of eight frequency bands,
sampled at 640 Hz, in five-second windows — an ``(8, 3200)`` matrix whose rows
are δ, θ, α1, α2, β1, β2, γ1, γ2.

Two rules keep training and live operation consistent:

* A window is always computed from the samples *up to its end* plus a short
  stretch of preceding context (:func:`extract_band_window`).  Training windows
  cut out of a recording therefore contain exactly what the live loop would
  have seen at that moment.
* Band amplitudes are expressed relative to a robust per-session baseline in
  the log domain (:func:`normalise_bands`), which removes differences in
  electrode contact and between people while keeping how the bands move.
"""
from __future__ import annotations

import logging
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy import fft as scipy_fft
from scipy import signal as scipy_signal

from .thinkgear import (
    POOR_SIGNAL_UNKNOWN,
    decode_bytes,
    hex_text_to_bytes,
    looks_like_thinkgear,
)

logger = logging.getLogger(__name__)

SAMPLE_RATE = 640
SEGMENT_SECONDS = 5.0

#: Name, lower edge and upper edge (Hz) of the eight bands, in row order.
BANDS: Tuple[Tuple[str, float, float], ...] = (
    ("delta", 0.5, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha1", 8.0, 10.0),
    ("alpha2", 10.0, 13.0),
    ("beta1", 13.0, 20.0),
    ("beta2", 20.0, 30.0),
    ("gamma1", 30.0, 45.0),
    ("gamma2", 45.0, 60.0),
)
BAND_NAMES: Tuple[str, ...] = tuple(name for name, _, _ in BANDS)

#: Sampling rates a headset plausibly runs at; measured rates snap to these.
COMMON_SAMPLE_RATES: Tuple[int, ...] = (128, 250, 256, 500, 512, 640, 1000, 1024)

_LOG_FLOOR = 1e-2
_MIN_BASELINE_STD = 0.05


class RecordingError(ValueError):
    """Raised when a recording cannot be decoded or is unusable."""


@dataclass(frozen=True)
class FeatureConfig:
    """Everything that defines how raw samples become model input.

    A copy is stored in each checkpoint so that a model is always fed features
    computed the way it was trained.
    """

    sample_rate: int = SAMPLE_RATE
    segment_seconds: float = SEGMENT_SECONDS
    bands: Tuple[Tuple[str, float, float], ...] = BANDS
    #: Rate the headset really samples at. ``None`` measures it from the stream's
    #: once-per-second status packets and falls back to ``sample_rate``.
    device_sample_rate: Optional[int] = None
    #: Mains frequency removed before band filtering (50 Hz sits inside γ2). ``None`` disables.
    notch_hz: Optional[float] = 50.0
    #: Signal before a window that is used to settle the filters.
    context_seconds: float = 2.0
    filter_order: int = 3
    # --- window quality gate -------------------------------------------------
    #: Headset contact-quality values above this count as poor (0 = good, 200 = off the head).
    poor_signal_threshold: int = 50
    max_poor_fraction: float = 0.2
    #: Raw amplitude treated as amplifier saturation.
    clip_level: float = 2040.0
    max_clip_fraction: float = 0.05
    #: Standard deviation below which the signal is considered a flat line.
    min_std: float = 0.5
    #: Samples further than this many robust standard deviations from the median are artefacts.
    artifact_sigma: float = 8.0
    max_artifact_fraction: float = 0.25

    @property
    def segment_samples(self) -> int:
        return int(round(self.sample_rate * self.segment_seconds))

    @property
    def context_samples(self) -> int:
        return int(round(self.sample_rate * self.context_seconds))

    @property
    def num_bands(self) -> int:
        return len(self.bands)

    def to_dict(self) -> dict:
        data = dict(self.__dict__)
        data["bands"] = [[name, float(lo), float(hi)] for name, lo, hi in self.bands]
        return data

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "FeatureConfig":
        if not data:
            return cls()
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        if "bands" in known:
            known["bands"] = tuple((str(n), float(lo), float(hi)) for n, lo, hi in known["bands"])
        return cls(**known)


DEFAULT_FEATURE_CONFIG = FeatureConfig()


# ---------------------------------------------------------------------------
# Sample rate handling
# ---------------------------------------------------------------------------


def snap_sample_rate(measured: float, *, tolerance: float = 0.04) -> int:
    """Round a measured rate to the nearest plausible device rate."""

    nearest = min(COMMON_SAMPLE_RATES, key=lambda rate: abs(rate - measured))
    if abs(nearest - measured) <= tolerance * nearest:
        return int(nearest)
    return int(round(measured))


def estimate_sample_rate(power_positions: Sequence[int]) -> Optional[int]:
    """Measure the device's sampling rate from its once-per-second status packets.

    The headset sends one status packet per second between raw samples, so the
    number of samples between two of them *is* the sampling rate.  Returns
    ``None`` when the stream does not contain enough status packets.
    """

    if len(power_positions) < 4:
        return None
    intervals = np.diff(np.asarray(power_positions, dtype=np.int64))
    intervals = intervals[intervals > 0]
    if intervals.size < 3:
        return None
    return snap_sample_rate(float(np.median(intervals)))


def resample_signal(samples: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """Polyphase resampling between two sampling rates."""

    samples = np.asarray(samples, dtype=np.float64)
    if from_rate == to_rate or samples.size == 0:
        return samples.astype(np.float32)
    ratio = Fraction(int(to_rate), int(from_rate)).limit_denominator(2000)
    resampled = scipy_signal.resample_poly(
        samples, ratio.numerator, ratio.denominator, padtype="line"
    )
    return resampled.astype(np.float32)


# ---------------------------------------------------------------------------
# Band envelopes
# ---------------------------------------------------------------------------


@lru_cache(maxsize=16)
def _filter_bank(
    sample_rate: int,
    bands: Tuple[Tuple[str, float, float], ...],
    order: int,
    notch_hz: Optional[float],
) -> Tuple[Optional[np.ndarray], Tuple[np.ndarray, ...]]:
    nyquist = sample_rate / 2.0
    band_filters = []
    for name, low, high in bands:
        if not 0 < low < high:
            raise ValueError(f"Band '{name}' has invalid edges {low}–{high} Hz.")
        if high >= nyquist:
            raise ValueError(
                f"Band '{name}' reaches {high} Hz but the sampling rate of {sample_rate} Hz "
                f"only resolves frequencies below {nyquist} Hz."
            )
        band_filters.append(
            scipy_signal.butter(order, [low, high], btype="bandpass", fs=sample_rate, output="sos")
        )
    notch = None
    if notch_hz and notch_hz < nyquist:
        b, a = scipy_signal.iirnotch(notch_hz, Q=30.0, fs=sample_rate)
        notch = scipy_signal.tf2sos(b, a)
    return notch, tuple(band_filters)


def band_envelopes(raw: np.ndarray, config: FeatureConfig = DEFAULT_FEATURE_CONFIG) -> np.ndarray:
    """Amplitude envelope of every band for a stretch of raw signal.

    Returns a ``float32`` array shaped ``(bands, len(raw))`` in raw amplitude
    units.  Each band is isolated with a zero-phase Butterworth filter and its
    envelope taken as the magnitude of the analytic signal.
    """

    x = np.asarray(raw, dtype=np.float64).ravel()
    n = x.size
    if n < config.sample_rate // 2:
        raise ValueError(
            f"Need at least half a second of signal to compute band envelopes, got {n} samples."
        )
    if not np.all(np.isfinite(x)):
        raise ValueError("Signal contains NaN or infinite values.")

    notch, filters = _filter_bank(config.sample_rate, config.bands, config.filter_order, config.notch_hz)

    # Extend the signal at both ends so filter and Hilbert transients fall outside it.
    pad = min(n - 1, 2 * config.sample_rate)
    target = scipy_fft.next_fast_len(n + 2 * pad, real=False)
    right = min(n - 1, target - n - pad)
    x = x - np.median(x)
    x = np.pad(x, (pad, right), mode="reflect", reflect_type="odd")

    if notch is not None:
        x = scipy_signal.sosfiltfilt(notch, x, padlen=0)

    out = np.empty((len(filters), n), dtype=np.float32)
    for index, sos in enumerate(filters):
        band = scipy_signal.sosfiltfilt(sos, x, padlen=0)
        out[index] = np.abs(scipy_signal.hilbert(band))[pad : pad + n]
    return out


def extract_band_window(
    raw: np.ndarray,
    end: int,
    config: FeatureConfig = DEFAULT_FEATURE_CONFIG,
    *,
    length: Optional[int] = None,
) -> np.ndarray:
    """Band matrix for the ``length`` samples that end at index ``end``.

    Only samples before ``end`` are used (the window plus the configured
    context), which is exactly the information the live loop has at that
    moment.  Use this for every window that reaches a model.
    """

    length = int(length or config.segment_samples)
    start = end - length
    if start < 0 or end > len(raw):
        raise ValueError(
            f"Window [{start}, {end}) lies outside the signal of {len(raw)} samples."
        )
    chunk_start = max(0, start - config.context_samples)
    envelopes = band_envelopes(raw[chunk_start:end], config)
    return envelopes[:, -length:]


# ---------------------------------------------------------------------------
# Window quality
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WindowQuality:
    """Outcome of the artefact / contact check for one window."""

    ok: bool
    reason: str
    poor_fraction: float = 0.0
    clip_fraction: float = 0.0
    artifact_fraction: float = 0.0
    std: float = 0.0

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "reason": self.reason,
            "poor_fraction": round(self.poor_fraction, 4),
            "clip_fraction": round(self.clip_fraction, 4),
            "artifact_fraction": round(self.artifact_fraction, 4),
        }


def robust_sigma(values: np.ndarray) -> float:
    """Standard deviation estimated from the median absolute deviation."""

    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 0.0
    median = np.median(values)
    return float(1.4826 * np.median(np.abs(values - median)))


def assess_window(
    raw: np.ndarray,
    poor_signal: Optional[np.ndarray] = None,
    config: FeatureConfig = DEFAULT_FEATURE_CONFIG,
    *,
    reference_sigma: Optional[float] = None,
) -> WindowQuality:
    """Decide whether a raw window is clean enough to be classified.

    ``reference_sigma`` is the typical spread of clean signal for this session;
    without it the window is judged against itself, which catches bursts but
    not a window that is noisy throughout.
    """

    raw = np.asarray(raw, dtype=np.float64)
    if raw.size == 0 or not np.all(np.isfinite(raw)):
        return WindowQuality(False, "non_finite")

    std = float(raw.std())
    poor_fraction = 0.0
    if poor_signal is not None and len(poor_signal):
        poor = np.asarray(poor_signal)
        poor_fraction = float(np.mean(poor > config.poor_signal_threshold))
    clip_fraction = float(np.mean(np.abs(raw) >= config.clip_level))

    sigma = reference_sigma if reference_sigma and reference_sigma > 0 else robust_sigma(raw)
    if sigma > 0:
        deviation = np.abs(raw - np.median(raw))
        artifact_fraction = float(np.mean(deviation > config.artifact_sigma * sigma))
    else:
        artifact_fraction = 0.0

    if poor_fraction > config.max_poor_fraction:
        reason = "poor_contact"
    elif std < config.min_std:
        reason = "flatline"
    elif clip_fraction > config.max_clip_fraction:
        reason = "clipping"
    elif artifact_fraction > config.max_artifact_fraction:
        reason = "artifact"
    else:
        reason = "ok"
    return WindowQuality(
        ok=reason == "ok",
        reason=reason,
        poor_fraction=poor_fraction,
        clip_fraction=clip_fraction,
        artifact_fraction=artifact_fraction,
        std=std,
    )


# ---------------------------------------------------------------------------
# Baseline normalisation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BandBaseline:
    """Typical log-amplitude of each band for one session."""

    mean: Tuple[float, ...]
    std: Tuple[float, ...]

    def as_dict(self) -> dict:
        return {"mean": list(self.mean), "std": list(self.std)}


def log_bands(band_matrix: np.ndarray) -> np.ndarray:
    """Log-compress band envelopes."""

    return np.log(np.maximum(np.asarray(band_matrix, dtype=np.float64), 0.0) + _LOG_FLOOR)


def _baseline_from_log(log_values: np.ndarray) -> BandBaseline:
    """Robust per-band location and spread; ``log_values`` is ``(bands, samples)``."""

    median = np.median(log_values, axis=1)
    spread = 1.4826 * np.median(np.abs(log_values - median[:, None]), axis=1)
    spread = np.maximum(spread, _MIN_BASELINE_STD)
    return BandBaseline(tuple(float(v) for v in median), tuple(float(v) for v in spread))


def compute_baseline(band_matrix: np.ndarray, mask: Optional[np.ndarray] = None) -> BandBaseline:
    """Baseline of a band matrix, optionally restricted to the samples in ``mask``."""

    values = log_bands(band_matrix)
    if mask is not None and mask.any() and mask.sum() >= 16:
        values = values[:, mask]
    return _baseline_from_log(values)


def pool_baselines(band_matrices: Iterable[np.ndarray]) -> BandBaseline:
    """One baseline for several recordings of the same sitting."""

    parts = [log_bands(matrix)[:, ::16] for matrix in band_matrices]
    if not parts:
        raise ValueError("pool_baselines needs at least one band matrix.")
    return _baseline_from_log(np.concatenate(parts, axis=1))


def normalise_bands(
    band_matrix: np.ndarray,
    baseline: Optional[BandBaseline] = None,
    *,
    clip: float = 6.0,
) -> np.ndarray:
    """Turn band envelopes into the standardised matrix fed to the models.

    With a ``baseline`` the result says how far each band is from that
    session's typical level; without one the window is standardised against
    itself.  Matrices containing negative values are not envelopes (legacy
    eight-channel data) and receive a plain per-channel z-score.
    """

    matrix = np.asarray(band_matrix, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("Band matrices must be 2-dimensional (bands, samples).")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Band matrix contains NaN or infinite values.")

    if matrix.size and float(matrix.min()) < 0.0:
        mean = matrix.mean(axis=1, keepdims=True)
        std = matrix.std(axis=1, keepdims=True)
        std[std < 1e-6] = 1.0
        result = (matrix - mean) / std
    else:
        values = log_bands(matrix)
        if baseline is None:
            baseline = _baseline_from_log(values)
        mean = np.asarray(baseline.mean)[:, None]
        std = np.asarray(baseline.std)[:, None]
        if mean.shape[0] != values.shape[0]:
            raise ValueError(
                f"Baseline describes {mean.shape[0]} bands but the matrix has {values.shape[0]}."
            )
        result = (values - mean) / std
    return np.clip(result, -clip, clip).astype(np.float32)


#: Rate (Hz) at which log band amplitudes are kept for baseline statistics.
BASELINE_RATE = 16.0


def _baseline_step(config: FeatureConfig) -> int:
    return max(1, int(round(config.sample_rate / BASELINE_RATE)))


def _block_means(log_values: np.ndarray, step: int) -> np.ndarray:
    """Average ``(bands, samples)`` over consecutive blocks of ``step`` samples."""

    usable = (log_values.shape[1] // step) * step
    return log_values[:, :usable].reshape(log_values.shape[0], -1, step).mean(axis=2)


class BaselineTracker:
    """Rolling session baseline for the live loop.

    Clean windows are added as they arrive; the baseline is the robust
    location and spread of the last ``history_seconds`` of them.  This is the
    live counterpart of :meth:`Recording.baseline` and lets the system adapt
    when the headset shifts or the wearer's state drifts during a session.
    """

    def __init__(
        self,
        config: FeatureConfig = DEFAULT_FEATURE_CONFIG,
        *,
        history_seconds: float = 180.0,
        min_seconds: float = 20.0,
    ) -> None:
        self.config = config
        self.min_seconds = float(min_seconds)
        self._step = _baseline_step(config)
        self._rate = config.sample_rate / self._step
        self._history: Deque[np.ndarray] = deque(maxlen=max(1, int(history_seconds * self._rate)))
        self._sigmas: Deque[float] = deque(maxlen=max(1, int(history_seconds)))
        self._cached: Optional[BandBaseline] = None

    @property
    def seconds(self) -> float:
        return len(self._history) / self._rate

    @property
    def ready(self) -> bool:
        return self.seconds >= self.min_seconds

    @property
    def reference_sigma(self) -> Optional[float]:
        """Typical raw-signal spread of the clean data seen so far."""

        if len(self._sigmas) < 3:
            return None
        return float(np.median(self._sigmas))

    def update(self, band_tail: np.ndarray, raw_tail: Optional[np.ndarray] = None) -> None:
        """Add newly arrived clean signal (do not pass overlapping samples twice)."""

        blocks = _block_means(log_bands(band_tail), self._step)
        if blocks.shape[1]:
            for column in blocks.T:
                self._history.append(column)
            self._cached = None
        if raw_tail is not None and len(raw_tail):
            sigma = robust_sigma(raw_tail)
            if sigma > 0:
                self._sigmas.append(sigma)

    def baseline(self) -> Optional[BandBaseline]:
        if not self._history:
            return None
        if self._cached is None:
            self._cached = _baseline_from_log(np.stack(self._history, axis=1))
        return self._cached


# ---------------------------------------------------------------------------
# Recordings on disk
# ---------------------------------------------------------------------------


@dataclass
class Recording:
    """A decoded recording, either raw signal or pre-computed band envelopes."""

    kind: str  # "raw" or "bands"
    config: FeatureConfig
    path: Optional[Path] = None
    raw: Optional[np.ndarray] = None
    poor_signal: Optional[np.ndarray] = None
    bands: Optional[np.ndarray] = None
    info: Dict[str, object] = field(default_factory=dict)
    _band_matrix: Optional[np.ndarray] = field(default=None, repr=False)
    _baseline: Optional[BandBaseline] = field(default=None, repr=False)
    _sigma: Optional[float] = field(default=None, repr=False)
    _blocks: Optional[np.ndarray] = field(default=None, repr=False)
    _block_mask: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def n_samples(self) -> int:
        if self.kind == "raw":
            assert self.raw is not None
            return int(self.raw.shape[0])
        assert self.bands is not None
        return int(self.bands.shape[1])

    @property
    def duration(self) -> float:
        return self.n_samples / self.config.sample_rate

    @property
    def name(self) -> str:
        return self.path.name if self.path is not None else "<memory>"

    def band_matrix(self) -> np.ndarray:
        """Band envelopes of the whole recording (for inspection, export and baselines)."""

        if self.kind == "bands":
            assert self.bands is not None
            return self.bands
        if self._band_matrix is None:
            assert self.raw is not None
            self._band_matrix = band_envelopes(self.raw, self.config)
        return self._band_matrix

    def clean_mask(self) -> Optional[np.ndarray]:
        """Samples with acceptable headset contact, or ``None`` if contact is unknown."""

        if self.poor_signal is None:
            return None
        return self.poor_signal <= self.config.poor_signal_threshold

    def baseline(self) -> BandBaseline:
        if self._baseline is None:
            self._baseline = compute_baseline(self.band_matrix(), self.clean_mask())
        return self._baseline

    def running_baseline(
        self,
        end: int,
        *,
        history_seconds: float = 180.0,
        min_seconds: float = 20.0,
    ) -> BandBaseline:
        """The baseline the live loop would hold at sample ``end``.

        It covers the ``history_seconds`` before ``end`` — like
        :class:`BaselineTracker` — and, near the start of a recording, the
        first ``min_seconds`` instead, mirroring the live calibration phase.
        Models that run live are trained on windows normalised this way.
        """

        step = _baseline_step(self.config)
        if self._blocks is None:
            blocks = _block_means(log_bands(self.band_matrix()), step)
            mask = self.clean_mask()
            if mask is not None and blocks.shape[1]:
                usable = blocks.shape[1] * step
                keep = mask[:usable].reshape(-1, step).mean(axis=1) > 0.5
                self._block_mask = keep if keep.sum() >= 8 else None
            self._blocks = blocks
        blocks = self._blocks
        if blocks.shape[1] == 0:
            return self.baseline()
        rate = self.config.sample_rate / step
        last = int(min(max(end / step, min_seconds * rate), blocks.shape[1]))
        first = int(max(0, min(end / step, last) - history_seconds * rate))
        first = min(first, max(0, last - int(min_seconds * rate)))
        values = blocks[:, first:last]
        if self._block_mask is not None:
            selected = self._block_mask[first:last]
            if selected.sum() >= 8:
                values = values[:, selected]
        if values.shape[1] < 2:
            return self.baseline()
        return _baseline_from_log(values)

    def reference_sigma(self) -> Optional[float]:
        if self.kind != "raw":
            return None
        if self._sigma is None:
            assert self.raw is not None
            mask = self.clean_mask()
            values = self.raw[mask] if mask is not None and mask.sum() > 16 else self.raw
            self._sigma = robust_sigma(values)
        return self._sigma

    def clamp_start(self, start: int, length: Optional[int] = None) -> int:
        """Nearest valid start index for a window of ``length`` samples."""

        length = int(length or self.config.segment_samples)
        if self.n_samples < length:
            raise RecordingError(
                f"{self.name} lasts {self.duration:.1f} s, shorter than one "
                f"{length / self.config.sample_rate:.1f} s window."
            )
        return int(min(max(0, start), self.n_samples - length))

    def window(self, start: int, length: Optional[int] = None) -> np.ndarray:
        """Un-normalised band matrix for the window starting at sample ``start``."""

        length = int(length or self.config.segment_samples)
        if start < 0 or start + length > self.n_samples:
            raise RecordingError(
                f"Window [{start}, {start + length}) is outside {self.name} "
                f"({self.n_samples} samples)."
            )
        if self.kind == "bands":
            assert self.bands is not None
            return self.bands[:, start : start + length]
        assert self.raw is not None
        return extract_band_window(self.raw, start + length, self.config, length=length)

    def window_quality(self, start: int, length: Optional[int] = None) -> WindowQuality:
        length = int(length or self.config.segment_samples)
        if self.kind != "raw":
            segment = self.bands[:, start : start + length]  # type: ignore[index]
            if not np.all(np.isfinite(segment)):
                return WindowQuality(False, "non_finite")
            if float(segment.std()) == 0.0:
                return WindowQuality(False, "flatline")
            return WindowQuality(True, "ok")
        assert self.raw is not None
        poor = None if self.poor_signal is None else self.poor_signal[start : start + length]
        return assess_window(
            self.raw[start : start + length],
            poor,
            self.config,
            reference_sigma=self.reference_sigma(),
        )

    def normalised_window(
        self,
        start: int,
        baseline: Optional[BandBaseline] = None,
        length: Optional[int] = None,
    ) -> np.ndarray:
        """Model-ready window; uses this recording's own baseline unless one is given."""

        if baseline is None and self.n_samples >= 2 * (length or self.config.segment_samples):
            baseline = self.baseline()
        return normalise_bands(self.window(start, length), baseline)


def decode_interleaved_int16(
    data: bytes,
    *,
    channels: int = 8,
    dtype: np.dtype | str = np.int16,
) -> np.ndarray:
    """Decode a legacy multi-channel dump (channels interleaved, little-endian integers).

    This is not what the headset produces; it remains for data that was already
    converted to eight channels elsewhere.  Values are scaled to ``[-1, 1]``.
    """

    dtype = np.dtype(dtype)
    usable = (len(data) // (dtype.itemsize * channels)) * dtype.itemsize * channels
    if usable == 0:
        raise RecordingError("The stream is too short to contain one sample per channel.")
    values = np.frombuffer(data[:usable], dtype=dtype.newbyteorder("<"))
    info = np.iinfo(dtype)
    scale = float(max(abs(info.min), abs(info.max)))
    return (values.reshape(-1, channels).T.astype(np.float32)) / scale


def _read_stream_bytes(path: Path) -> bytes:
    data = path.read_bytes()
    if not data:
        raise RecordingError(f"{path} is empty.")
    probe = data[:4096]
    printable = sum(1 for byte in probe if byte in b"0123456789abcdefABCDEFxX,;:.-[] \t\r\n")
    if printable >= 0.95 * len(probe):
        try:
            return hex_text_to_bytes(data.decode("ascii", errors="ignore"))
        except ValueError as error:  # pragma: no cover - defensive
            raise RecordingError(f"{path} is not valid hexadecimal text: {error}") from error
    return data


def recording_from_stream(
    data: bytes,
    config: FeatureConfig = DEFAULT_FEATURE_CONFIG,
    *,
    device_sample_rate: int | str | None = "auto",
    path: Optional[Path] = None,
    legacy_channels: int = 8,
) -> Recording:
    """Build a :class:`Recording` from the bytes of a headset stream."""

    label = str(path) if path is not None else "stream"
    if looks_like_thinkgear(data):
        decoded = decode_bytes(data)
        if decoded.raw.size == 0:
            raise RecordingError(f"{label} contains headset packets but no raw samples.")
        measured = estimate_sample_rate(decoded.power_positions)
        if device_sample_rate in (None, "auto"):
            device_sample_rate = config.device_sample_rate
        if device_sample_rate in (None, "auto"):
            device_rate = measured or config.sample_rate
        else:
            device_rate = int(device_sample_rate)
            if measured and abs(measured - device_rate) > 0.04 * device_rate:
                logger.warning(
                    "%s: configured device rate is %d Hz but the stream itself indicates %d Hz.",
                    label,
                    device_rate,
                    measured,
                )
        raw = resample_signal(decoded.raw, device_rate, config.sample_rate)
        poor = decoded.poor_signal
        if raw.shape[0] != poor.shape[0]:
            index = np.minimum(
                (np.arange(raw.shape[0]) * (device_rate / config.sample_rate)).astype(np.int64),
                poor.shape[0] - 1,
            )
            poor = poor[index]
        if np.all(poor == POOR_SIGNAL_UNKNOWN):
            poor_array: Optional[np.ndarray] = None
            poor_fraction = None
        else:
            poor_array = poor
            poor_fraction = float(np.mean(poor > config.poor_signal_threshold))
        stats = decoded.stats
        info: Dict[str, object] = {
            "format": "thinkgear",
            "device_sample_rate": int(device_rate),
            "measured_sample_rate": measured,
            "resampled": device_rate != config.sample_rate,
            "duration_seconds": round(raw.shape[0] / config.sample_rate, 3),
            "packets": stats.packets,
            "checksum_errors": stats.checksum_errors,
            "bytes_discarded": stats.bytes_discarded,
            "poor_signal_fraction": poor_fraction,
            "clip_fraction": float(np.mean(np.abs(decoded.raw) >= config.clip_level)),
        }
        return Recording(kind="raw", config=config, path=path, raw=raw, poor_signal=poor_array, info=info)

    bands = decode_interleaved_int16(data, channels=legacy_channels)
    info = {
        "format": "interleaved_int16",
        "duration_seconds": round(bands.shape[1] / config.sample_rate, 3),
        "note": "No headset packets found; decoded as interleaved little-endian int16 channels.",
    }
    return Recording(kind="bands", config=config, path=path, bands=bands, info=info)


_RECORDING_CACHE: "OrderedDict[tuple, Recording]" = OrderedDict()
_RECORDING_CACHE_SIZE = 64


def load_recording(
    path: Path | str,
    config: FeatureConfig = DEFAULT_FEATURE_CONFIG,
    *,
    device_sample_rate: int | str | None = "auto",
) -> Recording:
    """Load a recording from disk.

    Supported inputs:

    * hexadecimal text or binary headset streams (``.txt``, ``.hex``, ``.bin``);
    * ``.npy`` with one dimension: raw signal already at ``config.sample_rate``;
    * ``.npy`` with two dimensions: band envelopes shaped ``(bands, samples)``.
    """

    path = Path(path).expanduser()
    if not path.is_file():
        raise RecordingError(f"Recording not found: {path}")
    stat = path.stat()
    key = (str(path.resolve()), stat.st_mtime_ns, stat.st_size, config, str(device_sample_rate))
    cached = _RECORDING_CACHE.get(key)
    if cached is not None:
        _RECORDING_CACHE.move_to_end(key)
        return cached

    if path.suffix.lower() == ".npy":
        try:
            array = np.load(path, allow_pickle=False)
        except Exception as error:
            raise RecordingError(f"Could not read {path}: {error}") from error
        array = np.asarray(array, dtype=np.float32)
        if not np.all(np.isfinite(array)):
            raise RecordingError(f"{path} contains NaN or infinite values.")
        if array.ndim == 1:
            recording = Recording(
                kind="raw",
                config=config,
                path=path,
                raw=array,
                info={
                    "format": "npy_raw",
                    "duration_seconds": round(array.shape[0] / config.sample_rate, 3),
                },
            )
        elif array.ndim == 2:
            if array.shape[0] != config.num_bands and array.shape[1] == config.num_bands:
                array = array.T
            if array.shape[0] != config.num_bands:
                raise RecordingError(
                    f"{path} has shape {tuple(array.shape)}; expected ({config.num_bands}, samples)."
                )
            recording = Recording(
                kind="bands",
                config=config,
                path=path,
                bands=np.ascontiguousarray(array),
                info={
                    "format": "npy_bands",
                    "duration_seconds": round(array.shape[1] / config.sample_rate, 3),
                },
            )
        else:
            raise RecordingError(f"{path} has {array.ndim} dimensions; expected 1 or 2.")
    else:
        recording = recording_from_stream(
            _read_stream_bytes(path), config, device_sample_rate=device_sample_rate, path=path
        )

    if recording.n_samples == 0:
        raise RecordingError(f"{path} contains no samples.")

    _RECORDING_CACHE[key] = recording
    while len(_RECORDING_CACHE) > _RECORDING_CACHE_SIZE:
        _RECORDING_CACHE.popitem(last=False)
    return recording


RECORDING_SUFFIXES: Tuple[str, ...] = (".txt", ".hex", ".bin", ".npy")


def find_recordings(directory: Path | str, *, recursive: bool = False) -> List[Path]:
    """List recording files in ``directory`` in a stable order."""

    directory = Path(directory)
    if not directory.is_dir():
        return []
    pattern = "**/*" if recursive else "*"
    return sorted(
        path
        for path in directory.glob(pattern)
        if path.is_file() and path.suffix.lower() in RECORDING_SUFFIXES
    )


__all__ = [
    "BANDS",
    "BAND_NAMES",
    "BandBaseline",
    "BaselineTracker",
    "DEFAULT_FEATURE_CONFIG",
    "FeatureConfig",
    "RECORDING_SUFFIXES",
    "Recording",
    "RecordingError",
    "SAMPLE_RATE",
    "SEGMENT_SECONDS",
    "WindowQuality",
    "assess_window",
    "band_envelopes",
    "compute_baseline",
    "decode_interleaved_int16",
    "estimate_sample_rate",
    "extract_band_window",
    "find_recordings",
    "load_recording",
    "log_bands",
    "normalise_bands",
    "pool_baselines",
    "recording_from_stream",
    "resample_signal",
    "robust_sigma",
    "snap_sample_rate",
]
