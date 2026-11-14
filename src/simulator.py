"""A stand-in for the headset and the experiments, for testing without hardware.

The simulator produces single-channel EEG-like signal in the headset's own
format (ThinkGear packets as hexadecimal text).  The signal is band-limited
noise whose band amplitudes depend on a hidden state — resting, enjoying,
creatable, or looking at a particular artwork — on top of person-specific
differences, slow natural fluctuation, eye blinks and mains hum.

:func:`generate_demo_dataset` writes a complete miniature study in the layout
the pipeline expects, with relationships a working pipeline must be able to
recover: listeners enjoy particular moments, readers enjoy particular passages,
and the creator was in the creatable state while writing exactly those
passages.  The ground truth is saved next to the data so results can be checked.

Nothing here claims to model real brain activity.  It exists to exercise every
code path with data that has the right format and a known answer.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw
from scipy import signal as scipy_signal
from scipy.io import wavfile

from .eeg_features import BANDS
from .eeg_sources import GeneratorSource
from .parameters import ParameterCodec
from .thinkgear import bytes_to_hex_text, encode_stream

#: Typical amplitude (raw counts, RMS) of each band at rest.
BASE_AMPLITUDE = np.array([30.0, 18.0, 12.0, 12.0, 8.0, 6.0, 4.0, 3.0])

#: Multiplicative change of each band's amplitude in a given state.
STATE_GAINS: Dict[str, np.ndarray] = {
    "rest": np.ones(8),
    "enjoy": np.array([1.35, 1.9, 0.8, 0.8, 1.0, 1.0, 1.45, 1.0]),
    "creatable": np.array([0.9, 1.6, 1.0, 1.9, 0.7, 0.7, 1.0, 1.35]),
}


@dataclass(frozen=True)
class Segment:
    """A stretch of simulated signal in one hidden state."""

    duration: float
    state: str = "rest"
    #: 0 = indistinguishable from rest, 1 = the full effect of ``state``.
    intensity: float = 1.0
    #: Explicit band gains (overrides ``state``), e.g. for an artwork.
    gains: Optional[Tuple[float, ...]] = None
    #: Headset contact quality reported during the segment (0 good … 200 off the head).
    poor_signal: int = 0

    def band_gains(self) -> np.ndarray:
        if self.gains is not None:
            return np.asarray(self.gains, dtype=np.float64)
        return np.exp(self.intensity * np.log(STATE_GAINS[self.state]))


@dataclass(frozen=True)
class Person:
    """Individual differences: band profile, overall signal level, blink rate."""

    band_factors: Tuple[float, ...]
    level: float
    blink_interval: float

    @classmethod
    def random(cls, rng: np.random.Generator) -> "Person":
        return cls(
            band_factors=tuple(float(v) for v in np.exp(rng.normal(0.0, 0.2, 8))),
            level=float(np.exp(rng.normal(0.0, 0.25))),
            blink_interval=float(rng.uniform(4.0, 8.0)),
        )


def _band_noise(rng: np.random.Generator, samples: int, sample_rate: int) -> np.ndarray:
    """Unit-variance noise restricted to each band, shaped ``(8, samples)``."""

    pad = sample_rate
    white = rng.standard_normal((len(BANDS), samples + 2 * pad))
    out = np.empty((len(BANDS), samples))
    for index, (_name, low, high) in enumerate(BANDS):
        sos = scipy_signal.butter(3, [low, high], btype="bandpass", fs=sample_rate, output="sos")
        band = scipy_signal.sosfiltfilt(sos, white[index])[pad:-pad]
        out[index] = band / max(band.std(), 1e-9)
    return out


def _smooth_steps(values: np.ndarray, sample_rate: int, seconds: float = 0.5) -> np.ndarray:
    width = max(1, int(seconds * sample_rate))
    kernel = np.hanning(width + 2)[1:-1]
    kernel /= kernel.sum()
    padded = np.pad(values, ((0, 0), (width, width)), mode="edge")
    return np.stack([np.convolve(row, kernel, mode="same") for row in padded])[:, width:-width]


def simulate_raw(
    segments: Sequence[Segment],
    *,
    sample_rate: int = 640,
    seed: int = 0,
    person: Optional[Person] = None,
    variability: float = 0.22,
) -> np.ndarray:
    """Synthesise the raw signal for a sequence of segments (``int16`` counts)."""

    rng = np.random.default_rng(seed)
    person = person or Person.random(rng)
    lengths = [max(1, int(round(segment.duration * sample_rate))) for segment in segments]
    total = int(sum(lengths))

    gains = np.concatenate(
        [np.repeat(segment.band_gains()[:, None], length, axis=1) for segment, length in zip(segments, lengths)],
        axis=1,
    )
    gains = _smooth_steps(gains, sample_rate)

    # Slow, band-specific fluctuation: no two seconds of real signal are alike.
    knots = max(4, total // (2 * sample_rate) + 2)
    drift = rng.normal(0.0, variability, (len(BANDS), knots))
    positions = np.linspace(0, knots - 1, total)
    fluctuation = np.stack([np.interp(positions, np.arange(knots), row) for row in drift])

    amplitude = BASE_AMPLITUDE[:, None] * np.asarray(person.band_factors)[:, None] * gains * np.exp(fluctuation)
    signal = (amplitude * _band_noise(rng, total, sample_rate)).sum(axis=0) * person.level

    time = np.arange(total) / sample_rate
    signal += 4.0 * np.sin(2 * np.pi * 50.0 * time + rng.uniform(0, 2 * np.pi))
    signal += rng.normal(0.0, 2.0, total)

    # Eye blinks: large, slow deflections every few seconds.
    moment = rng.exponential(person.blink_interval)
    while moment < total / sample_rate:
        centre = int(moment * sample_rate)
        width = int(0.35 * sample_rate)
        lo, hi = max(0, centre - width), min(total, centre + width)
        shape = np.hanning(2 * width)[lo - (centre - width) : hi - (centre - width)]
        signal[lo:hi] += rng.uniform(120.0, 260.0) * person.level * shape
        moment += rng.exponential(person.blink_interval) + 0.5

    return np.clip(np.round(signal), -2047, 2047).astype(np.int16)


def simulate_stream(
    segments: Sequence[Segment],
    *,
    sample_rate: int = 640,
    seed: int = 0,
    person: Optional[Person] = None,
) -> bytes:
    """The byte stream a headset would transmit for these segments."""

    raw = simulate_raw(segments, sample_rate=sample_rate, seed=seed, person=person)
    quality: List[int] = []
    for segment in segments:
        quality.extend([segment.poor_signal] * max(1, int(round(segment.duration))))
    if any(segment.poor_signal > 100 for segment in segments):
        # With the headset off the head the electrode picks up large noise.
        rng = np.random.default_rng(seed + 1)
        cursor = 0
        for segment in segments:
            length = max(1, int(round(segment.duration * sample_rate)))
            if segment.poor_signal > 100:
                raw[cursor : cursor + length] = np.clip(rng.normal(0, 700, length), -2047, 2047).astype(np.int16)
            cursor += length
    return encode_stream(raw, sample_rate, poor_signal=quality)


def write_hex_recording(
    path: Path | str,
    segments: Sequence[Segment],
    *,
    sample_rate: int = 640,
    seed: int = 0,
    person: Optional[Person] = None,
) -> Path:
    """Write a simulated recording as a hexadecimal text log."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = simulate_stream(segments, sample_rate=sample_rate, seed=seed, person=person)
    path.write_text(bytes_to_hex_text(stream), encoding="ascii")
    return path


# ---------------------------------------------------------------------------
# Live source
# ---------------------------------------------------------------------------

#: Hidden states of the simulated live session, repeated for as long as it runs.
LIVE_SCENARIO: Tuple[Segment, ...] = (
    Segment(40.0, "rest"),
    Segment(45.0, "creatable"),
    Segment(35.0, "rest"),
    Segment(45.0, "creatable"),
)


def scenario_state_at(seconds: float, scenario: Sequence[Segment] = LIVE_SCENARIO) -> str:
    """Hidden state of the looping scenario at a given time."""

    period = sum(segment.duration for segment in scenario)
    position = seconds % period
    for segment in scenario:
        if position < segment.duration:
            return segment.state
        position -= segment.duration
    return scenario[-1].state


def simulated_source(
    *,
    seed: int = 0,
    speed: float = 1.0,
    sample_rate: int = 640,
    scenario: Sequence[Segment] = LIVE_SCENARIO,
    loops: Optional[int] = None,
) -> GeneratorSource:
    """A live source that alternates between rest and the creatable state.

    ``speed`` 1.0 runs in real time, 0 as fast as the loop can process.
    """

    import time

    def generate() -> Iterator[bytes]:
        rng = np.random.default_rng(seed)
        person = Person.random(rng)
        cycle = 0
        started = time.monotonic()
        emitted = 0.0
        while loops is None or cycle < loops:
            stream = simulate_stream(scenario, sample_rate=sample_rate, seed=seed + 1000 * cycle, person=person)
            seconds = sum(segment.duration for segment in scenario)
            step = max(64, int(len(stream) / seconds * 0.1))
            for offset in range(0, len(stream), step):
                if speed > 0:
                    delay = started + emitted / speed - time.monotonic()
                    if delay > 0:
                        time.sleep(delay)
                emitted += 0.1
                yield stream[offset : offset + step]
            cycle += 1

    return GeneratorSource(generate, description=f"simulator (seed {seed})")


# ---------------------------------------------------------------------------
# Demo study
# ---------------------------------------------------------------------------

_WORDS = (
    "light water stone garden window silence river morning shadow paper voice winter "
    "glass thread harbour ember meadow lantern echo salt orchard dusk feather current "
    "threshold rain copper mirror breath field ash signal tide velvet hollow ridge"
).split()


def _passage(rng: np.random.Generator, sentences: int) -> str:
    parts = []
    for _ in range(sentences):
        words = [str(rng.choice(_WORDS)) for _ in range(int(rng.integers(7, 14)))]
        parts.append(" ".join(words).capitalize() + ".")
    return " ".join(parts)


def _music(rng: np.random.Generator, seconds: float, climaxes: Sequence[Tuple[float, float]], rate: int = 22050) -> np.ndarray:
    """A plain melody that swells during the given intervals."""

    time = np.arange(int(seconds * rate)) / rate
    scale = np.array([220.0, 246.9, 277.2, 329.6, 370.0, 440.0, 493.9, 554.4])
    audio = np.zeros_like(time)
    for note_start in np.arange(0.0, seconds, 0.5):
        frequency = float(rng.choice(scale))
        inside = (time >= note_start) & (time < note_start + 0.5)
        local = time[inside] - note_start
        audio[inside] += np.sin(2 * np.pi * frequency * local) * np.exp(-3.0 * local)
    loudness = np.full_like(time, 0.25)
    for start, end in climaxes:
        loudness += 0.6 * np.exp(-0.5 * ((time - (start + end) / 2.0) / max((end - start) / 2.0, 0.5)) ** 2)
        swell = (time >= start) & (time < end)
        audio[swell] += 0.5 * np.sin(2 * np.pi * 660.0 * time[swell])
    audio *= loudness
    return (audio / max(np.abs(audio).max(), 1e-9) * 0.9 * 32767).astype(np.int16)


def _artwork(rng: np.random.Generator, style: int, size: Tuple[int, int] = (480, 360)) -> Image.Image:
    """An abstract composition; each style differs clearly in colour and texture."""

    width, height = size
    hue = rng.uniform(0, 360)

    def colour(h: float, s: float, v: float) -> Tuple[int, int, int]:
        import colorsys

        r, g, b = colorsys.hsv_to_rgb((h % 360) / 360.0, s, v)
        return int(r * 255), int(g * 255), int(b * 255)

    kind = style % 6
    if kind == 0:  # large flat blocks on a pale ground
        image = Image.new("RGB", size, colour(hue, 0.08, 0.95))
        draw = ImageDraw.Draw(image)
        for index in range(5):
            x, y = rng.integers(0, width - 120), rng.integers(0, height - 100)
            draw.rectangle([x, y, x + rng.integers(80, 200), y + rng.integers(60, 160)], fill=colour(hue + 90 * index, 0.85, 0.9))
    elif kind == 1:  # concentric circles on dark
        image = Image.new("RGB", size, colour(hue, 0.6, 0.15))
        draw = ImageDraw.Draw(image)
        for _ in range(9):
            cx, cy = rng.integers(40, width - 40), rng.integers(40, height - 40)
            for ring, radius in enumerate(range(int(rng.integers(40, 70)), 6, -10)):
                draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=colour(hue + 47 * ring, 0.9, 0.95))
    elif kind == 2:  # fine horizontal stripes
        image = Image.new("RGB", size, colour(hue, 0.3, 0.5))
        draw = ImageDraw.Draw(image)
        for y in range(0, height, 6):
            draw.rectangle([0, y, width, y + 2], fill=colour(hue + rng.uniform(-25, 25), 0.55, rng.uniform(0.6, 1.0)))
    elif kind == 3:  # dense speckle
        pixels = rng.integers(0, 256, (height // 3, width // 3, 3), dtype=np.uint8)
        image = Image.fromarray(pixels).resize(size, Image.Resampling.NEAREST)
    elif kind == 4:  # soft gradient
        ramp = np.linspace(0, 1, width)[None, :] * np.ones((height, 1))
        top, bottom = np.array(colour(hue, 0.5, 0.9)), np.array(colour(hue + 40, 0.6, 0.4))
        image = Image.fromarray((top * (1 - ramp[..., None]) + bottom * ramp[..., None]).astype(np.uint8))
    else:  # slanted bars, strong contrast
        image = Image.new("RGB", size, (245, 240, 230))
        draw = ImageDraw.Draw(image)
        for index in range(7):
            x = rng.integers(0, width)
            draw.polygon(
                [(x, 0), (x + 40, 0), (x - 120, height), (x - 160, height)],
                fill=colour(hue + 180 * (index % 2), 0.9, 0.35 + 0.5 * (index % 2)),
            )
    return image


_ARTWORK_MIXING = np.random.default_rng(20251025).normal(0.0, 1.0, (8, ParameterCodec().dim))


def artwork_band_gains(parameters: Dict[str, float], strength: float = 0.55) -> np.ndarray:
    """Band gains evoked by an artwork, as a fixed function of its parameters."""

    vector = ParameterCodec().encode(parameters)
    return np.exp(strength * np.tanh(_ARTWORK_MIXING @ vector / math.sqrt(vector.size) * 2.5))


def generate_demo_dataset(
    root: Path | str,
    *,
    seed: int = 0,
    sample_rate: int = 640,
    listeners: int = 4,
    pieces: int = 2,
    readers: int = 4,
    passages: int = 12,
    artworks: int = 6,
    viewers: int = 4,
    live_seconds: float = 330.0,
) -> Dict[str, object]:
    """Write a complete simulated study under ``root`` and return its ground truth.

    The layout matches ``data/README.md``.  ``demo_truth.json`` records which
    passages were made enjoyable and when the simulated live session was in
    the creatable state.
    """

    from .processing_features import analyse_image

    root = Path(root)
    rng = np.random.default_rng(seed)
    truth: Dict[str, object] = {"seed": seed, "sample_rate": sample_rate}
    next_seed = iter(range(seed * 10007 + 1, seed * 10007 + 100000))

    # --- Step 1: a couple of plain recordings to convert -------------------------
    step1 = root / "step1_hex_eeg"
    for index in range(2):
        write_hex_recording(
            step1 / f"session_{index + 1:03d}.txt",
            [Segment(22.0, "rest")],
            sample_rate=sample_rate,
            seed=next(next_seed),
        )

    # --- Step 2: listening sessions ------------------------------------------------
    step2 = root / "step2_viewer_training"
    piece_seconds = 80.0
    piece_climaxes: List[List[Tuple[float, float]]] = []
    for piece in range(pieces):
        starts = sorted(rng.choice(np.arange(8.0, piece_seconds - 14.0, 9.0), size=4, replace=False).tolist())
        climaxes = [(float(s), float(s) + 6.0) for s in starts]
        piece_climaxes.append(climaxes)
        (step2 / "audio").mkdir(parents=True, exist_ok=True)
        wavfile.write(step2 / "audio" / f"piece_{piece + 1}.wav", 22050, _music(rng, piece_seconds, climaxes))

    people = [Person.random(rng) for _ in range(max(listeners, readers, viewers) + 1)]
    sessions = []
    for listener in range(listeners):
        for piece in range(pieces):
            # Each listener is moved by most, not all, of a piece's high points.
            enjoyed = [c for c in piece_climaxes[piece] if rng.random() < 0.8] or piece_climaxes[piece][:1]
            segments: List[Segment] = []
            cursor = 0.0
            for start, end in enjoyed:
                segments.append(Segment(start - cursor, "rest"))
                segments.append(Segment(end - start, "enjoy"))
                cursor = end
            segments.append(Segment(piece_seconds - cursor, "rest"))
            name = f"listener{listener + 1:02d}_piece{piece + 1}.txt"
            write_hex_recording(step2 / "eeg" / name, segments, sample_rate=sample_rate, seed=next(next_seed), person=people[listener])
            sessions.append(
                {
                    "eeg_file": f"step2_viewer_training/eeg/{name}",
                    "audio_file": f"step2_viewer_training/audio/piece_{piece + 1}.wav",
                    "participant": f"listener{listener + 1:02d}",
                    "enjoyed": [[round(s, 2), round(e, 2)] for s, e in enjoyed],
                }
            )
    (step2 / "annotations").mkdir(parents=True, exist_ok=True)
    (step2 / "annotations" / "annotations.json").write_text(json.dumps({"sessions": sessions}, indent=2), encoding="utf8")

    # --- Steps 3 and 4: an article, its readers, and the creator writing it ----------
    texts = [_passage(rng, int(rng.integers(2, 5))) for _ in range(passages)]
    order = rng.permutation(passages)
    excellent = sorted(int(i) for i in order[: passages // 2])
    quality = [1.0 if index in excellent else 0.0 for index in range(passages)]
    truth["excellent_passages"] = excellent

    step3 = root / "step3_article_reading"
    (step3 / "eeg").mkdir(parents=True, exist_ok=True)
    (step3 / "article.txt").write_text("\n\n".join(texts) + "\n", encoding="utf8")
    lengths = np.array([len(text) for text in texts], dtype=np.float64)
    for reader in range(readers):
        seconds_per_character = rng.uniform(0.085, 0.11)
        segments = [
            Segment(float(lengths[i] * seconds_per_character), "enjoy" if quality[i] else "rest")
            for i in range(passages)
        ]
        write_hex_recording(
            step3 / "eeg" / f"reader{reader + 1:02d}.txt",
            segments,
            sample_rate=sample_rate,
            seed=next(next_seed),
            person=people[reader],
        )

    step4 = root / "step4_creator_training"
    (step4 / "eeg").mkdir(parents=True, exist_ok=True)
    (step4 / "annotations").mkdir(parents=True, exist_ok=True)
    creator = people[-1]
    truth["creator"] = {"band_factors": list(creator.band_factors), "level": creator.level, "blink_interval": creator.blink_interval}
    segments = []
    events = [{"t": 0.0, "length": 0}]
    clock, written = 0.0, 0
    for index in range(passages):
        duration = float(lengths[index] * rng.uniform(0.2, 0.26))
        segments.append(Segment(duration, "creatable" if quality[index] else "rest"))
        characters = int(lengths[index]) + 2
        steps = max(2, int(duration // 2))
        for step in range(1, steps + 1):
            events.append({"t": round(clock + duration * step / steps, 2), "length": written + characters * step // steps})
        clock += duration
        written += characters
    write_hex_recording(step4 / "eeg" / "creator_session_001.txt", segments, sample_rate=sample_rate, seed=next(next_seed), person=creator)
    (step4 / "eeg" / "creator_session_001_keystrokes.jsonl").write_text(
        "\n".join(json.dumps(event) for event in events) + "\n", encoding="utf8"
    )
    (step4 / "annotations" / "creator_session_001.json").write_text(
        json.dumps(
            {
                "session_id": "creator_session_001",
                "eeg_file": "step4_creator_training/eeg/creator_session_001.txt",
                "keystroke_log": "step4_creator_training/eeg/creator_session_001_keystrokes.jsonl",
                "article": "step3_article_reading/article.txt",
            },
            indent=2,
        ),
        encoding="utf8",
    )

    # --- Step 5: viewing artworks ----------------------------------------------------
    step5 = root / "step5_image_training"
    (step5 / "artworks").mkdir(parents=True, exist_ok=True)
    gains: Dict[str, np.ndarray] = {}
    for index in range(artworks):
        name = f"artwork_{index + 1:02d}"
        image = _artwork(rng, index)
        image.save(step5 / "artworks" / f"{name}.png")
        gains[name] = artwork_band_gains(dict(analyse_image(image)))
    for viewer in range(viewers):
        for name, band_gains in gains.items():
            write_hex_recording(
                step5 / "brainwaves" / f"viewer{viewer + 1:02d}" / f"{name}.txt",
                [Segment(10.0, gains=tuple(float(g) for g in band_gains))],
                sample_rate=sample_rate,
                seed=next(next_seed),
                person=people[viewer],
            )

    # --- Step 6: a live session of the creator with a known course --------------------
    step6 = root / "step6_realtime"
    scenario: List[Segment] = []
    while sum(segment.duration for segment in scenario) < live_seconds:
        scenario.extend(LIVE_SCENARIO)
    write_hex_recording(step6 / "demo_live_session.txt", scenario, sample_rate=sample_rate, seed=next(next_seed), person=creator)
    timeline, cursor = [], 0.0
    for segment in scenario:
        timeline.append({"start": cursor, "end": cursor + segment.duration, "state": segment.state})
        cursor += segment.duration
    truth["live_session"] = timeline

    (root / "demo_truth.json").write_text(json.dumps(truth, indent=2), encoding="utf8")
    return truth


__all__ = [
    "BASE_AMPLITUDE",
    "LIVE_SCENARIO",
    "Person",
    "STATE_GAINS",
    "Segment",
    "artwork_band_gains",
    "generate_demo_dataset",
    "scenario_state_at",
    "simulate_raw",
    "simulate_stream",
    "simulated_source",
    "write_hex_recording",
]
