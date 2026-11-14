"""The live feedback loop (Step 6).

:class:`FeedbackController` holds the decision logic: given one normalised
five-second window it says whether the creator is in a creatable state and, if
so, which parameters the display should take.

:class:`RealtimeSession` wraps it with everything a real session needs:

* decoding the headset's byte stream and measuring its true sampling rate,
* a rolling window that is evaluated every ``update_interval`` seconds,
* a quality gate (poor contact, clipping, flat line, movement artefacts) —
  windows that fail are never classified,
* a session baseline that calibrates at the start and then adapts slowly,
* hysteresis and smoothing so the beacon does not flicker,
* recovery from a stalled or reconnecting device,
* a recording of the session (raw EEG in the headset's own hex format plus a
  log of every decision), which can be fed straight back into training.

Throughout, the rule from the project description holds: parameters are
refreshed only while the state is creatable; otherwise the display keeps the
last set it was given.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from torch import Tensor, nn

from .eeg_features import (
    DEFAULT_FEATURE_CONFIG,
    BandBaseline,
    BaselineTracker,
    FeatureConfig,
    anchored_baseline,
    assess_window,
    band_envelopes,
    normalise_bands,
    resample_signal,
    snap_sample_rate,
)
from .eeg_sources import EEGSource
from .parameters import ParameterCodec, blend_parameters
from .thinkgear import ThinkGearParser, bytes_to_hex_text
from .training import resolve_device

logger = logging.getLogger(__name__)

STATE_CREATABLE = "creatable"
STATE_IDLE = "idle"
STATE_CALIBRATING = "calibrating"
STATE_BAD_SIGNAL = "bad_signal"
STATE_NO_SIGNAL = "no_signal"
STATE_RECORDING = "recording"


@dataclass
class LoopSettings:
    """Behaviour of the feedback loop."""

    #: Probability at which the creatable state is entered.
    creatable_threshold: float = 0.6
    #: Probability below which it is left again (default: threshold − 0.1).
    release_threshold: Optional[float] = None
    #: Weight of the newest probability in the running average (1.0 = no smoothing).
    probability_smoothing: float = 0.6
    #: How far each update moves the parameters towards the new prediction (1.0 = jump).
    parameter_smoothing: float = 0.6
    #: Seconds between evaluations.
    update_interval: float = 5.0
    #: Clean signal required before the first decision is made.
    calibration_seconds: float = 20.0
    #: Length of the adaptive baseline.
    baseline_seconds: float = 180.0
    #: Silence after which the device counts as disconnected.
    stall_seconds: float = 3.0

    def release(self) -> float:
        if self.release_threshold is not None:
            return min(self.release_threshold, self.creatable_threshold)
        return max(0.0, self.creatable_threshold - 0.1)


@dataclass
class Decision:
    """Result of evaluating one window."""

    state: str
    is_creatable: bool = False
    probability: Optional[float] = None
    smoothed_probability: Optional[float] = None
    #: A fresh parameter set to send to the display, if the state is creatable.
    parameters: Optional[Dict[str, float]] = None
    #: What the display is currently showing (the last set sent).
    held_parameters: Optional[Dict[str, float]] = None
    reason: str = ""

    def as_record(self) -> Dict[str, Any]:
        return {
            "state": self.state,
            "is_creatable": self.is_creatable,
            "probability": None if self.probability is None else round(self.probability, 4),
            "smoothed_probability": None
            if self.smoothed_probability is None
            else round(self.smoothed_probability, 4),
            "updated": self.parameters is not None,
            "reason": self.reason,
        }


class FeedbackController:
    """Turns classified windows into display updates."""

    def __init__(
        self,
        creator_model: nn.Module,
        parameter_model: nn.Module,
        settings: Optional[LoopSettings] = None,
        *,
        device: str | torch.device | None = None,
    ) -> None:
        self.settings = settings or LoopSettings()
        self.device = resolve_device(device)
        self.creator_model = creator_model.to(self.device).eval()
        self.parameter_model = parameter_model.to(self.device).eval()
        self.codec: ParameterCodec = getattr(parameter_model, "codec", None) or ParameterCodec()
        self.parameter_names = tuple(self.codec.names)
        self.is_creatable = False
        self.smoothed: Optional[float] = None
        self.last_parameters: Optional[Dict[str, float]] = None
        reference = getattr(creator_model, "reference", None)
        self.reference: Optional[BandBaseline] = reference() if callable(reference) else None
        self.baseline_adaptation = DEFAULT_FEATURE_CONFIG.baseline_adaptation

    def baseline_for(self, session: Optional[BandBaseline]) -> Optional[BandBaseline]:
        """The baseline to normalise a live window with.

        When the creator model carries the creator's reference profile, the
        session baseline only corrects it for gain and slow drift; otherwise
        the session baseline is used as it is.
        """

        if session is None or self.reference is None:
            return session
        return anchored_baseline(session, self.reference, self.baseline_adaptation)

    def _probability(self, window: Tensor) -> float:
        with torch.no_grad():
            if hasattr(self.creator_model, "predict_proba"):
                return float(self.creator_model.predict_proba(window).flatten()[0])
            return float(torch.sigmoid(self.creator_model(window)).flatten()[0])

    def _parameters(self, window: Tensor) -> Dict[str, float]:
        with torch.no_grad():
            if hasattr(self.parameter_model, "predict_parameters"):
                return self.parameter_model.predict_parameters(window)[0]
            return self.codec.decode(self.parameter_model(window)[0].cpu().numpy())

    def step(self, eeg_window: np.ndarray | Tensor) -> Decision:
        """Evaluate one normalised ``(8, 3200)`` window."""

        window = torch.as_tensor(eeg_window, dtype=torch.float32)
        if window.dim() == 2:
            window = window.unsqueeze(0)
        if not bool(torch.isfinite(window).all()):
            return self.hold(STATE_BAD_SIGNAL, "non_finite")
        window = window.to(self.device)

        probability = self._probability(window)
        if not np.isfinite(probability):
            return self.hold(STATE_BAD_SIGNAL, "model_output")
        weight = min(max(self.settings.probability_smoothing, 0.05), 1.0)
        self.smoothed = probability if self.smoothed is None else weight * probability + (1 - weight) * self.smoothed

        threshold = self.settings.release() if self.is_creatable else self.settings.creatable_threshold
        self.is_creatable = self.smoothed >= threshold

        if not self.is_creatable:
            return Decision(
                STATE_IDLE,
                False,
                probability,
                self.smoothed,
                None,
                self.last_parameters,
            )

        weight = min(max(self.settings.parameter_smoothing, 0.05), 1.0)
        self.last_parameters = blend_parameters(self.last_parameters, self._parameters(window), weight)
        return Decision(
            STATE_CREATABLE,
            True,
            probability,
            self.smoothed,
            dict(self.last_parameters),
            self.last_parameters,
        )

    def hold(self, state: str, reason: str = "") -> Decision:
        """Report that no decision can be made; the display keeps its parameters."""

        self.is_creatable = False
        if state == STATE_NO_SIGNAL:
            self.smoothed = None  # after an outage, start the running average afresh
        return Decision(state, False, None, self.smoothed, None, self.last_parameters, reason)


class PassiveController:
    """Stands in for :class:`FeedbackController` when a session is only recorded.

    Used for the very first writing session, before any model exists: the EEG
    is checked and recorded as usual, but nothing is classified or displayed.
    """

    parameter_names: tuple = ()

    def __init__(self, settings: Optional[LoopSettings] = None) -> None:
        self.settings = settings or LoopSettings()

    def step(self, eeg_window: np.ndarray | Tensor) -> Decision:
        return Decision(STATE_RECORDING)

    def baseline_for(self, session: Optional[BandBaseline]) -> Optional[BandBaseline]:
        return session

    def hold(self, state: str, reason: str = "") -> Decision:
        return Decision(state, reason=reason)


# ---------------------------------------------------------------------------
# Session recording
# ---------------------------------------------------------------------------


class SessionRecorder:
    """Writes a live session to disk.

    ``eeg.txt``
        the raw byte stream as hexadecimal text, the same format the headset
        logs use, so the file can be used as training data as it is;
    ``events.jsonl``
        one line per evaluated window: time, state, probability, signal quality
        and the parameters sent;
    ``keystrokes.jsonl``
        how the watched document grew (only with ``watch_document``);
    ``session.json``
        settings, the measured sampling rate and a summary.
    """

    def __init__(self, directory: Path | str, *, watch_document: Path | str | None = None) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.started_at = time.time()
        self._eeg = open(self.directory / "eeg.txt", "w", encoding="ascii")
        self._events = open(self.directory / "events.jsonl", "w", encoding="utf8")
        self._column = 0
        self._last_flush = time.monotonic()
        self._metadata: Dict[str, Any] = {
            "started_at": self.started_at,
            "started": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self.started_at)),
            "eeg_file": "eeg.txt",
        }
        self._watcher: Optional[DocumentWatcher] = None
        if watch_document is not None:
            self._watcher = DocumentWatcher(watch_document, self.directory / "keystrokes.jsonl", self.started_at)
            self._watcher.start()
            self._metadata["keystroke_log"] = "keystrokes.jsonl"
            self._metadata["document"] = str(watch_document)
        self._write_metadata()

    def write_bytes(self, data: bytes) -> None:
        if not data:
            return
        pairs = bytes_to_hex_text(data, bytes_per_line=1 << 30).split()
        lines: List[str] = []
        index = 0
        while index < len(pairs):
            take = min(32 - self._column, len(pairs) - index)
            piece = " ".join(pairs[index : index + take])
            index += take
            self._column += take
            if self._column >= 32:
                lines.append(piece + "\n")
                self._column = 0
            else:
                lines.append(piece + " ")
        self._eeg.write("".join(lines))
        if time.monotonic() - self._last_flush > 2.0:
            self.flush()

    def log_event(self, event: Dict[str, Any]) -> None:
        self._events.write(json.dumps(event) + "\n")
        self._events.flush()

    def update_metadata(self, **values: Any) -> None:
        self._metadata.update(values)
        self._write_metadata()

    def flush(self) -> None:
        self._eeg.flush()
        self._last_flush = time.monotonic()

    def _write_metadata(self) -> None:
        temporary = self.directory / "session.json.tmp"
        with open(temporary, "w", encoding="utf8") as handle:
            json.dump(self._metadata, handle, indent=2)
        temporary.replace(self.directory / "session.json")

    def close(self) -> None:
        if self._watcher is not None:
            self._watcher.stop()
            self._watcher = None
        for handle in (self._eeg, self._events):
            if not handle.closed:
                handle.close()


class DocumentWatcher(threading.Thread):
    """Logs how a text file grows while the creator writes in any editor.

    The file is polled once a second; whenever its content changes (the editor
    saved or auto-saved), the time and the text are appended to the log.  The
    result is a keystroke log that Step 4 can align with the EEG.
    """

    def __init__(self, document: Path | str, log_path: Path | str, started_at: float, interval: float = 1.0) -> None:
        super().__init__(daemon=True, name="document-watcher")
        self.document = Path(document)
        self.log_path = Path(log_path)
        self.started_at = started_at
        self.interval = interval
        self._stop_event = threading.Event()
        self._last_text: Optional[str] = None
        self.log_path.touch()

    def poll(self) -> bool:
        """Check the document once; returns ``True`` if a change was logged."""

        try:
            text = self.document.read_text(encoding="utf8", errors="replace")
        except OSError:
            return False
        if text == self._last_text:
            return False
        self._last_text = text
        entry = {"t": round(time.time() - self.started_at, 3), "length": len(text), "text": text}
        with open(self.log_path, "a", encoding="utf8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return True

    def run(self) -> None:
        while not self._stop_event.is_set():
            self.poll()
            self._stop_event.wait(self.interval)

    def stop(self) -> None:
        self._stop_event.set()
        if self.is_alive():
            self.join(timeout=2.0)
        self.poll()


# ---------------------------------------------------------------------------
# The live session
# ---------------------------------------------------------------------------


@dataclass
class SessionSummary:
    """What happened during a live session."""

    stream_seconds: float = 0.0
    device_sample_rate: Optional[int] = None
    windows: int = 0
    updates_sent: int = 0
    states: Dict[str, int] = field(default_factory=dict)
    checksum_errors: int = 0
    reconnects: int = 0
    session_directory: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RealtimeSession:
    """Runs the feedback loop from a live byte source to the display."""

    def __init__(
        self,
        source: EEGSource,
        controller: "FeedbackController | PassiveController",
        *,
        feature_config: FeatureConfig = DEFAULT_FEATURE_CONFIG,
        bridge: Any = None,
        recorder: Optional[SessionRecorder] = None,
        device_sample_rate: int | str | None = "auto",
        on_decision: Optional[Callable[[float, Decision], None]] = None,
    ) -> None:
        self.source = source
        self.controller = controller
        self.settings = controller.settings
        self.config = feature_config
        if hasattr(controller, "baseline_adaptation"):
            controller.baseline_adaptation = feature_config.baseline_adaptation
        self.bridge = bridge
        self.recorder = recorder
        self.on_decision = on_decision

        self._parser = ThinkGearParser()
        if device_sample_rate in (None, "auto"):
            device_sample_rate = feature_config.device_sample_rate
        self._fixed_rate = None if device_sample_rate in (None, "auto") else int(device_sample_rate)  # type: ignore[arg-type]
        self._rate: Optional[int] = None
        self._power_positions: List[int] = []
        self._first_sample_wall: Optional[float] = None
        self._last_data_wall = time.monotonic()
        self._last_stall_report = 0.0
        self._stalled = False

        self._raw = np.empty(0, dtype=np.float32)
        self._poor = np.empty(0, dtype=np.int16)
        self._buffer_start = 0  # absolute index of self._raw[0]
        self._next_tick = 0
        self._tracker = BaselineTracker(
            feature_config,
            history_seconds=self.settings.baseline_seconds,
            min_seconds=self.settings.calibration_seconds,
        )
        self._previous_state: Optional[str] = None
        self.summary = SessionSummary(
            session_directory=str(recorder.directory) if recorder is not None else None
        )
        self._states: Counter = Counter()

    # -- sampling rate ------------------------------------------------------------
    def _decide_rate(self) -> None:
        """Settle the device's sampling rate from the evidence available so far."""

        if self._rate is not None:
            return
        if self._fixed_rate is not None:
            self._set_rate(self._fixed_rate, "configured")
            return
        if len(self._power_positions) >= 6:
            intervals = np.diff(self._power_positions)
            self._set_rate(snap_sample_rate(float(np.median(intervals))), "measured from the headset's status packets")
            return
        total = self._parser.total_samples
        if self.source.is_live:
            if self._first_sample_wall is None:
                return
            elapsed = time.monotonic() - self._first_sample_wall
            if elapsed >= 10.0 and not self._power_positions:
                measured = snap_sample_rate(total / elapsed)
                if abs(measured - self.config.sample_rate) <= 0.03 * self.config.sample_rate or measured in (256, 512, 1024):
                    self._set_rate(measured, "measured from the arrival rate")
                else:
                    self._set_rate(self.config.sample_rate, f"assumed (arrival rate of {total / elapsed:.0f} Hz is implausible)")
        elif total >= 6 * self.config.sample_rate and not self._power_positions:
            self._set_rate(self.config.sample_rate, "assumed (the stream has no status packets)")

    def _set_rate(self, rate: int, how: str) -> None:
        self._rate = int(rate)
        self.summary.device_sample_rate = self._rate
        self._window = int(np.ceil(self.config.segment_seconds * self._rate))
        self._context = int(round(self.config.context_seconds * self._rate))
        self._hop = max(1, int(round(self.settings.update_interval * self._rate)))
        self._next_tick = self._buffer_start + self._window
        note = "" if self._rate == self.config.sample_rate else f"; resampling to {self.config.sample_rate} Hz"
        logger.info("Headset sampling rate: %d Hz (%s)%s", self._rate, how, note)
        if self.recorder is not None:
            self.recorder.update_metadata(device_sample_rate=self._rate, sample_rate_source=how)

    # -- main loop ------------------------------------------------------------------
    def run(self, *, duration: Optional[float] = None, max_updates: Optional[int] = None) -> SessionSummary:
        """Process the source until it ends, ``duration`` seconds of signal have
        been handled, ``max_updates`` windows were evaluated, or Ctrl+C."""

        logger.info("Reading EEG from %s", self.source.description)
        wall_start = time.monotonic()
        try:
            for chunk in self.source.chunks():
                now = time.monotonic()
                if chunk:
                    if self.recorder is not None:
                        self.recorder.write_bytes(chunk)
                    parsed = self._parser.feed(chunk)
                    self._power_positions.extend(parsed.power_positions)
                    if parsed.raw.size:
                        if self._first_sample_wall is None:
                            self._first_sample_wall = now
                        if self._stalled:
                            logger.info("Signal is back.")
                            self._stalled = False
                            self.summary.reconnects += 1
                        self._last_data_wall = now
                        self._raw = np.concatenate([self._raw, parsed.raw.astype(np.float32)])
                        self._poor = np.concatenate([self._poor, parsed.poor_signal])
                        self._decide_rate()
                        self._process_ticks()
                elif self.source.is_live:
                    self._decide_rate()
                    self._check_stall(now)

                if max_updates is not None and self.summary.windows >= max_updates:
                    break
                if duration is not None:
                    if self.source.is_live:
                        if now - wall_start >= duration:
                            break
                    elif self._rate and self._parser.total_samples / self._rate >= duration:
                        break
        except KeyboardInterrupt:
            logger.info("Stopping the feedback loop.")
        finally:
            self._finish()
        return self.summary

    def _check_stall(self, now: float) -> None:
        if now - self._last_data_wall < self.settings.stall_seconds:
            return
        if not self._stalled:
            self._stalled = True
            logger.warning("No data from the headset; the display keeps its current parameters.")
            # Never classify a window that spans the gap.
            self._buffer_start += len(self._raw)
            self._raw = np.empty(0, dtype=np.float32)
            self._poor = np.empty(0, dtype=np.int16)
            if self._rate is not None:
                self._next_tick = self._buffer_start + self._window
            self._last_stall_report = 0.0
        if now - self._last_stall_report >= self.settings.update_interval:
            self._last_stall_report = now
            stream_time = self._parser.total_samples / (self._rate or self.config.sample_rate)
            self._dispatch(stream_time, self.controller.hold(STATE_NO_SIGNAL, "no_data"), None)

    def _process_ticks(self) -> None:
        if self._rate is None:
            return
        end_of_buffer = self._buffer_start + len(self._raw)
        while self._next_tick <= end_of_buffer:
            self._evaluate(self._next_tick)
            self._next_tick += self._hop
        # Keep only what the next window needs.
        keep_from = max(self._buffer_start, self._next_tick - self._window - self._context)
        drop = keep_from - self._buffer_start
        if drop > 0:
            self._raw = self._raw[drop:]
            self._poor = self._poor[drop:]
            self._buffer_start = keep_from

    def _evaluate(self, tick: int) -> None:
        assert self._rate is not None
        end = tick - self._buffer_start
        start = end - self._window
        chunk_start = max(0, start - self._context)
        raw_window = self._raw[start:end]
        quality = assess_window(
            raw_window,
            self._poor[start:end],
            self.config,
            reference_sigma=self._tracker.reference_sigma,
        )
        stream_time = tick / self._rate

        if not quality.ok:
            self._dispatch(stream_time, self.controller.hold(STATE_BAD_SIGNAL, quality.reason), quality.as_dict())
            return

        try:
            chunk = resample_signal(self._raw[chunk_start:end], self._rate, self.config.sample_rate)
            bands = band_envelopes(chunk, self.config)[:, -self.config.segment_samples :]
        except ValueError as error:
            self._dispatch(stream_time, self.controller.hold(STATE_BAD_SIGNAL, str(error)), quality.as_dict())
            return

        fresh = min(self.config.segment_samples, int(round(self._hop * self.config.sample_rate / self._rate)))
        if self._tracker.seconds == 0:
            fresh = self.config.segment_samples
        self._tracker.update(bands[:, -fresh:], raw_window)

        if not self._tracker.ready:
            decision = self.controller.hold(STATE_CALIBRATING, f"{self._tracker.seconds:.0f}s of {self._tracker.min_seconds:.0f}s")
        else:
            baseline = self.controller.baseline_for(self._tracker.baseline())
            decision = self.controller.step(normalise_bands(bands, baseline))
        self._dispatch(stream_time, decision, quality.as_dict())

    def _dispatch(self, stream_time: float, decision: Decision, quality: Optional[Dict[str, Any]]) -> None:
        self.summary.windows += 1
        self._states[decision.state] += 1
        confidence = decision.smoothed_probability if decision.smoothed_probability is not None else 0.0

        if self.bridge is not None:
            try:
                self.bridge.send_state_change(decision.is_creatable, confidence, decision.state)
                if decision.parameters is not None:
                    self.bridge.send_parameters(decision.parameters)
            except Exception as error:  # the loop must outlive display problems
                logger.error("Could not reach the display: %s", error)
        if decision.parameters is not None:
            self.summary.updates_sent += 1

        if self.recorder is not None:
            event = {"t": round(stream_time, 3), **decision.as_record()}
            if quality is not None:
                event["quality"] = quality
            if decision.parameters is not None:
                event["parameters"] = {k: round(float(v), 5) for k, v in decision.parameters.items()}
            self.recorder.log_event(event)

        if decision.state != self._previous_state:
            detail = f" ({decision.reason})" if decision.reason else ""
            logger.info("[%7.1fs] state: %s%s", stream_time, decision.state.upper().replace("_", " "), detail)
            self._previous_state = decision.state
        if decision.probability is not None:
            logger.info(
                "[%7.1fs] p(creatable)=%.2f  smoothed=%.2f  %s",
                stream_time,
                decision.probability,
                decision.smoothed_probability or 0.0,
                "→ parameters updated" if decision.parameters is not None else "· holding",
            )
        if self.on_decision is not None:
            self.on_decision(stream_time, decision)

    def _finish(self) -> None:
        rate = self._rate or self.config.sample_rate
        self.summary.stream_seconds = round(self._parser.total_samples / rate, 2)
        self.summary.states = dict(self._states)
        self.summary.checksum_errors = self._parser.stats.checksum_errors
        try:
            self.source.close()
        except Exception:  # pragma: no cover - best effort
            pass
        if self.recorder is not None:
            self.recorder.flush()
            self.recorder.update_metadata(summary=self.summary.as_dict(), parser=self._parser.stats.as_dict())
            self.recorder.close()
        if self.bridge is not None:
            try:
                self.bridge.close()
            except Exception:  # pragma: no cover - best effort
                pass
        logger.info(
            "Session ended: %.0f s of signal, %d windows, %d parameter updates. States: %s",
            self.summary.stream_seconds,
            self.summary.windows,
            self.summary.updates_sent,
            ", ".join(f"{name}={count}" for name, count in sorted(self._states.items())) or "none",
        )


__all__ = [
    "Decision",
    "DocumentWatcher",
    "FeedbackController",
    "LoopSettings",
    "PassiveController",
    "RealtimeSession",
    "SessionRecorder",
    "SessionSummary",
    "STATE_BAD_SIGNAL",
    "STATE_CALIBRATING",
    "STATE_CREATABLE",
    "STATE_IDLE",
    "STATE_NO_SIGNAL",
    "STATE_RECORDING",
]
