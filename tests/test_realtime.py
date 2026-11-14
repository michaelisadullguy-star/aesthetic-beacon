import json
import os
import sys
import threading
import time

import numpy as np
import pytest
import torch
from torch import nn

from src import thinkgear as tg
from src.eeg_sources import EEGSource, FollowSource, GeneratorSource, ReplaySource, SerialSource, SourceError, open_source
from src.parameters import PARAMETER_SPECS, ParameterCodec, default_parameters
from src.pipeline import run_feedback_loop, timeline_from_keystrokes
from src.realtime import DocumentWatcher, FeedbackController, LoopSettings, RealtimeSession, SessionRecorder
from src.simulator import Segment, simulate_stream


class ScriptedCreator(nn.Module):
    """Returns a fixed sequence of probabilities."""

    def __init__(self, probabilities):
        super().__init__()
        self.probabilities = list(probabilities)
        self.calls = 0

    def predict_proba(self, window):
        value = self.probabilities[min(self.calls, len(self.probabilities) - 1)]
        self.calls += 1
        return torch.tensor([value])


class FixedParameters(nn.Module):
    codec = ParameterCodec()

    def __init__(self, cell_size=40):
        super().__init__()
        self.cell_size = cell_size

    def predict_parameters(self, window):
        return [{**default_parameters(), "cellSize": float(self.cell_size), "backgroundHue": 350.0}]


WINDOW = np.zeros((8, 3200), dtype=np.float32)


def _controller(probabilities, **settings):
    return FeedbackController(ScriptedCreator(probabilities), FixedParameters(), LoopSettings(**settings), device="cpu")


def test_updates_only_while_creatable_and_holds_otherwise():
    controller = _controller([0.9, 0.9, 0.1, 0.1, 0.95], probability_smoothing=1.0, parameter_smoothing=1.0)
    decisions = [controller.step(WINDOW) for _ in range(5)]
    assert [d.state for d in decisions] == ["creatable", "creatable", "idle", "idle", "creatable"]
    assert [d.parameters is not None for d in decisions] == [True, True, False, False, True]
    # While idle the display keeps the last parameter set.
    assert decisions[2].held_parameters == decisions[1].parameters
    assert decisions[0].parameters["cellSize"] == 40


def test_hysteresis_prevents_flicker():
    hovering = [0.62, 0.57, 0.63, 0.55, 0.61, 0.52]
    plain = _controller(hovering, probability_smoothing=1.0, release_threshold=0.6)
    steady = _controller(hovering, probability_smoothing=1.0)  # leaves only below 0.5
    assert sum(1 for _ in range(6) if not plain.step(WINDOW).is_creatable) == 3
    assert all(steady.step(WINDOW).is_creatable for _ in range(6))


def test_smoothing_ignores_a_single_spike():
    controller = _controller([0.1, 0.1, 0.95, 0.1, 0.1], probability_smoothing=0.5)
    assert not any(controller.step(WINDOW).is_creatable for _ in range(5))


def test_parameters_move_gradually_and_stay_valid():
    controller = _controller([0.9] * 3, parameter_smoothing=0.5, probability_smoothing=1.0)
    controller.last_parameters = {**default_parameters(), "cellSize": 20.0, "backgroundHue": 10.0}
    first = controller.step(WINDOW).parameters
    assert first["cellSize"] == 30 and min(first["backgroundHue"], 360 - first["backgroundHue"]) < 1
    for spec in PARAMETER_SPECS:
        assert spec.low <= first[spec.name] <= spec.high


def test_bad_input_never_reaches_the_models():
    controller = _controller([0.9])
    broken = WINDOW.copy()
    broken[0, 0] = np.nan
    decision = controller.step(broken)
    assert decision.state == "bad_signal" and decision.parameters is None
    assert controller.creator_model.calls == 0
    assert _controller([float("nan")]).step(WINDOW).state == "bad_signal"


def test_run_feedback_loop_callback_styles():
    windows = [np.abs(np.random.default_rng(i).normal(5, 1, (8, 3200))).astype(np.float32) for i in range(4)]
    two_argument, one_argument = [], []
    common = dict(parameter_model=FixedParameters(), probability_smoothing=1.0, device="cpu")
    records = run_feedback_loop(
        windows, creator_model=ScriptedCreator([0.9, 0.1, 0.1, 0.9]), update_callback=lambda p, c: two_argument.append(c), **common
    )
    run_feedback_loop(windows, creator_model=ScriptedCreator([0.9, 0.1, 0.1, 0.9]), update_callback=one_argument.append, **common)
    assert two_argument == [True, False, False, True]
    assert len(one_argument) == 2  # only fresh parameter sets
    assert [r["state"] for r in records] == ["creatable", "idle", "idle", "creatable"]

    calls = iter(windows[:2] + [None])
    started = time.monotonic()
    supplied = run_feedback_loop(
        creator_model=ScriptedCreator([0.9]), update_callback=lambda p, c: None, eeg_supplier=lambda: next(calls),
        update_interval=0.05, **common,
    )
    assert len(supplied) == 2 and time.monotonic() - started < 2


# --- the full session ---------------------------------------------------------------------------


def _session(source, probabilities=(0.9,), bridge=None, recorder=None, **settings):
    settings.setdefault("calibration_seconds", 10.0)
    controller = _controller(list(probabilities), **settings)
    events = []
    session = RealtimeSession(
        source, controller, bridge=bridge, recorder=recorder, on_decision=lambda t, d: events.append((t, d))
    )
    return session, events


def _source(stream, live=False, chunk=4096):
    source = GeneratorSource(lambda: (stream[i : i + chunk] for i in range(0, len(stream), chunk)))
    source.is_live = live
    return source


def test_session_calibrates_then_updates_every_interval():
    stream = simulate_stream([Segment(40.0)], seed=1)
    session, events = _session(_source(stream))
    summary = session.run()
    assert summary.device_sample_rate == 640 and summary.stream_seconds == 40.0
    assert [t for t, _ in events] == [5.0 * k for k in range(1, 9)]
    states = [d.state for _, d in events]
    assert states[:1] == ["calibrating"] and states[-1] == "creatable"
    assert "creatable" not in states[: states.index("creatable")]
    assert summary.updates_sent == states.count("creatable") >= 5


def test_session_measures_a_512_hz_headset():
    stream = simulate_stream([Segment(30.0)], seed=2, sample_rate=512)
    session, events = _session(_source(stream))
    summary = session.run()
    assert summary.device_sample_rate == 512
    assert summary.stream_seconds == 30.0
    assert [t for t, _ in events][:3] == [5.0, 10.0, 15.0]


def test_session_holds_during_poor_contact():
    stream = simulate_stream(
        [Segment(25.0), Segment(15.0, poor_signal=200), Segment(20.0)], seed=3
    )
    session, events = _session(_source(stream), probability_smoothing=1.0)
    session.run()
    by_time = {t: d for t, d in events}
    assert by_time[20.0].state == "creatable"
    for moment in (35.0, 40.0):
        assert by_time[moment].state == "bad_signal" and by_time[moment].parameters is None
        assert by_time[moment].held_parameters is not None
    assert by_time[55.0].state == "creatable"


def test_session_survives_corrupted_bytes_and_garbage():
    stream = bytearray(simulate_stream([Segment(30.0)], seed=4))
    rng = np.random.default_rng(0)
    for position in rng.integers(0, len(stream), 200):
        stream[position] = rng.integers(0, 256)
    session, events = _session(_source(bytes(rng.integers(0, 256, 5000, dtype=np.uint8)) + bytes(stream)))
    summary = session.run()
    assert summary.windows >= 5 and events[-1][1].state == "creatable"


class DroppingSource(EEGSource):
    """A live device that goes silent for a while, then comes back."""

    is_live = True
    description = "flaky device"

    def __init__(self, first, second, silence):
        self.parts, self.silence = (first, second), silence

    def chunks(self):
        for index, part in enumerate(self.parts):
            for i in range(0, len(part), 8192):
                yield part[i : i + 8192]
            if index == 0:
                until = time.monotonic() + self.silence
                while time.monotonic() < until:
                    time.sleep(0.02)
                    yield b""


def test_session_recovers_from_a_disconnect():
    first = simulate_stream([Segment(22.0)], seed=5)
    second = simulate_stream([Segment(22.0)], seed=6)
    session, events = _session(DroppingSource(first, second, 0.6), stall_seconds=0.2, update_interval=5.0, probability_smoothing=1.0)
    summary = session.run()
    states = [d.state for _, d in events]
    assert "no_signal" in states and summary.reconnects == 1
    gap = states.index("no_signal")
    assert states[gap - 1] == "creatable"
    assert events[gap][1].held_parameters is not None and events[gap][1].parameters is None
    # No window may mix signal from before and after the gap: the next decision
    # needs a full fresh window (5 s after the 22 s already received).
    after = [t for t, d in events[gap + 1 :] if d.state != "no_signal"]
    assert after[0] == pytest.approx(27.0, abs=0.01)
    assert states[-1] == "creatable"


def test_recorder_writes_a_reusable_session(tmp_path):
    from src.eeg_features import load_recording

    stream = simulate_stream([Segment(30.0)], seed=7)
    document = tmp_path / "draft.txt"
    document.write_text("First line.")
    recorder = SessionRecorder(tmp_path / "session", watch_document=document)
    session, _ = _session(_source(stream, chunk=777), recorder=recorder)
    session.run()

    recording = load_recording(tmp_path / "session" / "eeg.txt")
    assert recording.info["format"] == "thinkgear" and recording.duration == pytest.approx(30.0)
    assert (tmp_path / "session" / "eeg.txt").read_text().splitlines()[0].count(" ") == 31
    events = [json.loads(line) for line in (tmp_path / "session" / "events.jsonl").read_text().splitlines()]
    assert len(events) == 6 and events[-1]["updated"] and len(events[-1]["parameters"]) == 50
    metadata = json.loads((tmp_path / "session" / "session.json").read_text())
    assert metadata["device_sample_rate"] == 640 and metadata["summary"]["windows"] == 6
    assert metadata["keystroke_log"] == "keystrokes.jsonl"


def test_document_watcher_feeds_the_writing_timeline(tmp_path):
    passages = ["a" * 40, "b" * 60, "c" * 30]
    document = tmp_path / "draft.txt"
    watcher = DocumentWatcher(document, tmp_path / "keys.jsonl", started_at=time.time())
    text = ""
    events = []
    for index, passage in enumerate(passages):
        for half in (passage[: len(passage) // 2], passage[len(passage) // 2 :]):
            text += half
            document.write_text(text)
            assert watcher.poll()
        text += "\n\n"
    assert not watcher.poll()  # unchanged file: nothing logged
    for step, line in enumerate((tmp_path / "keys.jsonl").read_text().splitlines()):
        event = json.loads(line)
        event["t"] = 10.0 * (step + 1)  # pretend ten seconds passed between saves
        events.append(event)
    timeline = timeline_from_keystrokes(events, passages)
    assert [entry["passage"] for entry in timeline] == [0, 1, 2]
    assert timeline[1]["start_time"] == 20.0 and timeline[1]["end_time"] == 40.0


# --- sources ----------------------------------------------------------------------------------


def _collect(source, wanted_samples, timeout=10.0):
    parser = tg.ThinkGearParser()
    samples = []
    deadline = time.monotonic() + timeout
    for chunk in source.chunks():
        samples.append(parser.feed(chunk).raw)
        if parser.total_samples >= wanted_samples or time.monotonic() > deadline:
            break
    source.close()
    return np.concatenate(samples)


def test_replay_source(tmp_path):
    values = np.arange(-900, 900)
    path = tmp_path / "recording.txt"
    path.write_text(tg.bytes_to_hex_text(tg.encode_stream(values, 600)))
    assert np.array_equal(_collect(ReplaySource(path, speed=0), 1800), values.astype(np.int16))
    started = time.monotonic()
    _collect(ReplaySource(path, speed=10), 1800)
    assert 0.2 < time.monotonic() - started < 2.0  # three seconds of signal at ten times speed
    assert isinstance(open_source(str(path), speed=0), ReplaySource)
    with pytest.raises(SourceError):
        ReplaySource(tmp_path / "missing.txt")
    with pytest.raises(SourceError):
        open_source("bluetooth:foo")


def test_follow_source_reads_a_growing_log(tmp_path):
    values = np.arange(-500, 500)
    text = tg.bytes_to_hex_text(tg.encode_stream(values, 250))
    path = tmp_path / "live.txt"
    path.write_text("")

    def writer():
        with open(path, "a") as handle:
            for i in range(0, len(text), 1013):
                handle.write(text[i : i + 1013])
                handle.flush()
                time.sleep(0.005)

    thread = threading.Thread(target=writer)
    source = FollowSource(path)
    thread.start()
    received = _collect(source, 1000)
    thread.join()
    assert np.array_equal(received, values.astype(np.int16))


@pytest.mark.skipif(sys.platform == "win32", reason="uses a pseudo-terminal")
def test_serial_source_reads_and_reconnects():
    pytest.importorskip("serial")
    import tty

    values = np.arange(-400, 400)
    stream = tg.encode_stream(values, 200)
    master, slave = os.openpty()
    tty.setraw(slave)
    port = os.ttyname(slave)

    def writer():
        time.sleep(0.5)  # opening a serial port discards what was sent before
        for i in range(0, len(stream), 512):
            os.write(master, stream[i : i + 512])
            time.sleep(0.002)

    thread = threading.Thread(target=writer)
    thread.start()
    received = _collect(open_source(f"serial:{port}@57600"), 800)
    thread.join()
    os.close(master)
    os.close(slave)
    assert np.array_equal(received, values.astype(np.int16))

    # A port that does not exist yields empty chunks instead of raising.
    missing = SerialSource("/dev/does-not-exist")
    iterator = missing.chunks()
    assert [next(iterator) for _ in range(3)] == [b"", b"", b""]
    missing.close()
