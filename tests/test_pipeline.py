import json

import numpy as np
import pytest
import torch
from scipy.io import wavfile

from src import pipeline as pl
from src import thinkgear as tg
from src.eeg_features import load_recording
from src.model import CheckpointError, load_checkpoint, save_checkpoint
from src.eeg_features import DEFAULT_FEATURE_CONFIG
from src.simulator import Segment, simulate_stream, write_hex_recording
from src.training import fit_temperature, grouped_split, roc_auc

# --- Step 1 -------------------------------------------------------------------------------------


def test_step1_segments_from_headset_hex():
    text = tg.bytes_to_hex_text(simulate_stream([Segment(12.0)], seed=1))
    segments = list(pl.iter_brainwave_segments(text))
    assert len(segments) == 2 and all(s.shape == (8, 3200) for s in segments)
    assert all(s.min() >= 0 for s in segments)  # amplitude envelopes
    assert np.array_equal(pl.decode_hex_eeg_segment(text), segments[0])
    assert np.array_equal(pl.decode_hex_eeg_segment(text.encode("ascii")), segments[0])
    with pytest.raises(ValueError, match="too short"):
        pl.decode_hex_eeg_segment(tg.bytes_to_hex_text(simulate_stream([Segment(2.0)], seed=1)))
    with pytest.raises(ValueError):
        pl.decode_hex_eeg_segment("zz")


def test_step1_still_reads_legacy_interleaved_dumps():
    values = (np.arange(8 * 3200 * 2) % 2000 - 1000).astype("<i2")
    segments = list(pl.iter_brainwave_segments(values.tobytes().hex()))
    assert len(segments) == 2 and segments[0].shape == (8, 3200)
    assert segments[0][3, 0] == pytest.approx(values[3] / 32768)


def test_step1_conversion_report(tmp_path):
    path = write_hex_recording(tmp_path / "rec.txt", [Segment(11.0), Segment(10.0, poor_signal=200)], sample_rate=512, seed=2)
    report = pl.convert_recording(path, tmp_path / "out")
    assert report["segments"] == 4 and report["clean_segments"] == 2
    assert report["device_sample_rate"] == 512 and report["resampled"]
    assert 0.4 < report["poor_signal_fraction"] < 0.55
    assert np.load(tmp_path / "out" / "rec_segment_0003.npy").shape == (8, 3200)


# --- Step 2 -------------------------------------------------------------------------------------


@pytest.fixture()
def listening(tmp_path):
    eeg = write_hex_recording(
        tmp_path / "eeg" / "s1.txt",
        [Segment(15.0), Segment(6.0, "enjoy"), Segment(15.0), Segment(6.0, "enjoy"), Segment(18.0)],
        seed=3,
    )
    (tmp_path / "audio").mkdir()
    tone = (np.sin(np.linspace(0, 60 * 2 * np.pi * 220, 60 * 16000)) * 20000).astype(np.int16)
    wavfile.write(tmp_path / "audio" / "piece.wav", 16000, np.stack([tone, tone], axis=1))  # stereo, other rate
    annotations = tmp_path / "annotations.json"
    annotations.write_text(
        json.dumps({"sessions": [{"eeg_file": "eeg/s1.txt", "audio_file": "audio/piece.wav", "enjoyed": [[15, 21], 39.0]}]})
    )
    return annotations


def test_step2_samples_from_session_annotations(listening):
    samples = pl.build_viewer_samples(listening)
    positives = [s for s in samples if s.enjoyed]
    controls = [s for s in samples if not s.enjoyed]
    assert len(positives) == len(controls) > 0
    # Controls keep clear of every enjoyed moment.
    for control in controls:
        for start, end in ((15, 21), (36.5, 41.5)):
            assert control.start_time + 5 <= start - 2.5 or control.start_time >= end + 2.5
    assert all(0 <= s.start_time <= 55 for s in samples)
    assert len({s.group for s in positives}) == 2

    dataset = pl.ViewerEnjoymentDataset(samples)
    eeg, audio, label = dataset[0]
    assert eeg.shape == (8, 3200) and audio.shape == (64, 345) and label.shape == (1,)
    assert torch.isfinite(dataset.eeg).all() and torch.isfinite(dataset.audio).all()


def test_step2_accepts_precut_pairs_and_reports_problems(tmp_path, listening):
    np.save(tmp_path / "clip.npy", np.abs(np.random.default_rng(0).normal(5, 1, (8, 3200))).astype(np.float32))
    np.save(tmp_path / "sound.npy", (np.random.default_rng(0).normal(0, 3000, 44100 * 5)).astype(np.int16))
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps({"samples": [{"eeg_file": "clip.npy", "audio_file": "sound.npy", "enjoyed": True}]}))
    dataset = pl.ViewerEnjoymentDataset(pl.build_viewer_samples(legacy))
    assert len(dataset) == 1 and dataset[0][1].shape == (64, 345)
    with pytest.raises(pl.PipelineError, match="both kinds"):
        pl.train_viewer_model(dataset, epochs=1)

    with pytest.raises(pl.PipelineError, match="not found"):
        pl.build_viewer_samples(tmp_path / "nope.json")
    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    with pytest.raises(pl.PipelineError, match="not valid JSON"):
        pl.build_viewer_samples(broken)
    empty = tmp_path / "empty.json"
    empty.write_text("{}")
    with pytest.raises(pl.PipelineError, match="no usable data"):
        pl.build_viewer_samples(empty)


def test_log_mel_shapes():
    spectrogram = pl.compute_log_mel_spectrogram(np.random.default_rng(0).normal(0, 0.1, 22050 * 5), sample_rate=22050)
    assert spectrogram.shape == (64, 345) and torch.isfinite(spectrogram).all()
    assert pl.compute_log_mel_spectrogram(np.zeros(10), sample_rate=22050).shape[0] == 64


# --- Step 3 -------------------------------------------------------------------------------------


def test_passage_helpers():
    passages = pl.split_passages("One.\n\nTwo two two.\r\n\r\nThree.")
    assert passages == ["One.", "Two two two.", "Three."]
    assert pl.split_passages("a\nb") == ["a", "b"]
    intervals = pl.passage_intervals(["x" * 10, "x" * 30], 100.0, start=20.0)
    assert intervals == [(20.0, 40.0), (40.0, 100.0)]

    def windows(scores):
        return [pl.EnjoymentWindow(float(i), float(i + 5), s) for i, s in enumerate(scores)]

    # Reader B scores everything higher; the ranking must not depend on that.
    readers = {"a": windows([0.1] * 10 + [0.6] * 10), "b": windows([0.7] * 10 + [0.95] * 10)}
    intervals = {name: [(0.0, 12.5), (12.5, 30.0)] for name in readers}
    scores = pl.score_passages(readers, intervals, ["first", "second"])
    assert scores[1].score > 0 > scores[0].score and scores[0].readers == 2
    assert pl.select_passages(scores, 0.3) == ([1], [0])
    assert pl.select_passages(scores[:1]) == ([0], [])


# --- Step 4 -------------------------------------------------------------------------------------


def test_keystroke_log_formats_give_the_same_timeline():
    passages = ["a" * 48, "b" * 98]
    by_length = [{"t": 0, "length": 0}, {"t": 10, "length": 25}, {"t": 20, "length": 50}, {"t": 30, "length": 100}, {"t": 40, "length": 150}]
    by_text = [{"t": e["t"], "text": "x" * e["length"]} for e in by_length]
    by_clock = [{"timestamp": 1000 + e["t"], "length": e["length"]} for e in by_length]
    expected = [
        {"passage": 0, "start_time": 0, "end_time": 20},
        {"passage": 1, "start_time": 20, "end_time": 40},
    ]
    assert pl.timeline_from_keystrokes(by_length, passages) == expected
    assert pl.timeline_from_keystrokes(by_text, passages) == expected
    assert pl.timeline_from_keystrokes(by_clock, passages, started_at=1000) == expected
    keys = [{"t": i * 0.5, "key": "a"} for i in range(1, 147)] + [{"t": 80, "key": "Shift"}]
    assert [e["passage"] for e in pl.timeline_from_keystrokes(keys, passages)] == [0, 1]
    # A long pause before typing resumes is not counted as writing time.
    paused = [{"t": 0, "length": 0}, {"t": 5, "length": 30}, {"t": 300, "length": 60}]
    timeline = pl.timeline_from_keystrokes(paused, passages)
    assert timeline[-1]["start_time"] == 280 and timeline[-1]["end_time"] == 300
    assert pl.timeline_from_keystrokes([], passages) == []


def test_creator_dataset_from_annotations(tmp_path):
    eeg = write_hex_recording(tmp_path / "creator.txt", [Segment(40.0), Segment(40.0, "creatable")], seed=5)
    timeline = [{"passage": 0, "start_time": 5.0, "end_time": 35.0}, {"passage": 1, "start_time": 45.0, "end_time": 75.0},
                {"passage": 2, "start_time": 76.0, "end_time": 77.0}]
    annotations = pl.annotations_from_timeline(eeg, timeline, excellent=[1, 2], control=[0])
    assert [(a.is_creatable, a.start_time) for a in annotations] == [(False, 5.0), (True, 45.0)]
    dataset = pl.CreatorStateDataset(annotations)
    assert len(dataset) == 22 and dataset.eeg.shape == (22, 8, 3200)
    assert dataset.labels.sum() == 11 and len(set(dataset.groups)) == 4
    segment, label = dataset[0]
    assert segment.shape == (8, 3200) and label.shape == (1,)

    manual = tmp_path / "manual.json"
    manual.write_text(json.dumps({"eeg_file": "creator.txt", "excellent_passages": [
        {"start_time": 50, "end_time": 55, "label": "creatable"}, {"start_time": 10, "end_time": 15, "label": "not_creatable"},
        {"start_time": 500, "end_time": 505, "label": "creatable"}]}))
    loaded = pl.annotations_from_manual_file(manual)
    assert [a.is_creatable for a in loaded] == [True, False, True]
    assert len(pl.CreatorStateDataset(loaded)) == 2  # the stretch beyond the recording is dropped

    with pytest.raises(pl.PipelineError, match="no usable"):
        pl.CreatorStateDataset([pl.CreatorAnnotation(eeg, 900.0, 905.0, True)])
    ready = pl.CreatorStateDataset.from_segments([(torch.zeros(8, 3200), 1.0), (torch.ones(8, 3200), 0.0)])
    assert len(ready) == 2 and ready.labels.tolist() == [[1.0], [0.0]]


# --- training utilities and checkpoints ------------------------------------------------------------


def test_grouped_split_keeps_groups_together_and_stratifies():
    groups = [f"g{i // 4}" for i in range(40)]
    labels = [int(i >= 20) for i in range(40)]
    train, validation = grouped_split(groups, labels, 0.2, seed=0)
    assert not {groups[i] for i in train} & {groups[i] for i in validation}
    assert {labels[i] for i in validation} == {0, 1} and len(train) + len(validation) == 40
    assert grouped_split(groups, labels, 0.2, seed=0) == (train, validation)
    assert grouped_split(["a", "b"], [0, 1], 0.2, seed=0)[1] == []


def test_metrics_and_calibration():
    assert roc_auc(np.array([0.1, 0.4, 0.35, 0.8]), np.array([0, 0, 1, 1])) == 0.75
    assert np.isnan(roc_auc(np.array([0.1, 0.2]), np.array([1, 1])))
    labels = torch.tensor([1.0, 0.0] * 50)
    overconfident = torch.where(torch.rand(100, generator=torch.Generator().manual_seed(0)) < 0.7, labels * 2 - 1, 1 - labels * 2) * 8
    assert fit_temperature(overconfident, labels) > 3
    assert fit_temperature(labels * 2 - 1, labels) == 1.0  # never sharpened
    assert fit_temperature(torch.tensor([5.0]), torch.tensor([1.0])) == 1.0


def test_checkpoints_are_self_describing(tmp_path):
    from src.model import CreatorStateModel, create_model

    model = CreatorStateModel(hidden_channels=16, input_stride=4)
    model.set_temperature(0.2)
    assert float(model.temperature) == 1.0
    model.set_temperature(2.5)
    path = save_checkpoint(tmp_path / "m" / "creator.pt", kind="creator", model=model, feature_config=DEFAULT_FEATURE_CONFIG,
                           extra={"metrics": {"auc": np.float32(0.9), "path": tmp_path}})
    loaded, payload = load_checkpoint(path, kind="creator")
    assert loaded.hidden_channels == 16 and loaded.input_stride == 4 and float(loaded.temperature) == 2.5
    assert payload["feature_config"]["sample_rate"] == 640 and not loaded.training
    window = torch.randn(2, 8, 3200)
    assert torch.allclose(model.eval().predict_proba(window), loaded.predict_proba(window))

    parameter = create_model((8, 3200), hidden_channels=16)
    save_checkpoint(tmp_path / "p.pt", kind="parameter", model=parameter, feature_config=DEFAULT_FEATURE_CONFIG)
    restored, payload = load_checkpoint(tmp_path / "p.pt")
    assert len(payload["parameter_names"]) == 50 and len(restored.predict_parameters(window)[0]) == 50

    with pytest.raises(CheckpointError, match="'viewer' model is required"):
        load_checkpoint(path, kind="viewer")
    with pytest.raises(CheckpointError, match="not found"):
        load_checkpoint(tmp_path / "missing.pt")
    torch.save({"model_state": {}}, tmp_path / "old.pt")
    with pytest.raises(CheckpointError, match="Re-run"):
        load_checkpoint(tmp_path / "old.pt")
    (tmp_path / "junk.pt").write_bytes(b"not a model")
    with pytest.raises(CheckpointError):
        load_checkpoint(tmp_path / "junk.pt")


def test_models_accept_any_batch_size_and_missing_audio():
    from src.model import ViewerEnjoymentModel

    model = ViewerEnjoymentModel(hidden_channels=16)
    eeg, audio = torch.randn(1, 8, 3200), torch.randn(1, 64, 345)
    model.train()
    assert model(eeg, audio).shape == (1, 1)  # a batch of one must work during training
    model.eval()
    assert model(eeg, None).shape == (1, 1)
    assert 0.0 <= float(model.predict_proba(eeg)) <= 1.0
