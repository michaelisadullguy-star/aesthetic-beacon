"""The whole loop on a simulated study: decode → train → rank → label → train → live."""
import json
import subprocess
import sys

import numpy as np
import pytest

from src.eeg_features import load_recording
from src.model import load_checkpoint
from src.parameters import PARAMETER_SPECS
from src.workflow_orchestrator import WorkflowConfig, WorkflowOrchestrator, main


def test_demo_runs_and_recovers_the_ground_truth(demo):
    assert demo["exit_code"] == 0  # the demo checks passages and live detection itself
    data = demo["data"]
    truth = json.loads((data / "demo_truth.json").read_text())

    scores = json.loads((data / "step3_article_reading" / "results" / "passage_scores.json").read_text())
    assert set(scores["excellent"]) <= set(truth["excellent_passages"])
    assert not set(scores["control"]) & set(truth["excellent_passages"])

    derived = json.loads((data / "step4_creator_training" / "annotations" / "derived" / "creator_session_001.json").read_text())
    assert {p["label"] for p in derived["excellent_passages"]} == {"creatable", "not_creatable"}

    for name, kind in (("viewer_model", "viewer"), ("creator_model", "creator"), ("parameter_model", "parameter")):
        model, payload = load_checkpoint(data / "step6_realtime" / "models" / f"{name}.pt", kind=kind)
        history = json.loads((data / "step6_realtime" / "models" / f"{name}.history.json").read_text())
        assert history["losses"] and history["metrics"]["evaluated_on"] == "validation"
    assert json.loads((data / "step6_realtime" / "models" / "creator_model.history.json").read_text())["metrics"]["auc"] > 0.8


def test_live_session_output_is_valid_and_reusable(demo):
    data = demo["data"]
    sessions = sorted((data / "step6_realtime" / "sessions").iterdir())
    assert len(sessions) == 1
    events = [json.loads(line) for line in (sessions[0] / "events.jsonl").read_text().splitlines()]
    states = [event["state"] for event in events]
    assert states[0] == "calibrating" and {"creatable", "idle"} <= set(states)
    # Parameters are sent exactly when the state is creatable, and always in range.
    for event in events:
        assert ("parameters" in event) == (event["state"] == "creatable")
        if "parameters" in event:
            for spec in PARAMETER_SPECS:
                assert spec.low <= event["parameters"][spec.name] <= spec.high
    updates = [event["parameters"] for event in events if "parameters" in event]
    assert len({json.dumps(u, sort_keys=True) for u in updates}) > 1  # they follow the EEG, not a constant

    current = json.loads((data / "step6_realtime" / "outputs" / "current_parameters.json").read_text())
    assert current["sequence"] == len(updates) and len(current["palette"]) == 4

    # The recorded EEG is byte-for-byte what the headset sent, so it can be trained on.
    original = load_recording(data / "step6_realtime" / "demo_live_session.txt")
    recorded = load_recording(sessions[0] / "eeg.txt")
    assert np.array_equal(original.raw, recorded.raw)


def test_rerun_skips_finished_steps_and_retrains_on_new_data(demo, caplog):
    config = WorkflowConfig(demo["config"], {"data_root": str(demo["data"])})
    orchestrator = WorkflowOrchestrator(config)
    model = demo["data"] / "step6_realtime" / "models" / "creator_model.pt"
    before = model.stat().st_mtime_ns
    with caplog.at_level("INFO"):
        orchestrator.run_training()
    assert sum("up to date" in record.message for record in caplog.records) == 4
    assert model.stat().st_mtime_ns == before

    # A hand-made annotation is new input for Step 4 only.
    extra = demo["data"] / "step4_creator_training" / "annotations" / "manual.json"
    extra.write_text(json.dumps({
        "eeg_file": "step4_creator_training/eeg/creator_session_001.txt",
        "excellent_passages": [{"start_time": 1.0, "end_time": 9.0, "label": "not_creatable"}],
    }))
    caplog.clear()
    with caplog.at_level("INFO"):
        orchestrator.run_training()
    assert sum("up to date" in record.message for record in caplog.records) == 3
    assert model.stat().st_mtime_ns != before
    extra.unlink()


def test_individual_steps_and_documented_commands(demo, tmp_path, repo_root):
    # Without the demo's data_root override the configured paths point at an empty tree.
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"data_root": str(tmp_path / "nothing"), "device": "cpu"}))
    for step in ("1", "2", "3", "4", "5", "6"):
        assert main(["--config", str(empty), "--step", step, "--source", "sim"]) == 1  # clear error, no traceback
    assert main(["--config", str(empty), "--check"]) == 1

    # The commands from the README work when run as plain scripts from another directory.
    script = repo_root / "src" / "workflow_orchestrator.py"
    result = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True, cwd=tmp_path)
    assert result.returncode == 0 and "--demo" in result.stdout
    data_config = tmp_path / "demo_config.json"
    settings = json.loads(demo["config"].read_text())
    settings["data_root"] = str(demo["data"])
    data_config.write_text(json.dumps(settings))
    result = subprocess.run(
        [sys.executable, str(script), "--config", str(data_config), "--step", "1", "--train"],
        capture_output=True, text=True, cwd=tmp_path,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Converted 2 recording(s)" in result.stdout
    assert (demo["data"] / "step1_hex_eeg" / "converted" / "session_001_segment_0003.npy").is_file()

    result = subprocess.run(
        [sys.executable, str(script), "--config", str(data_config), "--step", "6", "--realtime", "--source", "sim:3",
         "--speed", "0", "--duration", "60", "--no-osc", "--no-record"],
        capture_output=True, text=True, cwd=tmp_path,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Session ended: 60 s of signal, 12 windows" in result.stdout


def test_configuration_merging(tmp_path):
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"data_root": "mydata", "step5": {"epochs": 7}, "typo_section": 1, "step6": {"sorce": "x"}}))
    config = WorkflowConfig(path)
    assert config.get("step5.epochs") == 7 and config.get("step5.batch_size") == 32  # other defaults survive
    assert config.path("step5.dataset_root") == tmp_path / "mydata" / "step5_image_training"
    assert set(config.unknown_keys) == {"typo_section", "step6.sorce"}
    assert config.feature_config().sample_rate == 640
    with pytest.raises(Exception, match="not found"):
        WorkflowConfig(tmp_path / "missing.json")


def test_first_session_can_be_recorded_without_models(tmp_path):
    """Bootstrapping: before any model exists, a writing session is recorded only."""

    config = tmp_path / "c.json"
    config.write_text(json.dumps({"data_root": str(tmp_path / "data"), "device": "cpu"}))
    draft = tmp_path / "draft.txt"
    draft.write_text("An opening line.")
    assert main(["--config", str(config), "--step", "6", "--source", "sim:1", "--speed", "0", "--duration", "40"]) == 1
    code = main([
        "--config", str(config), "--step", "6", "--record-only", "--source", "sim:1", "--speed", "0",
        "--duration", "40", "--watch-document", str(draft),
    ])
    assert code == 0
    (session,) = (tmp_path / "data" / "step6_realtime" / "sessions").iterdir()
    events = [json.loads(line) for line in (session / "events.jsonl").read_text().splitlines()]
    assert {e["state"] for e in events} == {"calibrating", "recording"} and not any(e["updated"] for e in events)
    assert load_recording(session / "eeg.txt").duration >= 40
    assert (session / "article.txt").read_text() == "An opening line."
    assert json.loads((session / "keystrokes.jsonl").read_text().splitlines()[0])["length"] == 16
    assert not (tmp_path / "data" / "step6_realtime" / "outputs" / "current_parameters.json").exists()


def test_recorded_session_is_labelled_automatically(demo):
    """Closing the loop: a recorded writing session becomes creator training data
    as soon as readers have ranked the article it produced."""

    import shutil

    data = demo["data"]
    article = (data / "step3_article_reading" / "article.txt").read_text()
    session = data / "step6_realtime" / "sessions" / "20250101-120000"
    session.mkdir(parents=True)
    try:
        shutil.copyfile(data / "step4_creator_training" / "eeg" / "creator_session_001.txt", session / "eeg.txt")
        (session / "article.txt").write_text(article)
        source_log = (data / "step4_creator_training" / "eeg" / "creator_session_001_keystrokes.jsonl").read_text().splitlines()
        snapshots = [{"t": json.loads(l)["t"], "text": article[: json.loads(l)["length"]]} for l in source_log]
        (session / "keystrokes.jsonl").write_text("\n".join(json.dumps(s) for s in snapshots) + "\n")
        (session / "session.json").write_text(json.dumps({"eeg_file": "eeg.txt", "keystroke_log": "keystrokes.jsonl", "started_at": 0}))

        orchestrator = WorkflowOrchestrator(WorkflowConfig(demo["config"], {"data_root": str(data)}))
        annotations = orchestrator._creator_annotations()
        derived = data / "step4_creator_training" / "annotations" / "derived" / "20250101-120000.json"
        labels = json.loads(derived.read_text())["excellent_passages"]
        assert {p["label"] for p in labels} == {"creatable", "not_creatable"}
        assert any(str(a.brainwave_path).endswith("20250101-120000/eeg.txt") for a in annotations)

        # A session that wrote a different article is not labelled from this ranking.
        other = data / "step6_realtime" / "sessions" / "20250102-120000"
        shutil.copytree(session, other)
        (other / "article.txt").write_text("Something else entirely.\n\nNot what the readers read.")
        orchestrator._creator_annotations()
        assert not (derived.parent / "20250102-120000.json").exists()
    finally:
        shutil.rmtree(session, ignore_errors=True)
        shutil.rmtree(data / "step6_realtime" / "sessions" / "20250102-120000", ignore_errors=True)
        for leftover in (data / "step4_creator_training" / "annotations" / "derived").glob("2025010*.json"):
            leftover.unlink()


def _replay(demo, tmp_path, segments, *, level=1.0, seed=77):
    """Run the trained live loop on a freshly simulated session of the demo's creator."""

    from src.simulator import Person, write_hex_recording

    creator = json.loads((demo["data"] / "demo_truth.json").read_text())["creator"]
    person = Person(tuple(creator["band_factors"]), creator["level"] * level, creator["blink_interval"])
    path = write_hex_recording(tmp_path / "scenario.txt", segments, seed=seed, person=person)
    orchestrator = WorkflowOrchestrator(WorkflowConfig(demo["config"], {"data_root": str(demo["data"])}))
    decisions = []
    orchestrator.run_step6_realtime(
        f"replay:{path}", speed=0, use_osc=False, record=False,
        on_decision=lambda when, decision: decisions.append((when, decision.state)),
    )
    return [(when, state) for when, state in decisions if state != "calibrating"]


def _share_creatable(decisions, start, end):
    inside = [state == "creatable" for when, state in decisions if start + 10 <= when <= end]
    return sum(inside) / max(len(inside), 1)


def test_a_session_of_pure_rest_stays_dark(demo, tmp_path):
    """The dominant state of a session must not be mistaken for 'neutral'."""

    from src.simulator import Segment

    decisions = _replay(demo, tmp_path, [Segment(300.0, "rest")])
    assert _share_creatable(decisions, 0, 300) < 0.15


def test_a_long_creatable_stretch_stays_lit(demo, tmp_path):
    """The adaptive baseline must not absorb a sustained state."""

    from src.simulator import Segment

    decisions = _replay(demo, tmp_path, [Segment(40.0, "rest"), Segment(360.0, "creatable"), Segment(60.0, "rest")])
    assert _share_creatable(decisions, 40, 400) > 0.85
    assert _share_creatable(decisions, 220, 400) > 0.85  # still on after five minutes
    assert _share_creatable(decisions, 400, 460) < 0.25


def test_starting_in_the_creatable_state_and_a_change_of_contact(demo, tmp_path):
    from src.simulator import Segment

    decisions = _replay(demo, tmp_path, [Segment(150.0, "creatable"), Segment(150.0, "rest")])
    assert _share_creatable(decisions, 0, 150) > 0.8 and _share_creatable(decisions, 150, 300) < 0.2
    # Twice the signal level (a better electrode contact) changes nothing.
    decisions = _replay(demo, tmp_path, [Segment(60.0, "rest"), Segment(120.0, "creatable"), Segment(100.0, "rest")], level=2.0)
    assert _share_creatable(decisions, 0, 60) < 0.2
    assert _share_creatable(decisions, 60, 180) > 0.8
    assert _share_creatable(decisions, 180, 280) < 0.2
