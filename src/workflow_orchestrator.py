"""Runs the complete Aesthetic Beacon workflow.

Usage (from the repository root)::

    # Try everything end to end with simulated data — no headset needed
    python src/workflow_orchestrator.py --demo

    # Train everything from the recordings under data/, then start the live loop
    python src/workflow_orchestrator.py --all --source serial:/dev/ttyUSB0

    # Individual steps
    python src/workflow_orchestrator.py --step 1
    python src/workflow_orchestrator.py --step 2 --train
    python src/workflow_orchestrator.py --step 3
    python src/workflow_orchestrator.py --step 4 --train
    python src/workflow_orchestrator.py --step 5 --train
    python src/workflow_orchestrator.py --step 6 --realtime --source serial:COM3

``--all`` only redoes the steps whose inputs changed since the last run, so
after adding new recordings it can simply be run again.
"""
from __future__ import annotations

if __package__ in (None, ""):  # executed as a script: make the package importable
    import sys as _sys
    from pathlib import Path as _Path

    _root = str(_Path(__file__).resolve().parent.parent)
    _sys.path[:] = [entry for entry in _sys.path if _Path(entry or ".").resolve() != _Path(__file__).resolve().parent]
    _sys.path.insert(0, _root)
    __package__ = "src"
    __import__("src")

import argparse
import copy
import hashlib
import json
import logging
import shutil
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .eeg_features import (
    DEFAULT_FEATURE_CONFIG,
    FeatureConfig,
    RecordingError,
    find_recordings,
    load_recording,
)
from .eeg_sources import SourceError, open_source
from .model import CheckpointError, checkpoint_feature_config, load_checkpoint, save_checkpoint
from .osc_bridge import OSCConfig, create_bridge
from .pipeline import (
    CreatorAnnotation,
    CreatorStateDataset,
    PipelineError,
    ViewerEnjoymentDataset,
    annotations_from_manual_file,
    annotations_from_timeline,
    build_viewer_samples,
    convert_recording,
    evaluate_enjoyment_windows,
    passage_intervals,
    read_json,
    read_keystroke_log,
    resolve_path,
    score_passages,
    select_passages,
    split_passages,
    timeline_from_keystrokes,
    train_creator_model,
    train_processing_parameter_model,
    train_viewer_model,
)
from .realtime import FeedbackController, LoopSettings, PassiveController, RealtimeSession, SessionRecorder
from .training import TrainingError, TrainingHistory, resolve_device

logger = logging.getLogger("aesthetic_beacon")

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
SKETCH_DIRECTORY = REPOSITORY_ROOT / "scripts" / "processing_templates" / "dynamic_texture_osc"


def _defaults() -> Dict[str, Any]:
    """Default configuration. ``{data}`` is replaced by ``data_root``."""

    return {
        "data_root": "data",
        "model_dir": "{data}/step6_realtime/models",
        "output_dir": "{data}/step6_realtime/outputs",
        "device": "auto",
        "seed": 0,
        "features": {
            "sample_rate": 640,
            "device_sample_rate": "auto",
            "notch_hz": 50.0,
        },
        "model": {"hidden_channels": 128, "input_stride": 8},
        "step1": {
            "hex_dir": "{data}/step1_hex_eeg",
            "output_dir": "{data}/step1_hex_eeg/converted",
        },
        "step2": {
            "annotations": "{data}/step2_viewer_training/annotations/annotations.json",
            "model_path": "{data}/step6_realtime/models/viewer_model.pt",
            "epochs": 30,
            "batch_size": 16,
            "learning_rate": 1e-3,
            "validation_split": 0.2,
        },
        "step3": {
            "eeg_dir": "{data}/step3_article_reading/eeg",
            "article": "{data}/step3_article_reading/article.txt",
            "sessions": "{data}/step3_article_reading/sessions.json",
            "results_dir": "{data}/step3_article_reading/results",
            "viewer_model": "{data}/step6_realtime/models/viewer_model.pt",
            "window_size": 5.0,
            "stride": 1.0,
            "excellent_fraction": 0.3,
        },
        "step4": {
            "eeg_dir": "{data}/step4_creator_training/eeg",
            "annotations_dir": "{data}/step4_creator_training/annotations",
            "model_path": "{data}/step6_realtime/models/creator_model.pt",
            "epochs": 40,
            "batch_size": 8,
            "learning_rate": 5e-4,
            "validation_split": 0.2,
        },
        "step5": {
            "dataset_root": "{data}/step5_image_training",
            "model_path": "{data}/step6_realtime/models/parameter_model.pt",
            "processing_weights": None,
            "epochs": 120,
            "batch_size": 32,
            "learning_rate": 5e-4,
            "validation_split": 0.2,
            "num_blocks": 4,
            "dropout": 0.1,
        },
        "step6": {
            "creator_model": "{data}/step6_realtime/models/creator_model.pt",
            "parameter_model": "{data}/step6_realtime/models/parameter_model.pt",
            "sessions_dir": "{data}/step6_realtime/sessions",
            "source": None,
            "creatable_threshold": 0.6,
            "release_threshold": None,
            "update_interval": 5.0,
            "calibration_seconds": 20.0,
            "probability_smoothing": 0.6,
            "parameter_smoothing": 0.6,
            "record": True,
            "watch_document": None,
            "use_osc": True,
            "osc_host": "127.0.0.1",
            "osc_port": 12000,
        },
    }


_PATH_KEYS = {
    "data_root",
    "model_dir",
    "output_dir",
    "hex_dir",
    "annotations",
    "annotations_dir",
    "model_path",
    "eeg_dir",
    "article",
    "sessions",
    "results_dir",
    "viewer_model",
    "dataset_root",
    "processing_weights",
    "creator_model",
    "parameter_model",
    "sessions_dir",
    "watch_document",
    "audio_dir",
}


def _merge(base: Dict[str, Any], override: Dict[str, Any], prefix: str = "") -> List[str]:
    """Merge ``override`` into ``base`` recursively; returns keys ``base`` does not know."""

    unknown: List[str] = []
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            unknown.extend(_merge(base[key], value, f"{prefix}{key}."))
        else:
            if key not in base and key != "description":
                unknown.append(f"{prefix}{key}")
            base[key] = value
    return unknown


class WorkflowConfig:
    """Configuration for the complete workflow.

    Values from the user's JSON file are merged over the defaults, section by
    section, so a file only needs the settings it changes.  Relative paths are
    resolved against the directory of the configuration file, or against the
    repository when no file is given.
    """

    def __init__(self, config_path: Optional[Path | str] = None, overrides: Optional[Dict[str, Any]] = None):
        self.config = _defaults()
        self.base_dir = REPOSITORY_ROOT
        self.unknown_keys: List[str] = []

        user: Dict[str, Any] = {}
        if config_path is not None:
            config_path = Path(config_path)
            if not config_path.is_file():
                raise PipelineError(f"Configuration file not found: {config_path}")
            loaded = read_json(config_path)
            if not isinstance(loaded, dict):
                raise PipelineError(f"{config_path} must contain a JSON object.")
            user = loaded
            self.base_dir = config_path.resolve().parent
        self.unknown_keys = _merge(self.config, copy.deepcopy(user))
        if overrides:
            _merge(self.config, copy.deepcopy(overrides))
        self._expand(self.config)

    def _expand(self, section: Dict[str, Any]) -> None:
        data_root = str(self.config["data_root"])
        for key, value in section.items():
            if isinstance(value, dict):
                self._expand(value)
            elif isinstance(value, str) and "{data}" in value:
                section[key] = value.replace("{data}", data_root)

    def get(self, key: str, default: Any = None) -> Any:
        """Look up a value by dotted key, e.g. ``step5.epochs``."""

        value: Any = self.config
        for part in key.split("."):
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
        return default if value is None else value

    def path(self, key: str) -> Optional[Path]:
        """Like :meth:`get` for file-system paths, resolved to an absolute path."""

        value = self.get(key)
        if value is None:
            return None
        path = Path(str(value)).expanduser()
        return path if path.is_absolute() else self.base_dir / path

    def feature_config(self) -> FeatureConfig:
        device_rate = self.get("features.device_sample_rate", "auto")
        notch = self.config["features"].get("notch_hz")
        return replace(
            DEFAULT_FEATURE_CONFIG,
            sample_rate=int(self.get("features.sample_rate", 640)),
            device_sample_rate=None if device_rate in (None, "auto") else int(device_rate),
            notch_hz=float(notch) if notch else None,
        )

    def save(self, path: Path | str) -> None:
        with open(path, "w", encoding="utf8") as handle:
            json.dump(self.config, handle, indent=2)


def _file_signature(paths: Iterable[Path]) -> List[Tuple[str, int, int]]:
    signature = []
    for path in sorted(set(paths)):
        if path.is_file():
            stat = path.stat()
            signature.append((str(path), stat.st_size, stat.st_mtime_ns))
    return signature


def _tree(directory: Optional[Path], exclude: Sequence[str] = ()) -> List[Path]:
    if directory is None or not directory.is_dir():
        return []
    return [
        path
        for path in directory.rglob("*")
        if path.is_file() and not any(part in exclude for part in path.relative_to(directory).parts)
    ]


class WorkflowOrchestrator:
    """Runs the six steps and keeps track of what is up to date."""

    def __init__(self, config: WorkflowConfig):
        self.config = config
        self.device = resolve_device(config.get("device"))
        self.features = config.feature_config()
        self.seed = int(config.get("seed", 0))
        self._state_path = config.path("model_dir") / "pipeline_state.json"  # type: ignore[operator]
        logger.info("Using device: %s", self.device)
        for key in config.unknown_keys:
            logger.warning("Unknown configuration key '%s' (ignored).", key)

    # -- bookkeeping ------------------------------------------------------------------
    def _load_state(self) -> Dict[str, str]:
        try:
            return dict(read_json(self._state_path)) if self._state_path.is_file() else {}
        except PipelineError:
            return {}

    def _fingerprint(self, step: str, files: Iterable[Path]) -> str:
        payload = {
            "files": _file_signature(files),
            "settings": self.config.get(step, {}),
            "features": self.config.get("features"),
            "model": self.config.get("model"),
            "seed": self.seed,
        }
        return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode("utf8")).hexdigest()

    def _mark_done(self, step: str, fingerprint: str) -> None:
        state = self._load_state()
        state[step] = fingerprint
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._state_path, "w", encoding="utf8") as handle:
            json.dump(state, handle, indent=2)

    def _banner(self, title: str) -> None:
        logger.info("=" * 64)
        logger.info(title)
        logger.info("=" * 64)

    def _hidden(self, step: str) -> int:
        return int(self.config.get(f"{step}.hidden_channels", self.config.get("model.hidden_channels", 128)))

    def _save_history(self, model_path: Path, history: TrainingHistory) -> None:
        with open(model_path.with_suffix(".history.json"), "w", encoding="utf8") as handle:
            json.dump(history.to_dict(), handle, indent=2)

    @staticmethod
    def _format_metrics(metrics: Dict[str, Any]) -> str:
        parts = []
        for key, value in metrics.items():
            if isinstance(value, float):
                parts.append(f"{key}={value:.3f}")
            else:
                parts.append(f"{key}={value}")
        return ", ".join(parts)

    # -- Step 1 -------------------------------------------------------------------------
    def run_step1(
        self,
        input_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        *,
        required: bool = True,
    ) -> List[Dict[str, object]]:
        """Step 1: convert hexadecimal headset recordings into 8×3200 ``.npy`` segments."""

        self._banner("STEP 1: Converting hexadecimal EEG recordings")
        input_dir = Path(input_dir) if input_dir else self.config.path("step1.hex_dir")
        output_dir = Path(output_dir) if output_dir else self.config.path("step1.output_dir")
        assert input_dir is not None and output_dir is not None

        files = [input_dir] if input_dir.is_file() else [
            path for path in find_recordings(input_dir) if path.suffix.lower() != ".npy"
        ]
        if not files:
            message = (
                f"No recordings (.txt / .hex / .bin) found in {input_dir}. "
                "Put the headset's hex logs there, or run with --demo to try simulated data."
            )
            if required:
                raise PipelineError(message)
            logger.info("%s Skipping.", message.split(".")[0])
            return []

        reports: List[Dict[str, object]] = []
        for path in files:
            try:
                report = convert_recording(path, output_dir, self.features)
            except (RecordingError, ValueError) as error:
                logger.error("  %s: %s", path.name, error)
                reports.append({"file": path.name, "error": str(error)})
                continue
            reports.append(report)
            notes = []
            if report.get("resampled"):
                notes.append(f"device runs at {report['device_sample_rate']} Hz, resampled")
            if report.get("checksum_errors"):
                notes.append(f"{report['checksum_errors']} corrupt packets skipped")
            poor = report.get("poor_signal_fraction")
            if isinstance(poor, float) and poor > 0.1:
                notes.append(f"poor headset contact {poor:.0%} of the time")
            if report.get("format") != "thinkgear":
                notes.append("no headset packets found — decoded as legacy 8-channel dump")
            logger.info(
                "  %s: %.1f s → %d segments (%d clean)%s",
                path.name,
                float(report.get("duration_seconds", 0.0)),  # type: ignore[arg-type]
                report["segments"],
                report["clean_segments"],
                f"  [{'; '.join(notes)}]" if notes else "",
            )

        with open(output_dir / "conversion_report.json", "w", encoding="utf8") as handle:
            json.dump(reports, handle, indent=2)
        converted = [r for r in reports if "error" not in r]
        if required and not converted:
            raise PipelineError("None of the recordings could be converted; see the messages above.")
        logger.info("✓ Converted %d recording(s) into %s", len(converted), output_dir)
        return reports

    # -- Step 2 -------------------------------------------------------------------------
    def run_step2_train(self) -> TrainingHistory:
        """Step 2: train the "to viewer" model (aesthetic enjoyment)."""

        self._banner("STEP 2: Training the viewer enjoyment model")
        annotations = self.config.path("step2.annotations")
        assert annotations is not None
        samples = build_viewer_samples(annotations, self.config.path("data_root"), self.features, seed=self.seed)
        dataset = ViewerEnjoymentDataset(samples, feature_config=self.features)
        enjoyed = int(dataset.labels.sum())
        logger.info("%d listening windows: %d enjoyed, %d controls", len(dataset), enjoyed, len(dataset) - enjoyed)

        model, history = train_viewer_model(
            dataset,
            epochs=int(self.config.get("step2.epochs")),
            batch_size=int(self.config.get("step2.batch_size")),
            learning_rate=float(self.config.get("step2.learning_rate")),
            validation_split=float(self.config.get("step2.validation_split", 0.2)),
            hidden_channels=self._hidden("step2"),
            input_stride=int(self.config.get("model.input_stride", 8)),
            device=self.device,
            seed=self.seed,
        )
        model_path = self.config.path("step2.model_path")
        assert model_path is not None
        save_checkpoint(model_path, kind="viewer", model=model, feature_config=self.features, extra={"metrics": history.metrics})
        self._save_history(model_path, history)
        logger.info("✓ Viewer model saved to %s", model_path)
        logger.info("  %s", self._format_metrics(history.metrics))
        return history

    # -- Step 3 -------------------------------------------------------------------------
    def _reading_sessions(self, eeg_file: Optional[Path]) -> List[Dict[str, Any]]:
        if eeg_file is not None:
            return [{"eeg_path": Path(eeg_file), "reader": Path(eeg_file).stem}]
        sessions_file = self.config.path("step3.sessions")
        sessions: List[Dict[str, Any]] = []
        if sessions_file is not None and sessions_file.is_file():
            content = read_json(sessions_file)
            bases = (self.config.path("data_root"), sessions_file.parent, self.config.base_dir)
            for entry in content.get("sessions", []):
                if "eeg_file" not in entry:
                    raise PipelineError(f"{sessions_file}: a session is missing 'eeg_file'.")
                session = dict(entry)
                session["eeg_path"] = resolve_path(entry["eeg_file"], *bases)
                session.setdefault("reader", Path(entry["eeg_file"]).stem)
                sessions.append(session)
        else:
            eeg_dir = self.config.path("step3.eeg_dir")
            for path in find_recordings(eeg_dir) if eeg_dir else []:
                sessions.append({"eeg_path": path, "reader": path.stem})
        return sessions

    def run_step3(self, eeg_file: Optional[Path] = None) -> Optional[Dict[str, Any]]:
        """Step 3: score readers' EEG and rank the article's passages."""

        self._banner("STEP 3: Evaluating article-reading EEG")
        model, payload = load_checkpoint(self.config.path("step3.viewer_model"), kind="viewer", device=self.device)  # type: ignore[arg-type]
        features = checkpoint_feature_config(payload)

        sessions = self._reading_sessions(eeg_file)
        if not sessions:
            raise PipelineError(
                f"No reading recordings found in {self.config.path('step3.eeg_dir')}. "
                "Record readers with the headset while they read the article and put the logs there."
            )
        results_dir = self.config.path("step3.results_dir")
        assert results_dir is not None
        results_dir.mkdir(parents=True, exist_ok=True)

        article_path = self.config.path("step3.article")
        passages: List[str] = []
        if article_path is not None and article_path.is_file():
            passages = split_passages(article_path.read_text(encoding="utf8"))

        reader_windows: Dict[str, list] = {}
        reader_intervals: Dict[str, list] = {}
        for session in sessions:
            path: Path = session["eeg_path"]
            try:
                recording = load_recording(path, features)
            except RecordingError as error:
                logger.error("  %s: %s — skipped", path.name, error)
                continue
            windows = evaluate_enjoyment_windows(
                model,
                recording,
                sample_rate=features.sample_rate,
                window_size=float(self.config.get("step3.window_size")),
                stride=float(self.config.get("step3.stride")),
                device=self.device,
                feature_config=features,
            )
            if not windows:
                logger.warning("  %s: no usable windows (too short or poor signal) — skipped", path.name)
                continue
            reader = str(session["reader"])
            reader_windows[reader] = windows
            if passages:
                explicit = session.get("passage_times")
                if explicit:
                    reader_intervals[reader] = [(float(a), float(b)) for a, b in explicit]
                else:
                    reader_intervals[reader] = passage_intervals(
                        passages,
                        recording.duration,
                        start=float(session.get("reading_start", 0.0)),
                        end=session.get("reading_end"),
                    )
            with open(results_dir / f"{path.stem}_enjoyment.json", "w", encoding="utf8") as handle:
                json.dump(
                    {
                        "session_id": path.stem,
                        "eeg_file": str(path),
                        "windows": [
                            {
                                "start_time": round(w.start_time, 3),
                                "end_time": round(w.end_time, 3),
                                "enjoyment_score": round(float(w.enjoyment_score), 5),
                            }
                            for w in windows
                        ],
                    },
                    handle,
                    indent=2,
                )
            best = max(windows, key=lambda w: w.enjoyment_score)
            logger.info(
                "  %s: %d windows, peak enjoyment %.2f at %.0f–%.0f s",
                path.name,
                len(windows),
                best.enjoyment_score,
                best.start_time,
                best.end_time,
            )

        if not reader_windows:
            raise PipelineError("None of the reading recordings could be evaluated; see the messages above.")
        if not passages:
            logger.info(
                "✓ Window scores saved to %s. Add the article text as %s to rank its passages "
                "(needed for automatic creator labels in Step 4).",
                results_dir,
                article_path,
            )
            return None

        scores = score_passages(reader_windows, reader_intervals, passages)
        excellent, control = select_passages(scores, float(self.config.get("step3.excellent_fraction", 0.3)))
        summary = {
            "article": str(article_path),
            "article_hash": _article_hash(passages),
            "readers": len(reader_windows),
            "passages": [
                {
                    "index": s.index,
                    "score": round(s.score, 4),
                    "readers": s.readers,
                    "mean_probability": round(s.mean_probability, 4),
                    "preview": s.preview,
                }
                for s in scores
            ],
            "excellent": excellent,
            "control": control,
        }
        with open(results_dir / "passage_scores.json", "w", encoding="utf8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)

        logger.info("Passages ranked by reader enjoyment (%d readers):", len(reader_windows))
        for s in sorted(scores, key=lambda item: item.score, reverse=True):
            mark = "★" if s.index in excellent else ("·" if s.index in control else " ")
            logger.info("  %s passage %2d  score %+.2f  “%s…”", mark, s.index + 1, s.score, s.preview[:48])
        logger.info("✓ Passage scores saved to %s", results_dir / "passage_scores.json")
        return summary

    # -- Step 4 -------------------------------------------------------------------------
    def _creator_session_files(self) -> List[Path]:
        annotations_dir = self.config.path("step4.annotations_dir")
        files = sorted(annotations_dir.glob("*.json")) if annotations_dir and annotations_dir.is_dir() else []
        sessions_dir = self.config.path("step6.sessions_dir")
        if sessions_dir and sessions_dir.is_dir():
            files.extend(sorted(sessions_dir.glob("*/session.json")))
        return files

    def _derive_creator_labels(self) -> int:
        """Label creator sessions automatically from Step 3's passage ranking."""

        annotations_dir = self.config.path("step4.annotations_dir")
        results_dir = self.config.path("step3.results_dir")
        assert annotations_dir is not None and results_dir is not None
        scores_path = results_dir / "passage_scores.json"
        if not scores_path.is_file():
            return 0
        scores = read_json(scores_path)
        if not scores.get("excellent") or not scores.get("control"):
            return 0
        derived_dir = annotations_dir / "derived"
        default_article = self.config.path("step3.article")
        data_root = self.config.path("data_root")
        count = 0

        for path in self._creator_session_files():
            session = read_json(path)
            if not isinstance(session, dict) or session.get("excellent_passages"):
                continue
            if not (session.get("keystroke_log") or session.get("writing_timeline")) or not session.get("eeg_file"):
                continue
            bases = (path.parent, data_root, self.config.base_dir, data_root.parent if data_root else None)
            article = resolve_path(session["article"], *bases) if session.get("article") else None
            if article is None or not article.is_file():
                local = path.parent / "article.txt"
                article = local if local.is_file() else default_article
            if article is None or not article.is_file():
                continue
            passages = split_passages(article.read_text(encoding="utf8"))
            if _article_hash(passages) != scores.get("article_hash"):
                continue  # this session wrote a different article than the one readers scored

            if session.get("writing_timeline"):
                timeline = list(session["writing_timeline"])
            else:
                events = read_keystroke_log(resolve_path(session["keystroke_log"], *bases))
                timeline = timeline_from_keystrokes(events, passages, started_at=session.get("started_at"))
            eeg_path = resolve_path(session["eeg_file"], *bases)
            labelled = annotations_from_timeline(eeg_path, timeline, scores["excellent"], scores["control"])
            if not labelled:
                continue
            fallback_id = path.parent.name if path.name == "session.json" else path.stem
            session_id = str(session.get("session_id") or fallback_id)
            derived_dir.mkdir(parents=True, exist_ok=True)
            with open(derived_dir / f"{session_id}.json", "w", encoding="utf8") as handle:
                json.dump(
                    {
                        "session_id": session_id,
                        "eeg_file": str(eeg_path),
                        "derived_from": {"session": str(path), "article": str(article), "passage_scores": str(scores_path)},
                        "excellent_passages": [
                            {
                                "start_time": round(a.start_time, 3),
                                "end_time": round(a.end_time, 3),
                                "label": "creatable" if a.is_creatable else "not_creatable",
                            }
                            for a in labelled
                        ],
                    },
                    handle,
                    indent=2,
                )
            count += 1
            logger.info(
                "  %s: labelled %d stretches from the keystroke log and the passage ranking",
                session_id,
                len(labelled),
            )
        return count

    def _creator_annotations(self) -> List[CreatorAnnotation]:
        annotations_dir = self.config.path("step4.annotations_dir")
        assert annotations_dir is not None
        self._derive_creator_labels()
        files: List[Path] = []
        for directory in (annotations_dir, annotations_dir / "derived"):
            if directory.is_dir():
                files.extend(sorted(directory.glob("*.json")))
        legacy = self.config.path("step4.annotations")
        if legacy is not None and legacy.is_file() and legacy not in files:
            files.append(legacy)
        annotations: List[CreatorAnnotation] = []
        seen = set()
        for path in files:
            for annotation in annotations_from_manual_file(path, self.config.path("data_root")):
                key = (str(Path(annotation.brainwave_path).resolve()), round(annotation.start_time, 2), round(annotation.end_time, 2))
                if key not in seen:
                    seen.add(key)
                    annotations.append(annotation)
        return annotations

    def run_step4_train(self) -> TrainingHistory:
        """Step 4: train the "to creator" model (creatable state detection)."""

        self._banner("STEP 4: Training the creator state model")
        annotations = self._creator_annotations()
        if not annotations:
            raise PipelineError(
                "No creator labels available. Either\n"
                f"  • describe a writing session in {self.config.path('step4.annotations_dir')} with its "
                "'eeg_file' and 'keystroke_log' and run Step 3 so its passages are ranked, or\n"
                "  • mark stretches by hand under 'excellent_passages' (see data/README.md)."
            )
        dataset = CreatorStateDataset(annotations, sample_rate=self.features.sample_rate, feature_config=self.features)
        creatable = int(dataset.labels.sum())
        logger.info("%d creator windows: %d creatable, %d not", len(dataset), creatable, len(dataset) - creatable)

        model, history = train_creator_model(
            dataset,
            epochs=int(self.config.get("step4.epochs")),
            batch_size=int(self.config.get("step4.batch_size")),
            learning_rate=float(self.config.get("step4.learning_rate")),
            validation_split=float(self.config.get("step4.validation_split", 0.2)),
            hidden_channels=self._hidden("step4"),
            input_stride=int(self.config.get("model.input_stride", 8)),
            device=self.device,
            seed=self.seed,
        )
        model_path = self.config.path("step4.model_path")
        assert model_path is not None
        save_checkpoint(model_path, kind="creator", model=model, feature_config=self.features, extra={"metrics": history.metrics})
        self._save_history(model_path, history)
        logger.info("✓ Creator model saved to %s", model_path)
        logger.info("  %s", self._format_metrics(history.metrics))
        auc = history.metrics.get("auc")
        if isinstance(auc, float) and auc == auc and auc < 0.6:
            logger.warning(
                "  The creator model barely separates the two states on held-out data (AUC %.2f). "
                "The live loop will rarely or randomly fire until more sessions are recorded.",
                auc,
            )
        return history

    # -- Step 5 -------------------------------------------------------------------------
    def run_step5_train(self) -> TrainingHistory:
        """Step 5: train the EEG → Processing parameters model."""

        self._banner("STEP 5: Training the Processing parameter model")
        dataset_root = self.config.path("step5.dataset_root")
        assert dataset_root is not None
        if not dataset_root.is_dir():
            raise PipelineError(
                f"Dataset not found: {dataset_root}. It needs 'artworks/' with the images and "
                "'brainwaves/' with the EEG recorded while viewing them."
            )
        try:
            model, history, _names = train_processing_parameter_model(
                dataset_root,
                epochs=int(self.config.get("step5.epochs")),
                batch_size=int(self.config.get("step5.batch_size")),
                learning_rate=float(self.config.get("step5.learning_rate")),
                validation_split=float(self.config.get("step5.validation_split", 0.2)),
                hidden_channels=self._hidden("step5"),
                num_blocks=int(self.config.get("step5.num_blocks", 4)),
                dropout=float(self.config.get("step5.dropout", 0.1)),
                input_stride=int(self.config.get("model.input_stride", 8)),
                processing_weights=self.config.path("step5.processing_weights"),
                feature_config=self.features,
                device=self.device,
                seed=self.seed,
            )
        except (FileNotFoundError, ValueError) as error:
            raise PipelineError(str(error)) from error
        model_path = self.config.path("step5.model_path")
        assert model_path is not None
        save_checkpoint(model_path, kind="parameter", model=model, feature_config=self.features, extra={"metrics": history.metrics})
        self._save_history(model_path, history)
        logger.info("✓ Parameter model saved to %s", model_path)
        logger.info("  %s", self._format_metrics(history.metrics))
        return history

    # -- Step 6 -------------------------------------------------------------------------
    def run_step6_realtime(
        self,
        source: Optional[str] = None,
        *,
        duration: Optional[float] = None,
        speed: float = 1.0,
        use_osc: Optional[bool] = None,
        record: Optional[bool] = None,
        watch_document: Optional[Path] = None,
        on_decision: Any = None,
        record_only: bool = False,
    ):
        """Step 6: the real-time feedback loop.

        With ``record_only`` no models are needed: the session (EEG and, with
        ``watch_document``, the writing) is recorded for later training.
        """

        self._banner("STEP 6: Recording a session" if record_only else "STEP 6: Real-time feedback loop")
        if record_only:
            creator_model = parameter_model = None
            features = self.features
        else:
            try:
                creator_model, creator_payload = load_checkpoint(self.config.path("step6.creator_model"), kind="creator", device=self.device)  # type: ignore[arg-type]
                parameter_model, parameter_payload = load_checkpoint(self.config.path("step6.parameter_model"), kind="parameter", device=self.device)  # type: ignore[arg-type]
            except CheckpointError as error:
                raise PipelineError(
                    f"{error}\nTrain the models first (--all), or record a first writing session "
                    "without feedback using --record-only."
                ) from error
            features = checkpoint_feature_config(creator_payload)
            other = checkpoint_feature_config(parameter_payload)
            if (features.sample_rate, features.bands, features.segment_seconds) != (other.sample_rate, other.bands, other.segment_seconds):
                raise PipelineError(
                    "The creator model and the parameter model were trained with different signal "
                    "settings. Re-run Steps 4 and 5 with the same configuration."
                )
            features = replace(features, device_sample_rate=self.features.device_sample_rate)

        specification = source or self.config.get("step6.source")
        if not specification:
            raise PipelineError(
                "No EEG source given. Start the loop with one of:\n"
                "  --source serial:/dev/ttyUSB0     the headset's serial port (COM3 on Windows)\n"
                "  --source follow:/path/log.txt    a hex log another program is writing\n"
                "  --source replay:/path/log.txt    a finished recording\n"
                "  --source sim                     the built-in simulator"
            )
        try:
            eeg_source = open_source(str(specification), speed=speed, nominal_sample_rate=features.sample_rate)
        except SourceError as error:
            raise PipelineError(str(error)) from error

        settings = LoopSettings(
            creatable_threshold=float(self.config.get("step6.creatable_threshold")),
            release_threshold=self.config.get("step6.release_threshold"),
            probability_smoothing=float(self.config.get("step6.probability_smoothing", 0.6)),
            parameter_smoothing=float(self.config.get("step6.parameter_smoothing", 0.6)),
            update_interval=float(self.config.get("step6.update_interval")),
            calibration_seconds=float(self.config.get("step6.calibration_seconds", 20.0)),
        )
        output_dir = self.config.path("output_dir")
        assert output_dir is not None
        osc = self.config.get("step6.use_osc", True) if use_osc is None else use_osc
        if record_only:
            controller: Any = PassiveController(settings)
            bridge = None
            record = True
        else:
            controller = FeedbackController(creator_model, parameter_model, settings, device=self.device)
            bridge = create_bridge(
                use_osc=bool(osc),
                config=OSCConfig(host=str(self.config.get("step6.osc_host")), port=int(self.config.get("step6.osc_port"))),
                json_path=output_dir / "current_parameters.json",
            )

        recorder = None
        if self.config.get("step6.record", True) if record is None else record:
            sessions_dir = self.config.path("step6.sessions_dir")
            assert sessions_dir is not None
            watch = watch_document or self.config.path("step6.watch_document")
            recorder = SessionRecorder(sessions_dir / time.strftime("%Y%m%d-%H%M%S"), watch_document=watch)
            recorder.update_metadata(source=str(specification), settings=settings.__dict__, session_id=recorder.directory.name)
            logger.info("Recording this session to %s", recorder.directory)

        if not record_only:
            logger.info(
                "Display: OSC %s  |  parameters mirrored to %s",
                f"→ {self.config.get('step6.osc_host')}:{self.config.get('step6.osc_port')}" if osc else "off",
                output_dir / "current_parameters.json",
            )
            logger.info("Open scripts/processing_templates/dynamic_texture_osc in Processing and press Run.")
        logger.info("Press Ctrl+C to stop.")

        session = RealtimeSession(
            eeg_source,
            controller,
            feature_config=features,
            bridge=bridge,
            recorder=recorder,
            on_decision=on_decision,
        )
        summary = session.run(duration=duration)
        if recorder is not None and recorder._metadata.get("document"):
            document = Path(str(recorder._metadata["document"]))
            if document.is_file():
                shutil.copyfile(document, recorder.directory / "article.txt")
        return summary

    # -- everything ---------------------------------------------------------------------
    def run_training(self, *, force: bool = False) -> None:
        """Steps 1–5, skipping those whose inputs have not changed."""

        state = {} if force else self._load_state()
        data_root = self.config.path("data_root")
        assert data_root is not None

        self.run_step1(required=False)

        def stale(step: str, fingerprint: str, outputs: Sequence[Optional[Path]]) -> bool:
            if state.get(step) != fingerprint or any(p is None or not p.exists() for p in outputs):
                return True
            logger.info("Step %s is up to date — skipping (use --force to redo it).", step[-1])
            return False

        viewer_model = self.config.path("step2.model_path")
        fp2 = self._fingerprint("step2", _tree(data_root / "step2_viewer_training") + [self.config.path("step2.annotations")])  # type: ignore[list-item]
        if stale("step2", fp2, [viewer_model]):
            self.run_step2_train()
            self._mark_done("step2", fp2)

        results_dir = self.config.path("step3.results_dir")
        assert results_dir is not None
        step3_inputs = _tree(self.config.path("step3.eeg_dir")) + [
            p for p in (viewer_model, self.config.path("step3.article"), self.config.path("step3.sessions")) if p
        ]
        fp3 = self._fingerprint("step3", step3_inputs)
        if stale("step3", fp3, [results_dir]):
            self.run_step3()
            self._mark_done("step3", fp3)

        step4_inputs = (
            _tree(self.config.path("step4.eeg_dir"))
            + _tree(self.config.path("step4.annotations_dir"), exclude=("derived",))
            + [results_dir / "passage_scores.json"]
            + [
                p
                for p in _tree(self.config.path("step6.sessions_dir"))
                if p.name in ("keystrokes.jsonl", "article.txt")
            ]
        )
        fp4 = self._fingerprint("step4", step4_inputs)
        if stale("step4", fp4, [self.config.path("step4.model_path")]):
            self.run_step4_train()
            self._mark_done("step4", fp4)

        fp5 = self._fingerprint("step5", _tree(self.config.path("step5.dataset_root")))
        if stale("step5", fp5, [self.config.path("step5.model_path")]):
            self.run_step5_train()
            self._mark_done("step5", fp5)

    def check(self) -> bool:
        """Report which inputs each step can find, without running anything."""

        self._banner("Checking the data layout")
        ok = True

        def report(label: str, found: bool, detail: str) -> None:
            nonlocal ok
            ok = ok and found
            logger.info("  [%s] %-34s %s", "ok" if found else "missing", label, detail)

        annotations = self.config.path("step2.annotations")
        report("Step 2 listening annotations", bool(annotations and annotations.is_file()), str(annotations))
        eeg_dir = self.config.path("step3.eeg_dir")
        readings = find_recordings(eeg_dir) if eeg_dir else []
        report("Step 3 reading recordings", bool(readings), f"{len(readings)} in {eeg_dir}")
        article = self.config.path("step3.article")
        report("Step 3 article text", bool(article and article.is_file()), str(article))
        annotation_dir = self.config.path("step4.annotations_dir")
        sessions = list(annotation_dir.glob("*.json")) if annotation_dir and annotation_dir.is_dir() else []
        report("Step 4 creator sessions", bool(sessions), f"{len(sessions)} in {annotation_dir}")
        dataset_root = self.config.path("step5.dataset_root")
        artworks = list((dataset_root / "artworks").glob("*")) if dataset_root and (dataset_root / "artworks").is_dir() else []
        viewing = find_recordings(dataset_root / "brainwaves", recursive=True) if dataset_root else []
        report("Step 5 artworks", bool(artworks), f"{len(artworks)} images")
        report("Step 5 viewing recordings", bool(viewing), f"{len(viewing)} recordings")
        for name in ("step2.model_path", "step4.model_path", "step5.model_path"):
            path = self.config.path(name)
            logger.info("  [%s] %-34s %s", "ok" if path and path.is_file() else "--", f"trained model ({name.split('.')[0]})", path)
        return ok


def _article_hash(passages: Sequence[str]) -> str:
    normalised = "\n".join(" ".join(passage.split()) for passage in passages)
    return hashlib.sha1(normalised.encode("utf8")).hexdigest()


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def run_demo(args: argparse.Namespace) -> int:
    """Generate a simulated study, run all six steps on it and check the outcome."""

    from .simulator import generate_demo_dataset

    demo_root = Path(args.demo).expanduser().resolve()
    overrides: Dict[str, Any] = {"data_root": str(demo_root)}
    if args.quick:
        overrides.update(_quick_overrides())
    if args.device:
        overrides["device"] = args.device
    config = WorkflowConfig(args.config, overrides)
    _configure_logging(config.path("output_dir"))

    truth_path = demo_root / "demo_truth.json"
    if args.force or not truth_path.is_file():
        logger.info("Generating a simulated study in %s …", demo_root)
        if demo_root.exists() and args.force:
            shutil.rmtree(demo_root)
        truth = generate_demo_dataset(demo_root, seed=int(config.get("seed", 0)))
    else:
        logger.info("Using the simulated study already in %s (use --force to regenerate).", demo_root)
        truth = read_json(truth_path)

    orchestrator = WorkflowOrchestrator(config)
    orchestrator.run_training(force=args.force)

    decisions: List[Tuple[float, str]] = []
    source = args.source or f"replay:{demo_root / 'step6_realtime' / 'demo_live_session.txt'}"
    speed = args.speed if args.speed is not None else 0.0
    summary = orchestrator.run_step6_realtime(
        source,
        duration=args.duration,
        speed=speed,
        use_osc=not args.no_osc,
        record=not args.no_record,
        on_decision=lambda when, decision: decisions.append((when, decision.state)),
    )

    orchestrator._banner("DEMO RESULT")
    scores_path = config.path("step3.results_dir") / "passage_scores.json"  # type: ignore[operator]
    success = True
    if scores_path.is_file():
        found = set(read_json(scores_path).get("excellent", []))
        actual = set(truth.get("excellent_passages", []))
        logger.info(
            "Passages ranked excellent from reader EEG: %s — of these, %d/%d were truly made enjoyable.",
            sorted(i + 1 for i in found),
            len(found & actual),
            len(found),
        )
        success = success and len(found & actual) >= max(1, len(found) - 1)

    timeline = truth.get("live_session", [])
    judged = correct = 0
    for when, state in decisions:
        if state not in ("creatable", "idle"):
            continue
        window_centre = when - 2.5
        actual_state = next((seg["state"] for seg in timeline if seg["start"] <= window_centre < seg["end"]), None)
        if actual_state is None:
            continue
        # Windows that straddle a change of state have no single right answer.
        segment = next(seg for seg in timeline if seg["start"] <= window_centre < seg["end"])
        if when - 5.0 < segment["start"] or when > segment["end"]:
            continue
        judged += 1
        correct += int((state == "creatable") == (actual_state == "creatable"))
    if judged and not args.source:
        logger.info(
            "Live loop: %d of %d windows classified correctly (%.0f%%); %d parameter updates sent.",
            correct,
            judged,
            100.0 * correct / judged,
            summary.updates_sent,
        )
        success = success and correct / judged >= 0.7 and summary.updates_sent > 0
    logger.info("Result: %s", "the whole loop works end to end." if success else "the loop ran, but results are weaker than expected.")
    return 0 if success else 2


def _quick_overrides() -> Dict[str, Any]:
    return {
        "model": {"hidden_channels": 32},
        "step2": {"epochs": 12},
        "step4": {"epochs": 15},
        "step5": {"epochs": 30},
    }


def launch_sketch() -> Optional[subprocess.Popen]:
    """Start the Processing sketch if a Processing command-line tool is installed."""

    for command in (["processing-java", f"--sketch={SKETCH_DIRECTORY}", "--run"], ["processing", "cli", f"--sketch={SKETCH_DIRECTORY}", "--run"]):
        if shutil.which(command[0]):
            logger.info("Starting the Processing sketch (%s)…", command[0])
            return subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    logger.warning(
        "Could not start the sketch automatically (no 'processing-java' or 'processing' command found). "
        "Open %s in Processing and press Run.",
        SKETCH_DIRECTORY,
    )
    return None


def _configure_logging(output_dir: Optional[Path]) -> None:
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    root.setLevel(logging.INFO)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(console)
    if output_dir is not None:
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(output_dir / "workflow.log", encoding="utf8")
            file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
            root.addHandler(file_handler)
        except OSError:
            pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aesthetic Beacon: EEG-based creative feedback workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", type=Path, help="Configuration file (JSON); see config.example.json")
    parser.add_argument("--step", type=int, choices=[1, 2, 3, 4, 5, 6], help="Run a single step")
    parser.add_argument("--all", action="store_true", help="Run steps 1–5, then the live loop if a source is set")
    parser.add_argument("--demo", nargs="?", const="demo_data", metavar="DIR",
                        help="Run everything on simulated data written to DIR (default: demo_data)")
    parser.add_argument("--check", action="store_true", help="Show which inputs are present, then exit")
    parser.add_argument("--train", action="store_true", help="Accepted for steps 2, 4 and 5 (training is their only mode)")
    parser.add_argument("--realtime", action="store_true", help="Accepted for step 6")
    parser.add_argument("--input", type=Path, help="Step 1: recording or folder to convert. Step 3: one recording to score")
    parser.add_argument("--output", type=Path, help="Step 1: where to write the converted segments")
    parser.add_argument("--source", help="Step 6 EEG source: serial:<port>, follow:<file>, replay:<file>, tcp:<host>:<port>, sim")
    parser.add_argument("--duration", type=float, help="Step 6: stop after this many seconds")
    parser.add_argument("--speed", type=float, default=None, help="Replay/simulator speed (1 = real time, 0 = as fast as possible)")
    parser.add_argument("--watch-document", type=Path, help="Step 6: text file the creator is writing; its growth is logged")
    parser.add_argument("--launch-sketch", action="store_true", help="Step 6: start the Processing sketch if possible")
    parser.add_argument("--record-only", action="store_true",
                        help="Step 6: record EEG (and the document) without models or feedback — for the first session")
    parser.add_argument("--no-osc", action="store_true", help="Step 6: do not send OSC (JSON file only)")
    parser.add_argument("--no-record", action="store_true", help="Step 6: do not record the session")
    parser.add_argument("--force", action="store_true", help="Redo steps even if their inputs are unchanged")
    parser.add_argument("--quick", action="store_true", help="Small, fast models — for trying things out")
    parser.add_argument("--device", help="cpu, cuda or auto")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.demo:
            return run_demo(args)

        overrides: Dict[str, Any] = {}
        if args.quick:
            overrides.update(_quick_overrides())
        if args.device:
            overrides["device"] = args.device
        config = WorkflowConfig(args.config, overrides)
        _configure_logging(config.path("output_dir"))
        orchestrator = WorkflowOrchestrator(config)

        def live() -> None:
            sketch = launch_sketch() if args.launch_sketch and not args.record_only else None
            try:
                orchestrator.run_step6_realtime(
                    args.source,
                    duration=args.duration,
                    speed=1.0 if args.speed is None else args.speed,
                    use_osc=False if args.no_osc else None,
                    record=False if args.no_record else None,
                    watch_document=args.watch_document,
                    record_only=args.record_only,
                )
            finally:
                if sketch is not None:
                    sketch.terminate()

        if args.check:
            return 0 if orchestrator.check() else 1
        if args.all:
            orchestrator.run_training(force=args.force)
            if args.source or config.get("step6.source"):
                live()
            else:
                logger.info("")
                logger.info("Training is complete. Start the live loop with, for example:")
                logger.info("  python src/workflow_orchestrator.py --step 6 --source serial:/dev/ttyUSB0")
        elif args.step == 1:
            orchestrator.run_step1(args.input, args.output)
        elif args.step == 2:
            orchestrator.run_step2_train()
        elif args.step == 3:
            orchestrator.run_step3(args.input)
        elif args.step == 4:
            orchestrator.run_step4_train()
        elif args.step == 5:
            orchestrator.run_step5_train()
        elif args.step == 6:
            live()
        else:
            parser.print_help()
        return 0
    except (PipelineError, CheckpointError, TrainingError, RecordingError) as error:
        logging.getLogger().error("\nCannot continue: %s", error)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
