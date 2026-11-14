"""Shared fixtures.

The end-to-end fixture generates a small simulated study once per test run and
trains all models on it; several test modules reuse the result.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def demo(tmp_path_factory):
    """A trained demo study: returns the paths and the result of the run."""

    from src.workflow_orchestrator import main

    root = tmp_path_factory.mktemp("demo")
    data = root / "data"
    config = root / "config.json"
    config.write_text(
        json.dumps(
            {
                "device": "cpu",
                "model": {"hidden_channels": 24},
                "step2": {"epochs": 10},
                "step4": {"epochs": 12},
                "step5": {"epochs": 20},
                "step6": {"osc_port": 12987},
            }
        )
    )
    code = main(["--demo", str(data), "--config", str(config)])
    return {"root": root, "data": data, "config": config, "exit_code": code}
