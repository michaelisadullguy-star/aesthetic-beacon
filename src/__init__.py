"""Aesthetic Beacon: an EEG-driven creative feedback loop.

The public names of the six pipeline stages can be imported from here, e.g.
``from src import train_creator_model``; see :mod:`src.pipeline` for their
documentation.  They are loaded on first use, so light-weight modules such as
:mod:`src.thinkgear` can be used without the machine-learning stack.
"""
from __future__ import annotations

import importlib
from typing import Any


def __getattr__(name: str) -> Any:
    if name.startswith("_"):
        raise AttributeError(name)
    try:
        return importlib.import_module(f"{__name__}.{name}")
    except ModuleNotFoundError as error:
        if error.name != f"{__name__}.{name}":
            raise
    pipeline = importlib.import_module(f"{__name__}.pipeline")
    if name in pipeline.__all__:
        return getattr(pipeline, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
