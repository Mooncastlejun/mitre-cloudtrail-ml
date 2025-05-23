"""Persist / load the chosen detector so the serving layer can score live events."""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

from ..config import ARTIFACT_DIR

MODEL_PATH = ARTIFACT_DIR / "detector.pkl"


@dataclass
class Bundle:
    model: object
    threshold: float
    model_name: str
    backend: str


def save(bundle: Bundle, path: Path = MODEL_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(bundle, fh)
    return path


def load(path: Path = MODEL_PATH) -> Bundle:
    with open(path, "rb") as fh:
        return pickle.load(fh)
