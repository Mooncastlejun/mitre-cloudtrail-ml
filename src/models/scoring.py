"""Fit-time score normalization.

Anomaly detectors emit raw scores on arbitrary scales. To make a persisted
threshold meaningful at serving time, we freeze the min/max (and a robust
fallback) from the training scores and apply that same transform online — rather
than re-normalizing per batch, which would make any fixed threshold useless.
"""
from __future__ import annotations

import numpy as np


class Score01:
    def fit(self, raw: np.ndarray) -> "Score01":
        raw = np.asarray(raw, dtype=float)
        self.lo = float(np.min(raw))
        self.hi = float(np.max(raw))
        if self.hi - self.lo < 1e-9:
            self.hi = self.lo + 1.0
        return self

    def __call__(self, raw: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw, dtype=float)
        return np.clip((raw - self.lo) / (self.hi - self.lo), 0.0, 1.0)
