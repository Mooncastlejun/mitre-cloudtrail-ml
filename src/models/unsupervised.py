"""Unsupervised anomaly detectors on behavioral features: IsolationForest & LOF.

Trained with NO labels (labels used only for evaluation). Both return a
per-event anomaly score in [0, 1] (higher = more anomalous) so they can be
compared and thresholded consistently.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from ..config import RANDOM_SEED
from ..data.features import behavioral_matrix
from .scoring import Score01


class IsolationForestModel:
    name = "isolation_forest"

    def __init__(self, contamination: float = 0.43):
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=200, contamination=contamination, random_state=RANDOM_SEED, n_jobs=-1)
        self.norm = Score01()

    def fit(self, df: pd.DataFrame) -> "IsolationForestModel":
        X = self.scaler.fit_transform(behavioral_matrix(df))
        self.model.fit(X)
        self.norm.fit(-self.model.score_samples(X))
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        X = self.scaler.transform(behavioral_matrix(df))
        # sklearn: higher score_samples = more normal; invert so high = anomalous
        return self.norm(-self.model.score_samples(X))


class LOFModel:
    name = "lof"

    def __init__(self, contamination: float = 0.43, n_neighbors: int = 30):
        self.scaler = StandardScaler()
        self.model = LocalOutlierFactor(
            n_neighbors=n_neighbors, contamination=contamination, novelty=True)
        self.norm = Score01()

    def fit(self, df: pd.DataFrame) -> "LOFModel":
        X = self.scaler.fit_transform(behavioral_matrix(df))
        self.model.fit(X)
        self.norm.fit(-self.model.score_samples(X))
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        X = self.scaler.transform(behavioral_matrix(df))
        return self.norm(-self.model.score_samples(X))
