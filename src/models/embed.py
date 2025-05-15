"""MiniLM sentence-embedding + IsolationForest detector.

Embeds each event's text view (eventName + source + region + identity + error)
with a MiniLM sentence-transformer, then runs IsolationForest on the embeddings.
This captures semantic similarity between API calls that one-hot features miss.

Falls back to a hashing bag-of-tokens vectorizer when sentence-transformers /
torch aren't installed, keeping the pipeline runnable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from ..config import RANDOM_SEED
from .scoring import Score01

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class _HashingEmbedder:
    """Deterministic hashing bag-of-words fallback (no heavy deps)."""

    backend = "hashing"

    def __init__(self, dim: int = 128):
        self.dim = dim

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, t in enumerate(texts):
            for tok in str(t).lower().split():
                out[i, hash(tok) % self.dim] += 1.0
            n = np.linalg.norm(out[i]) or 1.0
            out[i] /= n
        return out


class _MiniLMEmbedder:
    backend = "minilm"

    def __init__(self):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(MODEL_NAME)

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, show_progress_bar=False), dtype="float32")


class EmbeddingIForestModel:
    name = "minilm_iforest"

    def __init__(self, contamination: float = 0.43):
        self.model = IsolationForest(
            n_estimators=200, contamination=contamination, random_state=RANDOM_SEED, n_jobs=-1)
        try:
            self.embedder = _MiniLMEmbedder()
        except Exception:
            self.embedder = _HashingEmbedder()
        self.backend = self.embedder.backend
        self.norm = Score01()

    def _emb(self, df: pd.DataFrame) -> np.ndarray:
        return self.embedder.encode(df["event_text"].fillna("").tolist())

    def fit(self, df: pd.DataFrame) -> "EmbeddingIForestModel":
        emb = self._emb(df)
        self.model.fit(emb)
        self.norm.fit(-self.model.score_samples(emb))
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        return self.norm(-self.model.score_samples(self._emb(df)))
