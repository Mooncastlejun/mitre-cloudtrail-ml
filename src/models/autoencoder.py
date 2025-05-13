"""Autoencoder anomaly detector.

Trains on behavioral features (unlabeled); high reconstruction error = anomaly.
Uses a small PyTorch MLP autoencoder when torch is available, and falls back to
a PCA-based linear autoencoder (numpy only) otherwise — so the pipeline runs in
a minimal environment while matching the project's torch model when installed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ..config import RANDOM_SEED
from ..data.features import behavioral_matrix
from .scoring import Score01


class AutoencoderModel:
    name = "autoencoder"

    def __init__(self, latent: int = 4, epochs: int = 40):
        self.scaler = StandardScaler()
        self.latent = latent
        self.epochs = epochs
        self.backend = "torch"
        self._impl = None
        self.norm = Score01()

    def fit(self, df: pd.DataFrame) -> "AutoencoderModel":
        X = self.scaler.fit_transform(behavioral_matrix(df)).astype("float32")
        try:
            self._impl = _TorchAE(X.shape[1], self.latent, self.epochs).fit(X)
        except Exception:
            self.backend = "pca"
            self._impl = _PCAAE(self.latent).fit(X)
        self.norm.fit(self._recon_error(X))
        return self

    def _recon_error(self, X: np.ndarray) -> np.ndarray:
        recon = self._impl.reconstruct(X)
        return np.mean((X - recon) ** 2, axis=1)

    def score(self, df: pd.DataFrame) -> np.ndarray:
        X = self.scaler.transform(behavioral_matrix(df)).astype("float32")
        return self.norm(self._recon_error(X))


class _PCAAE:
    """Linear autoencoder == truncated PCA reconstruction (numpy fallback)."""

    def __init__(self, latent: int):
        self.latent = latent

    def fit(self, X: np.ndarray) -> "_PCAAE":
        X = np.asarray(X, dtype=np.float64)
        self.mean = X.mean(axis=0)
        Xc = X - self.mean
        _u, _s, vt = np.linalg.svd(Xc, full_matrices=False)
        self.components = vt[: self.latent]
        return self

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        Xc = X - self.mean
        with np.errstate(all="ignore"):
            z = Xc @ self.components.T
            recon = z @ self.components + self.mean
        return np.nan_to_num(recon, nan=0.0, posinf=0.0, neginf=0.0)


class _TorchAE:
    def __init__(self, n_in: int, latent: int, epochs: int):
        import torch
        import torch.nn as nn

        torch.manual_seed(RANDOM_SEED)
        self.torch = torch
        hidden = max(latent * 2, 8)
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.ReLU(),
            nn.Linear(hidden, latent), nn.ReLU(),
            nn.Linear(latent, hidden), nn.ReLU(),
            nn.Linear(hidden, n_in),
        )
        self.epochs = epochs

    def fit(self, X: np.ndarray) -> "_TorchAE":
        torch = self.torch
        opt = torch.optim.Adam(self.net.parameters(), lr=1e-2)
        loss_fn = torch.nn.MSELoss()
        xt = torch.tensor(X)
        self.net.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            out = self.net(xt)
            loss = loss_fn(out, xt)
            loss.backward()
            opt.step()
        return self

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        torch = self.torch
        self.net.eval()
        with torch.no_grad():
            return self.net(torch.tensor(X)).numpy()
