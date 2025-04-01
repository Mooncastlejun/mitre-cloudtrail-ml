"""Feature engineering for CloudTrail anomaly detection.

Turns the flat event frame into a numeric matrix. Two feature families:

- **behavioral** — rarity + rate signals that don't encode the label directly
  (per-principal event counts, eventName rarity, error flags, off-hours, region
  novelty). These are what the *unsupervised* models rely on.
- **categorical one-hot** — eventName / source / region / identity type. Useful
  but leaky for supervised models (the label is derived from eventName), which
  is exactly the memorization trap we demonstrate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    n = max(len(df), 1)
    df["hour"] = df["eventTime"].dt.hour.fillna(0).astype(int) if "eventTime" in df else 0
    df["is_off_hours"] = ((df["hour"] < 6) | (df["hour"] >= 22)).astype(int)

    # per-principal activity as a *share* of traffic (scale-invariant: stable
    # across a 6k training set vs an 800-event live window, unlike raw counts)
    principal = df["identity_principal"].fillna(df["identity_username"]).fillna("anon")
    df["principal_event_share"] = principal.map(principal.value_counts()) / n

    # rarity of the API call (bounded) + its share
    name_counts = df["eventName"].value_counts()
    df["eventname_share"] = df["eventName"].map(name_counts).fillna(0) / n
    df["eventname_rarity"] = 1.0 / np.log1p(df["eventName"].map(name_counts).fillna(0) + 1)

    # (principal, eventName) novelty — first time this identity makes this call
    pe = principal.astype(str) + "|" + df["eventName"].astype(str)
    df["is_new_principal_api"] = (pe.map(pe.value_counts()) == 1).astype(int)

    # region spread per principal (lateral movement / unusual region)
    df["principal_region_spread"] = df.groupby(principal)["awsRegion"].transform("nunique").fillna(1)

    # error signals (probing / permission failures)
    df["has_error"] = df.get("has_error", 0)

    # source IP share + rarity
    ip_counts = df["sourceIPAddress"].value_counts()
    df["ip_share"] = df["sourceIPAddress"].map(ip_counts).fillna(0) / n
    df["ip_rarity"] = 1.0 / np.log1p(df["sourceIPAddress"].map(ip_counts).fillna(0) + 1)

    df["is_readonly"] = df["readOnly"].map({True: 1, False: 0, "true": 1, "false": 0}).fillna(0).astype(int)
    return df


BEHAVIORAL_COLS = [
    "is_off_hours", "principal_event_share", "eventname_share", "eventname_rarity",
    "is_new_principal_api", "principal_region_spread", "has_error",
    "ip_share", "ip_rarity", "is_readonly",
]


def behavioral_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df[BEHAVIORAL_COLS].astype(float).fillna(0.0)


def categorical_onehot(df: pd.DataFrame, top_k: int = 40) -> pd.DataFrame:
    """One-hot of high-cardinality categoricals, capped to the top_k values.

    NOTE: includes eventName, which the MITRE label is derived from — using this
    for supervised training is what causes label leakage / memorization.
    """
    frames = []
    for col, k in [("eventName", top_k), ("eventSource", 15), ("awsRegion", 12),
                   ("identity_type", 8)]:
        s = df[col].astype(str)
        keep = set(s.value_counts().head(k).index)
        s = s.where(s.isin(keep), other="__other__")
        frames.append(pd.get_dummies(s, prefix=col))
    return pd.concat(frames, axis=1).astype(float)


def build_matrix(df: pd.DataFrame, include_categorical: bool) -> pd.DataFrame:
    beh = behavioral_matrix(df)
    if include_categorical:
        return pd.concat([beh.reset_index(drop=True),
                          categorical_onehot(df).reset_index(drop=True)], axis=1)
    return beh
