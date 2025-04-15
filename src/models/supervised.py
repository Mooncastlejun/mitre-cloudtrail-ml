"""Supervised baseline — and the diagnosis of why it fails to generalize.

A RandomForest on one-hot(eventName) scores ~perfectly in a random split
because the weak label is *defined* from eventName: the model just memorizes the
label-defining columns. To expose that, we also run a **leave-one-tactic-out**
evaluation: hold out every attack of one tactic from training. Recall on the
held-out tactic collapses — the model never learned "maliciousness", only which
API names were in the training labels. That motivates the unsupervised approach.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from ..config import RANDOM_SEED
from ..data.features import build_matrix
from ..labeling.mitre_attack import OFFENSIVE_TACTICS


@dataclass
class SupervisedResult:
    random_split: dict
    leave_one_tactic_out: dict
    verdict: str


def _fit_eval(X_tr, y_tr, X_te, y_te) -> dict:
    clf = RandomForestClassifier(n_estimators=120, max_depth=None,
                                 random_state=RANDOM_SEED, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_te)
    return {
        "precision": round(float(precision_score(y_te, pred, zero_division=0)), 3),
        "recall": round(float(recall_score(y_te, pred, zero_division=0)), 3),
        "f1": round(float(f1_score(y_te, pred, zero_division=0)), 3),
    }


def run_supervised(df: pd.DataFrame) -> SupervisedResult:
    y = df["label"].values
    X = build_matrix(df, include_categorical=True).values  # includes leaky eventName

    # 1) random split — looks great (leakage / memorization)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_SEED, stratify=y)
    random_split = _fit_eval(X_tr, y_tr, X_te, y_te)

    # 2) leave-one-tactic-out — generalization to *unseen attack types*
    loto = {}
    for tactic in sorted(OFFENSIVE_TACTICS):
        held = (df["mitre_tactic"] == tactic).values
        attacks_of_tactic = held & (y == 1)
        if attacks_of_tactic.sum() < 5:
            continue
        train_mask = ~attacks_of_tactic  # remove this tactic's attacks from training
        test_mask = held                 # test on all events of the held tactic
        res = _fit_eval(X[train_mask], y[train_mask], X[test_mask], y[test_mask])
        loto[tactic] = res

    macro_recall = round(float(np.mean([v["recall"] for v in loto.values()])), 3) if loto else 0.0
    verdict = (
        f"random-split F1={random_split['f1']} but leave-one-tactic-out mean "
        f"recall={macro_recall} — the classifier memorizes label-defining API "
        f"names and misses unseen attack tactics."
    )
    return SupervisedResult(random_split=random_split, leave_one_tactic_out=loto, verdict=verdict)
