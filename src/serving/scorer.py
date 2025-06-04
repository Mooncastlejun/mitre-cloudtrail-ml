"""Online scorer that wraps the trained detector for streaming use.

Two calibration notes that matter for streaming:

- Behavioral features are relative to recent activity, so the scorer keeps a
  rolling buffer of recent events; each incoming event is appended, the buffer
  re-featurized, and the newest event's anomaly score returned.
- A model threshold fit on the full training set doesn't transfer to a small
  live window (train/serve skew in count features). So for the *stream* we flag
  relatively: an event is a threat if its score sits in the top decile of the
  recent score history. This matches the goal — surface the few real threats,
  not label ~40% of traffic. (Offline PR-AUC/F1 in `src.pipeline` still uses the
  fixed threshold for evaluation.)
"""
from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd

from ..data.features import add_behavioral_features
from ..data.parse import flatten_event
from ..labeling.mitre_attack import label_events
from ..models.registry import Bundle, load


class OnlineScorer:
    def __init__(self, bundle: Bundle | None = None, window: int = 800,
                 alert_quantile: float = 0.90, warmup: int = 60):
        self.bundle = bundle or load()
        self.buffer: deque[dict] = deque(maxlen=window)
        self.scores: deque[float] = deque(maxlen=window)
        self.alert_quantile = alert_quantile
        self.warmup = warmup

    def score_event(self, raw_event: dict) -> dict:
        row = flatten_event(raw_event)
        self.buffer.append(row)
        df = pd.DataFrame(list(self.buffer))
        df["eventTime"] = pd.to_datetime(df["eventTime"], errors="coerce", utc=True)
        df = label_events(df)
        df = add_behavioral_features(df)

        score = float(self.bundle.model.score(df)[-1])
        self.scores.append(score)

        if len(self.scores) < self.warmup:
            is_attack = False
            dyn_thr = 1.0
        else:
            dyn_thr = float(np.quantile(self.scores, self.alert_quantile))
            is_attack = score >= dyn_thr

        latest = df.iloc[-1]
        return {
            "eventName": latest.get("eventName"),
            "identity": latest.get("identity_username"),
            "sourceIP": latest.get("sourceIPAddress"),
            "mitre_tactic": latest.get("mitre_tactic"),
            "anomaly_score": round(score, 4),
            "alert_threshold": round(dyn_thr, 4),
            "is_attack": bool(is_attack),
            "weak_label": int(latest.get("label", 0)),
            "model": self.bundle.model_name,
        }
