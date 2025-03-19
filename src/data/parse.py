"""Parse raw AWS CloudTrail logs into a flat pandas DataFrame.

CloudTrail ships events either as JSONL (one event per line) or as the AWS
``{"Records": [...]}`` envelope (one or many files). This flattens the nested
``userIdentity`` and pulls the fields that matter for anomaly detection.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd

# fields we lift out of each raw event
_FIELDS = [
    "eventTime", "eventName", "eventSource", "awsRegion", "sourceIPAddress",
    "userAgent", "errorCode", "errorMessage", "readOnly", "eventType",
    "managementEvent", "recipientAccountId",
]


def _open(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def _iter_raw_events(path: Path) -> Iterator[dict]:
    with _open(path) as fh:
        text = fh.read().strip()
    if not text:
        return
    # JSONL?
    if "\n" in text and not text.lstrip().startswith("{\n") and text.lstrip()[0] != "[":
        for line in text.splitlines():
            line = line.strip()
            if line:
                obj = json.loads(line)
                if "Records" in obj:
                    yield from obj["Records"]
                else:
                    yield obj
        return
    # single JSON doc: either an envelope or a list
    obj = json.loads(text)
    if isinstance(obj, list):
        yield from obj
    elif "Records" in obj:
        yield from obj["Records"]
    else:
        yield obj


def flatten_event(ev: dict) -> dict:
    ident = ev.get("userIdentity", {}) or {}
    row = {f: ev.get(f) for f in _FIELDS}
    row["identity_type"] = ident.get("type")
    row["identity_arn"] = ident.get("arn")
    row["identity_account"] = ident.get("accountId")
    row["identity_principal"] = ident.get("principalId")
    row["identity_username"] = ident.get("userName") or _arn_name(ident.get("arn"))
    row["access_key_id"] = ident.get("accessKeyId")
    row["has_error"] = 1 if ev.get("errorCode") else 0
    # keep a compact text view for embedding models
    row["event_text"] = _event_text(ev)
    return row


def _arn_name(arn: str | None) -> str | None:
    if not arn:
        return None
    return arn.rsplit("/", 1)[-1].rsplit(":", 1)[-1]


def _event_text(ev: dict) -> str:
    parts = [
        ev.get("eventName", ""),
        (ev.get("eventSource", "") or "").replace(".amazonaws.com", ""),
        ev.get("awsRegion", ""),
        (ev.get("userIdentity", {}) or {}).get("type", ""),
        ev.get("errorCode", "") or "ok",
    ]
    return " ".join(str(p) for p in parts if p)


def load_events(paths: Iterable[Path]) -> pd.DataFrame:
    rows = []
    for p in paths:
        for ev in _iter_raw_events(Path(p)):
            rows.append(flatten_event(ev))
    df = pd.DataFrame(rows)
    if "eventTime" in df.columns:
        df["eventTime"] = pd.to_datetime(df["eventTime"], errors="coerce", utc=True)
    return df


def load_file(path: str | Path) -> pd.DataFrame:
    return load_events([Path(path)])
