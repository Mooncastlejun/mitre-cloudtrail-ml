"""MITRE ATT&CK tactic mapping + weak labeling for AWS CloudTrail events.

The flaws.cloud dataset isn't labeled, so we derive weak labels the way the
project did: map each CloudTrail API call to the ATT&CK tactic it most plausibly
belongs to, then mark events in offensive tactics (combined with risk signals
like errors / rare principals) as `attack`, the rest as `normal`.

This is intentionally a *rule* over eventName — which is precisely why a
supervised model trained on one-hot(eventName) memorizes the rule and fails to
generalize to novel attacker behavior. That limitation is the motivation for the
unsupervised approach.
"""
from __future__ import annotations

import re

import pandas as pd

# tactic -> list of eventName patterns (regex, matched case-insensitively)
TACTIC_PATTERNS: dict[str, list[str]] = {
    "discovery": [
        r"^List", r"^Describe", r"^Get(CallerIdentity|AccountAuthorizationDetails|BucketAcl|BucketPolicy)",
        r"^GetSessionToken", r"Enumerate", r"^Search",
    ],
    "credential_access": [
        r"CreateAccessKey", r"GetPasswordData", r"GetFederationToken",
        r"UpdateAccessKey", r"GetSecretValue", r"BatchGetSecretValue",
    ],
    "persistence": [
        r"CreateUser", r"CreateLoginProfile", r"CreateRole", r"CreateAccessKey$",
        r"CreateNetworkInterface",
    ],
    "privilege_escalation": [
        r"AttachUserPolicy", r"AttachRolePolicy", r"PutUserPolicy", r"PutRolePolicy",
        r"AddUserToGroup", r"CreatePolicyVersion", r"SetDefaultPolicyVersion",
        r"UpdateAssumeRolePolicy", r"PassRole",
    ],
    "defense_evasion": [
        r"StopLogging", r"DeleteTrail", r"UpdateTrail", r"DeleteFlowLogs",
        r"DeleteDetector", r"PutBucketAcl", r"DeleteConfigRule", r"DeleteLogGroup",
    ],
    "exfiltration": [
        r"GetObject", r"CopyObject", r"CreateSnapshot", r"ModifySnapshotAttribute",
        r"ModifyImageAttribute", r"SharedSnapshot", r"CreateDBSnapshot",
    ],
    "impact": [
        r"DeleteBucket", r"TerminateInstances", r"DeleteDBInstance", r"DeleteObject",
        r"StopInstances", r"DeleteVolume", r"PutBucketPolicy",
    ],
    "initial_access": [
        r"ConsoleLogin", r"AssumeRole", r"GetFederationToken",
    ],
}

# tactics considered offensive when combined with risk signals
OFFENSIVE_TACTICS = {
    "credential_access", "persistence", "privilege_escalation",
    "defense_evasion", "exfiltration", "impact",
}

_COMPILED = {t: [re.compile(p, re.I) for p in pats] for t, pats in TACTIC_PATTERNS.items()}


def map_tactic(event_name: str | None) -> str:
    if not event_name:
        return "none"
    for tactic, patterns in _COMPILED.items():
        if any(p.search(event_name) for p in patterns):
            return tactic
    return "none"


def label_events(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``mitre_tactic`` and binary ``label`` (1=attack, 0=normal)."""
    df = df.copy()
    df["mitre_tactic"] = df["eventName"].map(map_tactic)

    offensive = df["mitre_tactic"].isin(OFFENSIVE_TACTICS)
    # discovery is only suspicious in bulk / with errors (recon), not on its own
    noisy_recon = (df["mitre_tactic"] == "discovery") & (df.get("has_error", 0) == 1)
    # unauthorized / errored offensive calls are strong attack signals
    errored = df.get("has_error", 0) == 1

    df["label"] = ((offensive) | (offensive & errored) | noisy_recon).astype(int)
    return df


def label_summary(df: pd.DataFrame) -> dict:
    n = len(df)
    attack = int(df["label"].sum())
    by_tactic = df.groupby("mitre_tactic")["label"].agg(["count", "sum"]).to_dict("index")
    return {
        "events": n,
        "attack": attack,
        "normal": n - attack,
        "attack_ratio": round(attack / n, 3) if n else 0.0,
        "by_tactic": {k: {"count": int(v["count"]), "attack": int(v["sum"])}
                      for k, v in by_tactic.items()},
    }
