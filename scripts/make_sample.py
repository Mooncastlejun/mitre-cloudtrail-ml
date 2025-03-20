#!/usr/bin/env python3
"""Generate a small, realistic AWS CloudTrail sample (JSONL) for local runs.

The real project trains on the flaws.cloud CloudTrail dataset (~1.9M events,
Kaggle). That file is far too large to ship, so this synthesizes a structurally
faithful sample — normal automation/read traffic plus multi-step attack chains
(recon -> cred access -> persistence -> priv-esc -> defense evasion ->
exfiltration/impact) — letting anyone reproduce the full pipeline end to end.

Usage: python scripts/make_sample.py [n_events]
"""
from __future__ import annotations

import json
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "sample" / "cloudtrail_sample.jsonl"
random.seed(1337)

REGIONS = ["us-east-1", "us-west-2", "eu-west-1", "ap-northeast-2"]

NORMAL_EVENTS = [
    ("DescribeInstances", "ec2.amazonaws.com", True),
    ("ListBuckets", "s3.amazonaws.com", True),
    ("GetObject", "s3.amazonaws.com", True),
    ("DescribeVolumes", "ec2.amazonaws.com", True),
    ("ListMetrics", "monitoring.amazonaws.com", True),
    ("DescribeLogGroups", "logs.amazonaws.com", True),
    ("GetCallerIdentity", "sts.amazonaws.com", True),
    ("DescribeSecurityGroups", "ec2.amazonaws.com", True),
    ("HeadObject", "s3.amazonaws.com", True),
    ("DescribeDBInstances", "rds.amazonaws.com", True),
]

# attack chains: (eventName, source, readOnly, p_error)
ATTACK_STEPS = [
    ("GetCallerIdentity", "sts.amazonaws.com", True, 0.0),
    ("ListUsers", "iam.amazonaws.com", True, 0.1),
    ("ListRoles", "iam.amazonaws.com", True, 0.1),
    ("CreateAccessKey", "iam.amazonaws.com", False, 0.2),
    ("CreateUser", "iam.amazonaws.com", False, 0.2),
    ("AttachUserPolicy", "iam.amazonaws.com", False, 0.35),
    ("PutUserPolicy", "iam.amazonaws.com", False, 0.3),
    ("CreateLoginProfile", "iam.amazonaws.com", False, 0.3),
    ("StopLogging", "cloudtrail.amazonaws.com", False, 0.15),
    ("DeleteTrail", "cloudtrail.amazonaws.com", False, 0.2),
    ("GetObject", "s3.amazonaws.com", True, 0.1),
    ("CopyObject", "s3.amazonaws.com", False, 0.15),
    ("CreateSnapshot", "ec2.amazonaws.com", False, 0.2),
    ("ModifySnapshotAttribute", "ec2.amazonaws.com", False, 0.25),
    ("TerminateInstances", "ec2.amazonaws.com", False, 0.2),
    ("DeleteBucket", "s3.amazonaws.com", False, 0.3),
]

NORMAL_PRINCIPALS = [f"AIDAAUTO{i:04d}" for i in range(8)]
NORMAL_IPS = [f"10.0.{i}.{random.randint(2, 250)}" for i in range(6)]
ATTACK_IPS = ["45.79.13.7", "185.220.101.42", "104.244.72.115"]


def _event(name, source, readonly, principal, ip, ts, error=False, itype="IAMUser"):
    ev = {
        "eventVersion": "1.08",
        "eventTime": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "eventName": name,
        "eventSource": source,
        "awsRegion": random.choice(REGIONS),
        "sourceIPAddress": ip,
        "userAgent": "aws-cli/2.13.0" if not error else "python-requests/2.31",
        "readOnly": readonly,
        "eventType": "AwsApiCall",
        "managementEvent": True,
        "recipientAccountId": "811596193553",
        "userIdentity": {
            "type": itype,
            "principalId": principal,
            "arn": f"arn:aws:iam::811596193553:user/{principal}",
            "accountId": "811596193553",
            "accessKeyId": "ASIA" + principal[-8:],
            "userName": principal,
        },
    }
    if error:
        ev["errorCode"] = random.choice(["AccessDenied", "UnauthorizedOperation", "Client.UnauthorizedOperation"])
        ev["errorMessage"] = "User is not authorized to perform this action"
    return ev


def generate(n: int):
    start = datetime(2025, 3, 1, tzinfo=timezone.utc)
    events = []
    # normal background traffic (rest = multi-step attack chains).
    # ~0.52 normal here lands near the real dataset's 57/43 after MITRE labeling
    # (some recon steps inside attack chains label as normal).
    n_normal = int(n * 0.52)
    for _ in range(n_normal):
        name, source, ro = random.choice(NORMAL_EVENTS)
        ts = start + timedelta(seconds=random.randint(0, 60 * 60 * 24 * 20))
        err = random.random() < 0.03
        events.append(_event(name, source, ro, random.choice(NORMAL_PRINCIPALS),
                             random.choice(NORMAL_IPS), ts, error=err,
                             itype=random.choice(["IAMUser", "AssumedRole"])))
    # attack chains
    remaining = n - n_normal
    while remaining > 0:
        principal = "AIDAATTACK" + str(random.randint(100, 999))
        ip = random.choice(ATTACK_IPS)
        t = start + timedelta(seconds=random.randint(0, 60 * 60 * 24 * 20))
        chain_len = min(remaining, random.randint(4, len(ATTACK_STEPS)))
        for name, source, ro, p_err in ATTACK_STEPS[:chain_len]:
            t = t + timedelta(seconds=random.randint(1, 40))
            events.append(_event(name, source, ro, principal, ip, t,
                                 error=random.random() < p_err,
                                 itype="IAMUser"))
        remaining -= chain_len
    random.shuffle(events)
    return events


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
    events = generate(n)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as fh:
        for ev in events:
            fh.write(json.dumps(ev) + "\n")
    print(f"wrote {len(events)} events -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
