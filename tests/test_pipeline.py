"""End-to-end smoke test on a tiny in-memory CloudTrail batch."""
import json

import pandas as pd

from src.data.features import add_behavioral_features, behavioral_matrix
from src.data.parse import flatten_event, load_file
from src.labeling.mitre_attack import label_events
from src.models.unsupervised import IsolationForestModel


def _raw(name, err=False):
    ev = {
        "eventTime": "2025-03-01T10:00:00Z", "eventName": name,
        "eventSource": "iam.amazonaws.com", "awsRegion": "us-east-1",
        "sourceIPAddress": "1.2.3.4", "readOnly": False,
        "userIdentity": {"type": "IAMUser", "principalId": "AIDA1",
                         "arn": "arn:aws:iam::1:user/bob", "userName": "bob"},
    }
    if err:
        ev["errorCode"] = "AccessDenied"
    return ev


def test_flatten_pulls_identity():
    row = flatten_event(_raw("CreateUser"))
    assert row["identity_username"] == "bob"
    assert row["eventName"] == "CreateUser"
    assert "event_text" in row


def test_parse_sample_file(tmp_path):
    p = tmp_path / "ct.jsonl"
    p.write_text("\n".join(json.dumps(_raw(n)) for n in ["ListBuckets", "CreateUser", "DeleteBucket"]))
    df = load_file(p)
    assert len(df) == 3
    assert df["eventTime"].notna().all()


def test_behavioral_features_and_iforest():
    rows = [flatten_event(_raw(n, err=(i % 3 == 0)))
            for i, n in enumerate(["ListBuckets", "CreateUser", "AttachUserPolicy",
                                    "DeleteBucket", "DescribeInstances"] * 20)]
    df = pd.DataFrame(rows)
    df["eventTime"] = pd.to_datetime(df["eventTime"], utc=True)
    df = add_behavioral_features(label_events(df))
    X = behavioral_matrix(df)
    assert X.shape[0] == len(df) and not X.isnull().any().any()

    model = IsolationForestModel(contamination=0.3).fit(df)
    scores = model.score(df)
    assert len(scores) == len(df)
    assert scores.min() >= 0 and scores.max() <= 1
