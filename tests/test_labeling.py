import pandas as pd

from src.labeling.mitre_attack import label_events, map_tactic


def test_tactic_mapping():
    assert map_tactic("AttachUserPolicy") == "privilege_escalation"
    assert map_tactic("StopLogging") == "defense_evasion"
    assert map_tactic("ListBuckets") == "discovery"
    assert map_tactic("DescribeInstances") == "discovery"
    assert map_tactic("SomeRandomThing") == "none"


def test_label_events_marks_offensive_as_attack():
    df = pd.DataFrame({
        "eventName": ["DescribeInstances", "AttachUserPolicy", "DeleteBucket", "ListBuckets"],
        "has_error": [0, 0, 1, 0],
    })
    out = label_events(df)
    labels = dict(zip(out["eventName"], out["label"]))
    assert labels["AttachUserPolicy"] == 1
    assert labels["DeleteBucket"] == 1
    assert labels["DescribeInstances"] == 0
