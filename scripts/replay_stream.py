#!/usr/bin/env python3
"""Replay the CloudTrail sample as a live stream against the scorer service.

Simulates real-time ingestion: reads the sample JSONL and POSTs each event to
the running FastAPI scorer, printing flagged attacks as they arrive.

    uvicorn src.serving.app:app &          # start the service
    python scripts/replay_stream.py        # stream events at ~20/s
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "sample" / "cloudtrail_sample.jsonl"
BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
RATE = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0


def post(event: dict) -> dict:
    data = json.dumps(event).encode()
    req = urllib.request.Request(BASE + "/score", data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.load(r)


def main():
    events = [json.loads(l) for l in SAMPLE.read_text().splitlines() if l.strip()]
    print(f"streaming {len(events)} events to {BASE} at ~{RATE}/s")
    flagged = 0
    for i, ev in enumerate(events, 1):
        try:
            res = post(ev)
        except Exception as exc:
            print("post failed:", exc); time.sleep(1); continue
        if res.get("is_attack"):
            flagged += 1
            print(f"  🚨 {res['eventName']:<24} {res['mitre_tactic']:<20} "
                  f"score={res['anomaly_score']} ip={res['sourceIP']}")
        if i % 200 == 0:
            print(f"-- {i} events, {flagged} flagged")
        time.sleep(1.0 / RATE)


if __name__ == "__main__":
    main()
