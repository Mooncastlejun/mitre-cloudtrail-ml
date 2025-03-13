"""Project paths and shared constants."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SAMPLE_DIR = DATA_DIR / "sample"
RAW_DIR = DATA_DIR / "raw"
ARTIFACT_DIR = ROOT / "artifacts"
REPORT_DIR = ROOT / "reports"

SAMPLE_FILE = SAMPLE_DIR / "cloudtrail_sample.jsonl"

RANDOM_SEED = 42

# columns that, if fed to a supervised model, leak the label almost perfectly
# (the label is *derived* from them) — used to demonstrate memorization/leakage.
LEAKY_COLUMNS = ["eventName", "mitre_tactic"]

for _d in (DATA_DIR, SAMPLE_DIR, RAW_DIR, ARTIFACT_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)
