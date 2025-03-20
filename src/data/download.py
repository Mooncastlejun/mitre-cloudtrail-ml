"""Fetch the real flaws.cloud CloudTrail dataset from Kaggle.

Dataset: https://www.kaggle.com/datasets/nobukim/aws-cloudtrails-dataset-from-flaws-cloud
(~1.9M AWS CloudTrail events). Requires the Kaggle API + credentials
(`~/.kaggle/kaggle.json`).

    pip install kaggle
    python -m src.data.download

Falls back to printing manual instructions if the Kaggle CLI/credentials are
missing, so the repo stays reproducible via the synthetic sample.
"""
from __future__ import annotations

import subprocess
import sys

from ..config import RAW_DIR

DATASET = "nobukim/aws-cloudtrails-dataset-from-flaws-cloud"


def main() -> int:
    try:
        import kaggle  # noqa: F401
    except Exception:
        print("kaggle package not installed. Run: pip install kaggle")
        print(f"Then: kaggle datasets download -d {DATASET} -p {RAW_DIR} --unzip")
        return 1
    print(f"downloading {DATASET} -> {RAW_DIR}")
    rc = subprocess.call(
        ["kaggle", "datasets", "download", "-d", DATASET, "-p", str(RAW_DIR), "--unzip"]
    )
    if rc == 0:
        print("done. Point the pipeline at data/raw/ instead of the sample.")
    return rc


if __name__ == "__main__":
    sys.exit(main())
