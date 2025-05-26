"""End-to-end training pipeline.

    parse -> MITRE weak-label -> feature engineering
          -> supervised baseline (diagnose memorization / generalization failure)
          -> unsupervised models (IForest, LOF, Autoencoder, MiniLM+IForest)
          -> evaluate + compare -> persist the best detector

Run: `python -m src.pipeline [path_to_cloudtrail]`  (defaults to the sample)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from .config import REPORT_DIR, SAMPLE_FILE
from .data.features import add_behavioral_features
from .data.parse import load_file
from .labeling.mitre_attack import label_events, label_summary
from .models.autoencoder import AutoencoderModel
from .models.embed import EmbeddingIForestModel
from .models.evaluate import comparison_table, evaluate_scores
from .models.registry import Bundle, save
from .models.supervised import run_supervised
from .models.unsupervised import IsolationForestModel, LOFModel


def load_and_prepare(path: Path) -> pd.DataFrame:
    df = load_file(path)
    df = label_events(df)
    df = add_behavioral_features(df)
    return df


def run(path: Path | None = None, verbose: bool = True) -> dict:
    path = Path(path) if path else SAMPLE_FILE
    if not path.exists():
        raise SystemExit(f"no data at {path} — run `python scripts/make_sample.py` first")

    df = load_and_prepare(path)
    summary = label_summary(df)
    if verbose:
        print(f"loaded {summary['events']} events · "
              f"attack {summary['attack']} ({summary['attack_ratio']:.0%}) / "
              f"normal {summary['normal']}")

    # --- supervised baseline + failure diagnosis ---
    sup = run_supervised(df)
    if verbose:
        print("\n[supervised baseline]")
        print("  random split :", sup.random_split)
        print("  leave-one-tactic-out recall:",
              {k: v["recall"] for k, v in sup.leave_one_tactic_out.items()})
        print("  ->", sup.verdict)

    # --- unsupervised models (train unlabeled, eval on weak labels) ---
    y = df["label"].values
    models = [IsolationForestModel(), LOFModel(), AutoencoderModel(), EmbeddingIForestModel()]
    results, bundles = {}, {}
    for m in models:
        m.fit(df)
        scores = m.score(df)
        results[m.name] = evaluate_scores(y, scores)
        results[m.name]["backend"] = getattr(m, "backend", "sklearn")
        bundles[m.name] = (m, scores)

    if verbose:
        print("\n[unsupervised comparison]")
        print(comparison_table(results))

    # --- pick best by PR-AUC and persist ---
    best_name = max(results, key=lambda n: results[n]["pr_auc"])
    best_model, best_scores = bundles[best_name]
    threshold = results[best_name]["threshold"]
    save(Bundle(model=best_model, threshold=threshold, model_name=best_name,
                backend=results[best_name].get("backend", "sklearn")))

    report = {
        "data": summary,
        "supervised": {"random_split": sup.random_split,
                       "leave_one_tactic_out": sup.leave_one_tactic_out,
                       "verdict": sup.verdict},
        "unsupervised": results,
        "best_model": best_name,
    }
    (REPORT_DIR / "model_report.json").write_text(json.dumps(report, indent=2))
    if verbose:
        print(f"\nbest model: {best_name} (PR-AUC {results[best_name]['pr_auc']}) "
              f"-> saved detector + report to reports/model_report.json")
    return report


if __name__ == "__main__":
    run(Path(sys.argv[1]) if len(sys.argv) > 1 else None)
