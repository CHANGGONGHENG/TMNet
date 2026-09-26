"""Apply previously trained classifiers to a feature table.

Example
-------
python predict_ml_classifiers.py --features ../data/example/features_demo.csv \
                                 --model-dir ../data/expected_output/ml_demo/models \
                                 --output ../data/expected_output/ml_demo/predictions_new.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import load_features  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--models", default=None,
                   help="comma-separated subset of model names (default: every *.joblib found)")
    return p.parse_args()


def predict_probability(model, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(x)
        return np.asarray(prob)[:, 1] if np.ndim(prob) == 2 else np.asarray(prob, dtype=float)
    if hasattr(model, "decision_function"):
        score = np.asarray(model.decision_function(x), dtype=float)
        return 1.0 / (1.0 + np.exp(-np.clip(score, -50.0, 50.0)))
    return np.asarray(model.predict(x), dtype=float)


def main() -> int:
    args = parse_args()
    model_dir = Path(args.model_dir)
    features = load_features(args.features)

    requested = set(args.models.split(",")) if args.models else None
    files = sorted(p for p in model_dir.glob("*.joblib") if p.stem != "scaler")
    if requested is not None:
        files = [p for p in files if p.stem in requested]
    if not files:
        raise SystemExit(f"No *.joblib models found in {model_dir}")

    scaler_path = model_dir / "scaler.joblib"
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None

    rows = []
    for path in files:
        model = joblib.load(path)
        cols = list(getattr(model, "feature_names_in_", []))
        if not cols:
            cols = [c for c in features.columns if c != "ID"]
        x = features[cols]
        if scaler is not None:
            x = pd.DataFrame(scaler.transform(x), columns=cols, index=features.index)
        prob = predict_probability(model, x)
        rows.extend(
            {"model": path.stem, "ID": pid, "y_prob": float(p)}
            for pid, p in zip(features["ID"], prob)
        )

    out = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"wrote {args.output} ({len(out)} rows, {len(files)} models)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
