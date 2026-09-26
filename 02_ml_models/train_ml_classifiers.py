"""Train and evaluate the conventional radiomics machine-learning classifiers.

Each classifier in ``ml_config.yaml`` is fitted on the training cohort and evaluated on the
internal validation cohort and the external test cohorts. Feature standardisation is fitted on the
training cohort only. Trained estimators are written to the output directory (they are never
committed to the repository).

Example
-------
python train_ml_classifiers.py --features ../data/example/features_demo.csv \
                               --labels ../data/example/labels_demo.csv \
                               --config ml_config.yaml \
                               --output ../data/expected_output/ml_demo
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn import (
    discriminant_analysis,
    ensemble,
    gaussian_process,
    linear_model,
    naive_bayes,
    neighbors,
    neural_network,
    preprocessing,
    svm,
    tree,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (  # noqa: E402
    classification_metrics,
    cohort_split,
    feature_columns,
    load_features,
    load_labels,
    merge_tables,
)

SKLEARN_MODULES = (
    linear_model,
    svm,
    neighbors,
    naive_bayes,
    discriminant_analysis,
    tree,
    ensemble,
    neural_network,
    gaussian_process,
)


def resolve_estimator(name: str):
    """Resolve an estimator by class name, including the local and optional implementations."""
    if name in (None, "null"):
        return None
    # classifiers implemented in this repository
    try:
        import boosting

        if hasattr(boosting, name):
            return getattr(boosting, name)
    except ImportError:
        pass
    optional = {
        "XGBClassifier": "xgboost",
        "LGBMClassifier": "lightgbm",
        "CatBoostClassifier": "catboost",
    }
    if name in optional:
        try:
            module = __import__(optional[name])
            return getattr(module, name)
        except ImportError:
            return None
    for module in SKLEARN_MODULES:
        if hasattr(module, name):
            return getattr(module, name)
    return None


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", required=True, help="feature table (csv)")
    p.add_argument("--labels", required=True, help="label / cohort table (csv)")
    p.add_argument("--config", default=str(here / "ml_config.yaml"))
    p.add_argument("--output", required=True, help="output directory")
    p.add_argument("--seed", type=int, default=None, help="override the seed in the config")
    p.add_argument("--models", default=None,
                   help="comma-separated subset of classifier names (default: all enabled)")
    p.add_argument("--java-home", default=None,
                   help="path to the Java runtime used by the WEKA backend (or set JAVA_HOME)")
    p.add_argument("--weka-home", default=None,
                   help="directory holding the installed WEKA packages (or set WEKA_HOME)")
    p.add_argument("--rscript", default=None,
                   help="path to Rscript used by the R backend (or leave on the PATH)")
    p.add_argument("--feature-prefix", default=None,
                   help="comma-separated column prefixes to use as features, e.g. PRE_RF_ "
                        "(default: every column of the feature table)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output)
    (out_dir / "models").mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 42))
    label_col = cfg.get("label_column", "pCR")
    cohort_col = cfg.get("cohort_column", "cohort")

    features = load_features(args.features)
    labels = load_labels(args.labels)
    df = merge_tables(features, labels)

    # only the columns of the feature table are used as predictors (never label / cohort /
    # survival columns from the label table)
    cols = [c for c in features.columns if c != "ID"]
    if args.feature_prefix:
        prefixes = tuple(p.strip() for p in args.feature_prefix.split(",") if p.strip())
        cols = [c for c in cols if c.startswith(prefixes)]
        if not cols:
            raise SystemExit(f"No feature columns match {prefixes}")
    cohorts = cohort_split(df, cohort_col=cohort_col)
    if "train" not in cohorts:
        raise SystemExit("The label file must contain a 'train' cohort.")

    train = cohorts["train"]
    scaler = None
    if cfg.get("standardize", True):
        scaler = preprocessing.StandardScaler().fit(train[cols])
        df[cols] = scaler.transform(df[cols])

    requested = set(args.models.split(",")) if args.models else None
    rows = []
    preds = []
    n_fitted = 0

    for entry in cfg.get("models", cfg.get("classifiers", [])):
        name = entry["name"]
        if not entry.get("enabled", True):
            continue
        if requested is not None and name not in requested:
            continue

        cohort_order = list(cohorts.items())
        t0 = time.time()

        if entry.get("backend") == "r":
            import r_backend

            r_name = entry.get("r_model") or r_backend.R_MODELS.get(name)
            if r_name is None:
                print(f"[skip] {name}: no R model configured")
                continue
            try:
                prob_list = r_backend.fit_predict(
                    model_name=r_name,
                    x_train=train[cols].to_numpy(dtype=float),
                    y_train=train[label_col].to_numpy(dtype=int),
                    x_predict_list=[sub[cols].to_numpy(dtype=float) for _, sub in cohort_order],
                    feature_names=cols,
                    workdir=out_dir / "r_tmp" / name,
                    rscript=args.rscript,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[skip] {name}: R backend failed ({type(exc).__name__}: {exc})")
                continue
            fit_seconds = time.time() - t0
            n_fitted += 1
            for (cohort_name, sub), prob in zip(cohort_order, prob_list):
                rows.append(
                    {
                        "model": name,
                        "cohort": cohort_name,
                        "backend": "r",
                        "fit_seconds": round(fit_seconds, 3),
                        **classification_metrics(sub[label_col].to_numpy(), prob),
                    }
                )
                preds.extend(
                    {
                        "model": name,
                        "cohort": cohort_name,
                        "ID": pid,
                        "y_true": int(y),
                        "y_prob": float(p),
                    }
                    for pid, y, p in zip(sub["ID"], sub[label_col], prob)
                )
            continue

        if entry.get("backend") == "weka":
            import weka_backend

            classname = entry.get("classname") or weka_backend.WEKA_CLASSIFIERS.get(name)
            if classname is None:
                print(f"[skip] {name}: no WEKA class name configured")
                continue
            try:
                weka_backend.start_jvm(java_home=args.java_home, weka_home=args.weka_home)
                prob_list = weka_backend.fit_predict(
                    classname=classname,
                    options=list(entry.get("options") or []),
                    x_train=train[cols].to_numpy(dtype=float),
                    y_train=train[label_col].to_numpy(dtype=int),
                    x_predict_list=[sub[cols].to_numpy(dtype=float) for _, sub in cohort_order],
                    feature_names=cols,
                    workdir=out_dir / "weka_tmp" / name,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[skip] {name}: WEKA backend failed ({type(exc).__name__}: {exc})")
                continue
            fit_seconds = time.time() - t0
            n_fitted += 1
            for (cohort_name, sub), prob in zip(cohort_order, prob_list):
                rows.append(
                    {
                        "model": name,
                        "cohort": cohort_name,
                        "backend": "weka",
                        "fit_seconds": round(fit_seconds, 3),
                        **classification_metrics(sub[label_col].to_numpy(), prob),
                    }
                )
                preds.extend(
                    {
                        "model": name,
                        "cohort": cohort_name,
                        "ID": pid,
                        "y_true": int(y),
                        "y_prob": float(p),
                    }
                    for pid, y, p in zip(sub["ID"], sub[label_col], prob)
                )
            continue

        cls = resolve_estimator(entry["estimator"])
        if cls is None:
            print(f"[skip] {name}: estimator {entry['estimator']} is not available")
            continue

        params = dict(entry.get("params") or {})
        # 'log_loss' replaced 'log' as the SGDClassifier loss name in scikit-learn 1.1;
        # map it back so that the same configuration runs on the older environment used here.
        if entry["estimator"] == "SGDClassifier" and params.get("loss") == "log_loss":
            import sklearn

            if tuple(int(x) for x in sklearn.__version__.split(".")[:2]) < (1, 1):
                params["loss"] = "log"
        if "random_state" in cls().get_params():
            params["random_state"] = seed
        model = cls(**params)

        try:
            model.fit(train[cols], train[label_col])
        except Exception as exc:  # noqa: BLE001 - a classifier may not accept the feature scale
            print(f"[skip] {name}: fitting failed ({type(exc).__name__}: {exc})")
            continue
        fit_seconds = time.time() - t0
        n_fitted += 1

        joblib.dump(model, out_dir / "models" / f"{name}.joblib")

        for cohort_name, sub in cohorts.items():
            prob = predict_probability(model, sub[cols])
            rows.append(
                {
                    "model": name,
                    "cohort": cohort_name,
                    "backend": "sklearn",
                    "fit_seconds": round(fit_seconds, 3),
                    **classification_metrics(sub[label_col].to_numpy(), prob),
                }
            )
            preds.extend(
                {
                    "model": name,
                    "cohort": cohort_name,
                    "ID": pid,
                    "y_true": int(y),
                    "y_prob": float(p),
                }
                for pid, y, p in zip(sub["ID"], sub[label_col], prob)
            )

    if n_fitted == 0:
        raise SystemExit("No classifier could be fitted.")

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    pd.DataFrame(preds).to_csv(out_dir / "predictions.csv", index=False)

    if scaler is not None:
        joblib.dump(scaler, out_dir / "models" / "scaler.joblib")

    summary = {
        "n_classifiers_fitted": n_fitted,
        "n_features": len(cols),
        "n_cases": int(len(df)),
        "cohorts": {k: int(len(v)) for k, v in cohorts.items()},
        "seed": seed,
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"fitted {n_fitted} classifiers on {len(cols)} features / {len(train)} training cases")
    print("best AUC per cohort:")
    for cohort_name in cohorts:
        sub = metrics[metrics["cohort"] == cohort_name].sort_values("AUC", ascending=False)
        if not sub.empty:
            best = sub.iloc[0]
            print(f"  {cohort_name:>6}: {best['model']}  AUC={best['AUC']:.3f}")
    print(f"wrote {out_dir / 'metrics.csv'}")
    print(f"wrote {out_dir / 'predictions.csv'}")
    return 0


def predict_probability(model, x: pd.DataFrame) -> np.ndarray:
    """Return the probability of the positive class for any scikit-learn estimator."""
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(x)
        return np.asarray(prob)[:, 1] if np.ndim(prob) == 2 else np.asarray(prob, dtype=float)
    if hasattr(model, "decision_function"):
        score = np.asarray(model.decision_function(x), dtype=float)
        return 1.0 / (1.0 + np.exp(-np.clip(score, -50.0, 50.0)))
    return np.asarray(model.predict(x), dtype=float)


if __name__ == "__main__":
    raise SystemExit(main())

