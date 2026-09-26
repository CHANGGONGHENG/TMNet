"""Shared helpers for the machine-learning models.

Loading of the feature / label tables, cohort handling and performance metrics.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
)

COHORTS = ["train", "val", "test1", "test2"]
N_BOOTSTRAP = 1000
SEED = 42


def load_features(path: str | Path) -> pd.DataFrame:
    """Load a feature table: first column is the case ID, all other columns are features."""
    df = pd.read_csv(path)
    if df.columns[0] != "ID":
        df = df.rename(columns={df.columns[0]: "ID"})
    return df


def load_labels(path: str | Path) -> pd.DataFrame:
    """Load the label table (ID, cohort, pCR, and optional survival columns)."""
    df = pd.read_csv(path)
    if df.columns[0] != "ID":
        df = df.rename(columns={df.columns[0]: "ID"})
    return df


def merge_tables(features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Inner-join features and labels on the case ID."""
    df = features.merge(labels, on="ID", how="inner")
    if df.empty:
        raise ValueError("No overlapping IDs between the feature table and the label file.")
    return df


def feature_columns(df: pd.DataFrame, label_col: str = "pCR") -> list[str]:
    """All numeric columns that are not identifiers, labels, cohorts or outcome columns."""
    excluded = {"ID", "cohort", "centre", "center", "group", "site", label_col}
    out = []
    for c in df.columns:
        if c in excluded:
            continue
        low = str(c).lower()
        if low.startswith(("os_", "dfs_", "pfs_", "rfs_")) or low.endswith(("_event", "_time", "_time_months")):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            out.append(c)
    return out


def cohort_split(df: pd.DataFrame, cohort_col: str = "cohort") -> dict[str, pd.DataFrame]:
    """Split the table into the cohorts used in the manuscript."""
    out = {}
    for name in COHORTS:
        sub = df[df[cohort_col] == name]
        if not sub.empty:
            out[name] = sub
    return out


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """AUC (with a bootstrap 95% CI) and threshold-0.5 classification metrics."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= 0.5).astype(int)

    res: dict[str, float] = {}
    if len(np.unique(y_true)) > 1:
        res["AUC"] = float(roc_auc_score(y_true, y_prob))
        res["AUC_95CI_low"], res["AUC_95CI_high"] = bootstrap_auc_ci(y_true, y_prob)
    else:
        res["AUC"] = float("nan")
        res["AUC_95CI_low"] = float("nan")
        res["AUC_95CI_high"] = float("nan")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    res["accuracy"] = float(accuracy_score(y_true, y_pred))
    res["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) else float("nan")
    res["specificity"] = float(tn / (tn + fp)) if (tn + fp) else float("nan")
    res["PPV"] = float(tp / (tp + fp)) if (tp + fp) else float("nan")
    res["NPV"] = float(tn / (tn + fn)) if (tn + fn) else float("nan")
    res["n"] = int(len(y_true))
    res["n_events"] = int(y_true.sum())
    return res


def bootstrap_auc_ci(
    y_true: np.ndarray, y_prob: np.ndarray, n_boot: int = N_BOOTSTRAP, seed: int = SEED
) -> tuple[float, float]:
    """Percentile bootstrap 95% CI of the AUC."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
    if not aucs:
        return float("nan"), float("nan")
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))
