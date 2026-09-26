"""R backend for classifiers that are only available in R.

Currently used for the conditional inference tree (`partykit::ctree`), which has no scikit-learn
equivalent. The feature matrices are written to csv, `r_models.R` is executed with `Rscript` and the
predicted probabilities are read back.

`Rscript` must be on the PATH, or its full path passed with ``--rscript``.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

R_MODELS = {"ConditionalInferenceTree": "ConditionalInferenceTree"}

HERE = Path(__file__).resolve().parent


def find_rscript(path: str | None = None) -> str | None:
    """Locate Rscript (explicit path, then PATH)."""
    if path:
        return path if Path(path).exists() else None
    return shutil.which("Rscript") or shutil.which("Rscript.exe")


def is_available(path: str | None = None) -> bool:
    return find_rscript(path) is not None


def fit_predict(
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_predict_list: list[np.ndarray],
    feature_names: list[str],
    workdir: Path,
    rscript: str | None = None,
) -> list[np.ndarray]:
    """Train an R model and return P(y = 1) for every prediction matrix."""
    exe = find_rscript(rscript)
    if exe is None:
        raise RuntimeError("Rscript was not found; install R or pass --rscript.")

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    out_dir = workdir / "r_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_file = workdir / "train.csv"
    train_df = pd.DataFrame(x_train, columns=feature_names)
    train_df["y"] = np.asarray(y_train).astype(int)
    train_df.to_csv(train_file, index=False)

    pred_files = []
    for i, x in enumerate(x_predict_list):
        path = workdir / f"predict_{i}.csv"
        pd.DataFrame(x, columns=feature_names).to_csv(path, index=False)
        pred_files.append(str(path))

    cmd = [exe, str(HERE / "r_models.R"), model_name, str(train_file), str(out_dir)] + pred_files
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"R model failed: {proc.stderr.strip()[:400]}")

    results = []
    for i in range(len(x_predict_list)):
        out = out_dir / f"pred_{i}.csv"
        if not out.exists():
            raise RuntimeError(f"R model did not produce {out}")
        results.append(pd.read_csv(out)["p"].to_numpy(dtype=float))
    return results
