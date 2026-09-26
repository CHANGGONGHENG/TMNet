"""Train every candidate 2.5D network listed in ``dl_config.yaml`` and summarise the comparison.

One training run is launched per candidate, into ``<output>/<candidate>/``. The validation and test
performance of all candidates is then collected into ``candidate_comparison.csv``.

Example
-------
python run_candidate_comparison.py --manifest manifest.csv --mask-dir masks/ \
                                   --config dl_config.yaml --output runs/comparison
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", default=None, help="required unless --smoke-test is used")
    p.add_argument("--mask-dir", default=None)
    p.add_argument("--config", default=str(HERE / "dl_config.yaml"))
    p.add_argument("--output", required=True)
    p.add_argument("--models", default=None, help="comma-separated subset of the candidates")
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--smoke-test", action="store_true",
                   help="run every candidate on synthetic data (fast check of the installation)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    candidates = cfg.get("candidates") or [cfg.get("model", "smallcnn")]
    if args.models:
        wanted = {m.strip() for m in args.models.split(",")}
        candidates = [c for c in candidates if c in wanted]

    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for name in candidates:
        run_dir = out_root / name
        cmd = [sys.executable, str(HERE / "train_dl.py"),
               "--config", args.config, "--output", str(run_dir), "--model", name]
        if args.smoke_test:
            cmd.append("--smoke-test")
        else:
            if not args.manifest:
                raise SystemExit("--manifest is required unless --smoke-test is used.")
            cmd += ["--manifest", args.manifest]
            if args.mask_dir:
                cmd += ["--mask-dir", args.mask_dir]
        if args.pretrained:
            cmd.append("--pretrained")

        print(f"=== {name} ===", flush=True)
        result = subprocess.run(cmd, check=False)
        summary_path = run_dir / "run_config.json"
        if result.returncode != 0 or not summary_path.exists():
            rows.append({"model": name, "status": f"failed ({result.returncode})"})
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rows.append({
            "model": name,
            "status": "ok",
            "best_epoch": summary.get("best_epoch"),
            "val_auc": summary.get("best_val_auc"),
            "device": summary.get("device"),
        })

    table = pd.DataFrame(rows)
    table.to_csv(out_root / "candidate_comparison.csv", index=False)
    print(f"\nwrote {out_root / 'candidate_comparison.csv'}")
    if "val_auc" in table:
        print(table.sort_values("val_auc", ascending=False).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
