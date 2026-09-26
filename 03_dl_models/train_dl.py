"""Train a 2.5D deep-learning network on the pre- and post-nICT CT images.

The best checkpoint is written to the output directory in the local filesystem only; model weights
are never committed to the repository. Use ``--smoke-test`` to run a single tiny epoch on randomly
generated volumes, which verifies that the installation works without any image data.

Example
-------
python train_dl.py --manifest manifest.csv --mask-dir masks/ --images images/ \
                   --config dl_config.yaml --output runs/tmnet_2p5d

python train_dl.py --smoke-test --config dl_config.yaml --output runs/smoke
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset, random_split

from models_2p5d import build_model

try:
    from datasets import TwoPointFiveDDataset, make_2p5d
except ImportError:  # pragma: no cover
    TwoPointFiveDDataset = None
    make_2p5d = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SyntheticDataset(Dataset):
    """Random volumes with a weak signal, used by ``--smoke-test``."""

    def __init__(self, n: int = 16, size: int = 64, combine_timepoints: bool = False) -> None:
        rng = np.random.default_rng(0)
        self.n = n
        self.size = size
        channels = 6 if combine_timepoints else 3
        self.x = rng.normal(size=(n, channels, size, size)).astype("float32")
        self.z = rng.normal(size=n)
        self.y = (rng.random(n) < 1 / (1 + np.exp(-self.z))).astype("float32")
        self.x[:, 0] += self.z[:, None, None]

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        return f"SIM{i:03d}", torch.from_numpy(self.x[i]), torch.tensor(self.y[i])


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, np.ndarray, np.ndarray]:
    from sklearn.metrics import roc_auc_score

    model.eval()
    losses, probs, ys = [], [], []
    criterion = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for _, x, y in loader:
            x, y = x.to(device), y.to(device)
            logit = model(x)
            losses.append(float(criterion(logit, y)))
            probs.append(torch.sigmoid(logit).cpu().numpy())
            ys.append(y.cpu().numpy())
    prob = np.concatenate(probs) if probs else np.array([])
    y_true = np.concatenate(ys) if ys else np.array([])
    auc = float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) > 1 else float("nan")
    return float(np.mean(losses)) if losses else float("nan"), auc, prob


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", help="csv with columns ID, pre_path, post_path and the label")
    p.add_argument("--mask-dir", default=None)
    p.add_argument("--config", default=str(here / "dl_config.yaml"))
    p.add_argument("--output", required=True)
    p.add_argument("--model", default=None, help="override the model name in the config")
    p.add_argument("--label-column", default="pCR")
    p.add_argument("--pretrained", action="store_true", help="initialise from ImageNet weights")
    p.add_argument("--smoke-test", action="store_true",
                   help="run a single tiny epoch on synthetic data (no image files needed)")
    p.add_argument("--device", default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_name = args.model or cfg.get("model", "smallcnn")
    combine = bool(cfg.get("combine_timepoints", False))
    in_channels = 6 if combine else 3

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke_test:
        from models_2p5d import input_size_for

        smoke_size = input_size_for(model_name, int(cfg.get("smoke_size", 224)))
        dataset = SyntheticDataset(n=cfg.get("smoke_n", 16), size=smoke_size,
                                   combine_timepoints=combine)
        train_set, val_set = random_split(dataset, [12, len(dataset) - 12],
                                          generator=torch.Generator().manual_seed(seed))
    else:
        if not args.manifest:
            raise SystemExit("--manifest is required unless --smoke-test is used.")
        if TwoPointFiveDDataset is None:
            raise SystemExit("datasets.py could not be imported.")
        records = _records_from_manifest(Path(args.manifest), args.label_column)
        full = TwoPointFiveDDataset(records, mask_dir=args.mask_dir, combine_timepoints=combine)
        n_val = max(1, int(round(len(full) * cfg.get("val_fraction", 0.2))))
        train_set, val_set = random_split(full, [len(full) - n_val, n_val],
                                          generator=torch.Generator().manual_seed(seed))

    batch_size = int(cfg.get("batch_size", 8))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_model(model_name, in_channels=in_channels, pretrained=args.pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 1e-4)),
                                  weight_decay=float(cfg.get("weight_decay", 1e-4)))
    criterion = nn.BCEWithLogitsLoss()

    epochs = int(cfg.get("epochs", 50))
    patience = int(cfg.get("early_stopping_patience", 10))
    best_auc, best_epoch, bad_epochs = -np.inf, -1, 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        n_batches = 0
        for _, x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running += float(loss)
            n_batches += 1
        val_loss, val_auc, _ = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": running / max(n_batches, 1),
                        "val_loss": val_loss, "val_auc": val_auc,
                        "seconds": round(time.time() - t0, 2)})
        print(f"epoch {epoch:03d}  train_loss={running / max(n_batches, 1):.4f}  "
              f"val_loss={val_loss:.4f}  val_auc={val_auc:.3f}")

        if val_auc > best_auc + 1e-4:
            best_auc, best_epoch, bad_epochs = val_auc, epoch, 0
            torch.save({"model": model.state_dict(), "model_name": model_name,
                        "in_channels": in_channels, "epoch": epoch, "val_auc": val_auc},
                       out_dir / "best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"early stopping at epoch {epoch}")
                break

    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    (out_dir / "run_config.json").write_text(
        json.dumps({"model": model_name, "seed": seed, "device": str(device),
                    "best_epoch": best_epoch, "best_val_auc": best_auc,
                    "smoke_test": bool(args.smoke_test)}, indent=2), encoding="utf-8")
    print(f"best val AUC = {best_auc:.3f} at epoch {best_epoch}; checkpoint: {out_dir / 'best.pt'}")
    return 0


def _records_from_manifest(path: Path, label_column: str):
    df = pd.read_csv(path)
    base = path.parent
    records = []
    for _, row in df.iterrows():
        pre = Path(str(row["pre_path"]))
        post = Path(str(row["post_path"]))
        pre = pre if pre.is_absolute() else base / pre
        post = post if post.is_absolute() else base / post
        label = float(row[label_column]) if label_column in df.columns else None
        records.append((str(row["ID"]), pre, post, label))
    return records


if __name__ == "__main__":
    raise SystemExit(main())
