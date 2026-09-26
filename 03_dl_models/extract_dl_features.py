"""Extract deep-learning features from a trained 2.5D network.

A forward hook is attached to the global average-pooling layer (or the layer named on the command
line) and the resulting activation vector is written for every case.

Example
-------
python extract_dl_features.py --manifest manifest.csv --checkpoint runs/tmnet_2p5d/best.pt \
                              --output features_dl.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from models_2p5d import build_model
from train_dl import _records_from_manifest

try:
    from datasets import TwoPointFiveDDataset
except ImportError:  # pragma: no cover
    TwoPointFiveDDataset = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--mask-dir", default=None)
    p.add_argument("--layer", default="avgpool",
                   help="feature layer name (e.g. avgpool, global_pool, head)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default=None)
    return p.parse_args()


class FeatureExtractor:
    def __init__(self, model: torch.nn.Module, layer: str) -> None:
        self.features: torch.Tensor | None = None
        self.handle = None
        for name, module in model.named_modules():
            if name == layer or name.endswith(f".{layer}"):
                self.handle = module.register_forward_hook(self._hook)
                break
        if self.handle is None:
            raise ValueError(f"Layer '{layer}' not found in the model.")

    def _hook(self, module, inputs, output) -> None:
        out = output[0] if isinstance(output, (tuple, list)) else output
        self.features = torch.flatten(out, start_dim=1).detach()

    def close(self) -> None:
        if self.handle is not None:
            self.handle.remove()


def main() -> int:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = build_model(ckpt["model_name"], in_channels=ckpt.get("in_channels", 3))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    extractor = FeatureExtractor(model, args.layer)

    if TwoPointFiveDDataset is None:
        raise SystemExit("datasets.py could not be imported.")
    records = _records_from_manifest(Path(args.manifest), "pCR")
    dataset = TwoPointFiveDDataset(records, mask_dir=args.mask_dir, combine_timepoints=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    ids, feats = [], []
    with torch.no_grad():
        for case_id, x, *_ in loader:
            model(x.to(device))
            if extractor.features is None:
                raise RuntimeError("No features were captured; check --layer.")
            ids.extend(list(case_id))
            feats.append(extractor.features.cpu().numpy())
    extractor.close()

    matrix = np.vstack(feats)
    columns = [f"DL_{i + 1:03d}" for i in range(matrix.shape[1])]
    out = pd.DataFrame(matrix, columns=columns)
    out.insert(0, "ID", ids)
    out.to_csv(args.output, index=False)
    print(f"wrote {args.output} ({out.shape[0]} cases x {matrix.shape[1]} features)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
