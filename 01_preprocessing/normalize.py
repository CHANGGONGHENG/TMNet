"""Apply the fixed mediastinal window to CT images.

The manuscript uses a window width of 400 HU and a window level of 40 HU; intensities are clipped to
[level - width/2, level + width/2] and rescaled to [0, 1]. The same pipeline is applied to the pre-
and post-nICT images.

Example
-------
python normalize.py --input nifti/DEMO001.nii.gz --output normalized/DEMO001.nii.gz --ww 400 --wl 40
python normalize.py --manifest series_manifest.csv --input-dir nifti --output normalized
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", help="input NIfTI file")
    p.add_argument("--output", required=True, help="output NIfTI file, or directory with --manifest")
    p.add_argument("--manifest", default=None, help="csv with an ID column; normalises ID.nii.gz")
    p.add_argument("--input-dir", default=None, help="directory holding the resampled images")
    p.add_argument("--ww", type=float, default=400.0, help="window width in HU")
    p.add_argument("--wl", type=float, default=40.0, help="window level in HU")
    p.add_argument("--suffix", default=".nii.gz")
    return p.parse_args()


def window(image: sitk.Image, ww: float, wl: float) -> sitk.Image:
    arr = sitk.GetArrayFromImage(image).astype(np.float32)
    low, high = wl - ww / 2.0, wl + ww / 2.0
    arr = np.clip(arr, low, high)
    arr = (arr - low) / (high - low)
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(image)
    return out


def main() -> int:
    args = parse_args()
    if args.manifest:
        if not args.input_dir:
            raise SystemExit("--input-dir is required together with --manifest.")
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(args.manifest)
        n = 0
        for case_id in df["ID"]:
            src = Path(args.input_dir) / f"{case_id}{args.suffix}"
            if not src.exists():
                print(f"[skip] {case_id}: {src} not found")
                continue
            sitk.WriteImage(window(sitk.ReadImage(str(src)), args.ww, args.wl),
                            str(out_dir / f"{case_id}{args.suffix}"))
            n += 1
        print(f"normalised {n} images (ww={args.ww}, wl={args.wl}) to {out_dir}")
        return 0

    if not args.input:
        raise SystemExit("Provide --input or --manifest.")
    image = window(sitk.ReadImage(args.input), args.ww, args.wl)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(out))
    print(f"wrote {out}  window width={args.ww} HU, level={args.wl} HU")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
