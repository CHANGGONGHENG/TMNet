"""Resample CT images to an isotropic voxel spacing.

Reads either a DICOM series directory or a NIfTI file and writes a NIfTI file resampled to the
requested isotropic spacing (1 x 1 x 1 mm3 in the manuscript).

Example
-------
python resample.py --input raw/DEMO001 --output nifti/DEMO001.nii.gz --spacing 1 1 1
python resample.py --manifest series_manifest.csv --output nifti/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import SimpleITK as sitk


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", help="DICOM series directory or NIfTI file")
    p.add_argument("--output", required=True, help="output NIfTI file, or a directory with --manifest")
    p.add_argument("--manifest", default=None,
                   help="csv with columns ID and series_dir; resamples every row")
    p.add_argument("--spacing", nargs=3, type=float, default=[1.0, 1.0, 1.0],
                   metavar=("SX", "SY", "SZ"))
    p.add_argument("--interpolator", default="linear", choices=["linear", "bspline", "nearest"])
    p.add_argument("--suffix", default=".nii.gz")
    return p.parse_args()


def read_image(path: str) -> sitk.Image:
    p = Path(path)
    if p.is_dir():
        reader = sitk.ImageSeriesReader()
        files = reader.GetGDCMSeriesFileNames(str(p))
        if not files:
            raise FileNotFoundError(f"No DICOM series found in {p}")
        reader.SetFileNames(files)
        return reader.Execute()
    return sitk.ReadImage(str(p))


INTERPOLATORS = {
    "linear": sitk.sitkLinear,
    "bspline": sitk.sitkBSpline,
    "nearest": sitk.sitkNearestNeighbor,
}


def resample(image: sitk.Image, spacing: list[float], interpolator: str) -> sitk.Image:
    original = image.GetSpacing()
    size = [int(round(image.GetSize()[i] * original[i] / spacing[i])) for i in range(3)]
    return sitk.Resample(
        image,
        size,
        sitk.Transform(),
        INTERPOLATORS[interpolator],
        image.GetOrigin(),
        spacing,
        image.GetDirection(),
        0.0,
        image.GetPixelID(),
    )


def main() -> int:
    args = parse_args()
    if args.manifest:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(args.manifest)
        done = 0
        for _, row in df.iterrows():
            if not str(row.get("series_dir", "")).strip():
                print(f"[skip] {row.get('ID')}: no series_dir")
                continue
            image = resample(read_image(str(row["series_dir"])), args.spacing, args.interpolator)
            target = out_dir / f"{row['ID']}{args.suffix}"
            sitk.WriteImage(image, str(target))
            done += 1
        print(f"resampled {done} images to {out_dir}")
        return 0

    if not args.input:
        raise SystemExit("Provide --input or --manifest.")
    image = resample(read_image(args.input), args.spacing, args.interpolator)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(out))
    print(f"wrote {out}  size={image.GetSize()}  spacing={image.GetSpacing()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
