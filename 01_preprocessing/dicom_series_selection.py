"""Select the contrast-enhanced chest CT series from each study.

A study folder usually contains several series (scout, plain CT, arterial phase, venous phase,
lung window reconstructions). This script lists every series, reads a few DICOM tags and selects
the series that matches the expected description of the contrast-enhanced chest CT used in the
study. The selection is written to a manifest that the later steps consume, and can be reviewed and
edited by hand before continuing.

Example
-------
python dicom_series_selection.py --input /data/raw --output series_manifest.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import SimpleITK as sitk

# Series descriptions that identify a contrast-enhanced chest CT in the participating centres.
CONTRAST_KEYWORDS = ("arterial", "venous", "portal", "contrast", "enhanced", "ap", "vp")
EXCLUDE_KEYWORDS = ("scout", "topogram", "localizer", "plain", "non-contrast", "lung", "bone")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="root folder containing one subfolder per study")
    p.add_argument("--output", required=True, help="output manifest (csv)")
    p.add_argument("--pattern", default="*", help="glob used to find study folders")
    p.add_argument("--prefer", default="venous,arterial",
                   help="comma-separated series-description keywords in order of preference")
    return p.parse_args()


def series_description(directory: Path) -> str:
    try:
        reader = sitk.ImageSeriesReader()
        files = reader.GetGDCMSeriesFileNames(str(directory))
        if not files:
            return ""
        meta = sitk.ImageFileReader()
        meta.SetFileName(files[0])
        meta.LoadPrivateTagsOn()
        meta.ReadImageInformation()
        for key in ("0008|103e", "SeriesDescription"):
            if meta.HasMetaDataKey(key):
                return str(meta.GetMetaData(key))
    except Exception:  # noqa: BLE001 - unreadable series are reported as empty
        return ""
    return ""


def score(description: str, preferred: list[str]) -> int:
    d = description.lower()
    if any(k in d for k in EXCLUDE_KEYWORDS):
        return -1
    for rank, key in enumerate(preferred):
        if key and key in d:
            return 100 - rank
    return 10 if any(k in d for k in CONTRAST_KEYWORDS) else 0


def main() -> int:
    args = parse_args()
    preferred = [k.strip().lower() for k in args.prefer.split(",")]
    rows = []

    for study in sorted(Path(args.input).glob(args.pattern)):
        if not study.is_dir():
            continue
        candidates = []
        for series_dir in [study] + [p for p in study.iterdir() if p.is_dir()]:
            files = sitk.ImageSeriesReader().GetGDCMSeriesFileNames(str(series_dir))
            if not files:
                continue
            desc = series_description(series_dir)
            candidates.append((score(desc, preferred), len(files), desc, series_dir, files[0]))
        if not candidates:
            rows.append({"ID": study.name, "series_dir": "", "description": "",
                         "n_slices": 0, "selected": False, "note": "no readable DICOM series"})
            continue
        # prefer the contrast-enhanced description, then the largest series
        candidates.sort(key=lambda c: (c[0], c[1]), reverse=True)
        best = candidates[0]
        rows.append({"ID": study.name, "series_dir": str(best[3]), "description": best[2],
                     "n_slices": best[1], "selected": best[0] >= 0,
                     "note": "" if best[0] >= 0 else "no contrast-enhanced series identified"})

    out = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"wrote {args.output}: {len(out)} studies, {int(out['selected'].sum())} series selected")
    print("Please review the manifest and correct any mis-selected series before continuing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
