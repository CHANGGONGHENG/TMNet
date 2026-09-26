"""2.5D dataset construction for the deep-learning models.

A 2.5D input is built from a single time point: the axial slice through the largest tumour
cross-section, together with the corresponding sagittal and coronal slices, stacked as three
channels. The pre- and post-nICT time points are handled as separate inputs (and can additionally
be stacked into a six-channel input). All views are resized to a common square shape (256 x 256
pixels in the manuscript).

Two input formats are supported:

* NIfTI images (``.nii`` / ``.nii.gz``), optionally with a matching tumour mask (``<ID>_mask.nii.gz``)
  read with SimpleITK;
* NumPy arrays (``.npy``), used by the unit tests and the smoke test.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

try:  # SimpleITK is only required when NIfTI inputs are used
    import SimpleITK as sitk
except ImportError:  # pragma: no cover
    sitk = None

IMAGE_SUFFIXES = (".nii.gz", ".nii", ".npy")


def largest_tumour_slice(mask: np.ndarray) -> tuple[int, int, int]:
    """Return the (z, y, x) indices of the largest tumour cross-section in each plane."""
    if mask.sum() == 0:
        z, y, x = (s // 2 for s in mask.shape)
        return int(z), int(y), int(x)
    z = int(np.argmax(mask.sum(axis=(1, 2))))
    y = int(np.argmax(mask.sum(axis=(0, 2))))
    x = int(np.argmax(mask.sum(axis=(0, 1))))
    return z, y, x


def make_2p5d(volume: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Build a 3-channel 2.5D image (axial, sagittal, coronal) of shape (3, H, W).

    The three planes are extracted through the largest tumour cross-section and resized to a common
    square shape by nearest-neighbour interpolation.
    """
    volume = np.asarray(volume, dtype=np.float32)
    if mask is None:
        mask = np.ones_like(volume, dtype=np.uint8)
    mask = np.asarray(mask) > 0

    z, y, x = largest_tumour_slice(mask)
    axial = volume[z, :, :]
    sagittal = volume[:, :, x].T
    coronal = volume[:, y, :]

    planes = [np.rot90(p) for p in (axial, sagittal, coronal)]
    h = max(p.shape[0] for p in planes)
    w = max(p.shape[1] for p in planes)
    out = np.zeros((3, h, w), dtype=np.float32)
    for i, p in enumerate(planes):
        out[i, : p.shape[0], : p.shape[1]] = p
    return out


def read_nifti(path: Path) -> np.ndarray:
    if sitk is None:
        raise ImportError("SimpleITK is required to read NIfTI images.")
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.float32)


def load_case(image_path: Path, mask_path: Path | None = None) -> np.ndarray:
    if image_path.name.endswith(".npy"):
        volume = np.load(image_path)
        mask = np.load(mask_path) if mask_path is not None and Path(mask_path).exists() else None
    else:
        volume = read_nifti(image_path)
        mask = read_nifti(mask_path) if mask_path is not None and Path(mask_path).exists() else None
    return make_2p5d(volume, mask)


class TwoPointFiveDDataset(Dataset):
    """Dataset of pre- and post-nICT 2.5D inputs with an optional binary label.

    Parameters
    ----------
    records:
        Sequence of ``(case_id, pre_path, post_path, label)`` tuples. ``label`` may be ``None``.
    mask_dir:
        Optional directory holding ``<case_id>_mask.nii.gz`` (or ``.npy``) tumour masks.
    combine_timepoints:
        If ``True`` the pre- and post-nICT inputs are stacked into a 6-channel input.
    """

    def __init__(
        self,
        records,
        mask_dir: str | Path | None = None,
        combine_timepoints: bool = False,
    ) -> None:
        self.records = list(records)
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.combine_timepoints = combine_timepoints

    def __len__(self) -> int:
        return len(self.records)

    def _mask_for(self, case_id: str) -> Path | None:
        if self.mask_dir is None:
            return None
        for suffix in (".nii.gz", ".nii", ".npy"):
            candidate = self.mask_dir / f"{case_id}_mask{suffix}"
            if candidate.exists():
                return candidate
        return None

    def __getitem__(self, index: int):
        case_id, pre_path, post_path, label = self.records[index]
        mask_path = self._mask_for(case_id)
        pre = load_case(Path(pre_path), mask_path)
        post = load_case(Path(post_path), mask_path) if post_path else pre
        if self.combine_timepoints:
            image = np.concatenate([pre, post], axis=0)
        else:
            image = np.stack([pre, post], axis=0)  # (2, 3, H, W)
        tensor = torch.from_numpy(image)
        if label is None:
            return case_id, tensor
        return case_id, tensor, torch.tensor(float(label), dtype=torch.float32)


def make_records(features_csv: str | Path, label_column: str = "pCR"):
    """Build ``(case_id, pre_path, post_path, label)`` records from a manifest table.

    The manifest must contain the columns ``ID``, ``pre_path`` and ``post_path``, and optionally
    the label column. Paths may be absolute or relative to the manifest file.
    """
    import pandas as pd

    manifest_path = Path(features_csv)
    df = pd.read_csv(manifest_path)
    base = manifest_path.parent
    records = []
    for _, row in df.iterrows():
        pre = Path(row["pre_path"])
        post = Path(row["post_path"])
        pre = pre if pre.is_absolute() else base / pre
        post = post if post.is_absolute() else base / post
        label = float(row[label_column]) if label_column in df.columns else None
        records.append((str(row["ID"]), pre, post, label))
    return records
