# 01 — Image pre-processing

Prepares the pre- and post-nICT contrast-enhanced CT series for feature extraction and modelling.

**Inputs**

- DICOM series of contrast-enhanced chest CT, one folder per study and per time point
  (pre-nICT: within 2 weeks before the start of nICT; post-nICT: after 2–3 cycles, before surgery).

**Outputs**

- Selected series manifest (`series_manifest.csv`).
- Images resampled to 1 × 1 × 1 mm³ isotropic voxels.
- Intensity-normalised images (fixed mediastinal window: width 400 HU, level 40 HU).

**Scripts**

| Script | Purpose |
| --- | --- |
| `dicom_series_selection.py` | List every DICOM series in a study and select the contrast-enhanced chest CT series; writes a manifest for manual review |
| `resample.py` | Resample a DICOM series or NIfTI file to an isotropic voxel spacing |
| `normalize.py` | Apply the fixed mediastinal window to CT images |

**Usage**

```bash
python dicom_series_selection.py --input raw/ --output series_manifest.csv

python resample.py  --manifest series_manifest.csv --output nifti/ --spacing 1 1 1
python normalize.py --manifest series_manifest.csv --input-dir nifti --output normalized --ww 400 --wl 40
```

Single-case use:

```bash
python resample.py  --input raw/DEMO001 --output nifti/DEMO001.nii.gz
python normalize.py --input nifti/DEMO001.nii.gz --output normalized/DEMO001.nii.gz
```

**Notes**

- The same pre-processing pipeline is applied to the pre- and post-nICT images.
- `resample.py` and `normalize.py` require SimpleITK (`requirements-research.txt`).
- The series-selection manifest should be reviewed by hand; the automatic rule matches the series
  description against the contrast-enhanced chest CT keywords used at the participating centres.
