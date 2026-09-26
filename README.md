# TMNet — Time-Series Multi-domain Net for response prediction in oesophageal squamous cell carcinoma

Code accompanying the manuscript:

> **Longitudinal CT radiomics and deep learning for response prediction and prognostic stratification in esophageal squamous cell carcinoma after neoadjuvant immunochemotherapy**

The prediction models integrate pre- and post-neoadjuvant immunochemotherapy (nICT) contrast-enhanced
CT to predict pathological complete response (pCR) and to stratify overall survival (OS) and
disease-free survival (DFS).

---

## Repository contents

This release contains three components of the analysis pipeline.

| Folder | Description |
| --- | --- |
| `01_preprocessing/` | Image pre-processing: DICOM series selection, resampling to 1 × 1 × 1 mm³ and intensity normalisation (window width 400 HU / level 40 HU) |
| `02_ml_models/` | The 30 conventional radiomics machine-learning models compared in the manuscript (Figure 1A/1B); 25 of them are implemented here and 5 are reported as unavailable in scikit-learn |
| `03_dl_models/` | The 32 candidate 2.5D deep-learning networks compared in the manuscript (Figure 1C/1D), drawn from a zoo of 50 architectures |
| `data/` | Small simulated demo dataset and the expected output |

---

## 1. System requirements

**Operating system**

- Windows 11 (24H2) — development and test platform
- Linux (x86-64) — expected to work; not tested

**Python environments**

| Environment | Python | Purpose |
| --- | --- | --- |
| `deeplearning` | 3.11.15 | 2.5D networks and the TMNet fusion model (PyTorch) |
| `research` | 3.9 | Image pre-processing and the 30 machine-learning classifiers |

Exact package versions: `requirements-deeplearning.txt`, `requirements-research.txt`.

**Non-standard hardware**

Training and inference of the deep-learning models require an NVIDIA GPU. All deep-learning results
in the manuscript were produced on an **NVIDIA GeForce RTX 5090 (32 GB)** with **CUDA 12.8**
(driver 610.47). Pre-processing and the machine-learning classifiers run on a standard desktop CPU.

**Versions tested**

| Component | Version |
| --- | --- |
| PyTorch | 2.11.0+cu128 |
| scikit-learn | 1.0.2 (machine-learning environment) / 1.9.0 (deep-learning environment) |
| SimpleITK | 2.4.1 |
| NumPy | 1.24.2 |
| OpenJDK | 17 (optional, for the two WEKA models) |

No other versions have been tested.

**Optional backends: WEKA (Java) and R**

Of the 30 machine-learning models, 27 run with scikit-learn / boosting libraries alone. Two
(RotationForest, LogitBoost) are provided by the WEKA toolkit and need a Java runtime (JDK/JRE 8+)
with `python-weka-wrapper3`; one (ConditionalInferenceTree) is provided by R with `partykit`. If
Java, WEKA or R is missing, the corresponding models are skipped and the rest still run. See
`02_ml_models/README.md` for the setup.

---

## 2. Installation guide

```bash
# 1. deep-learning environment
conda create -n deeplearning python=3.11
conda activate deeplearning
pip install -r requirements-deeplearning.txt

# 2. pre-processing / machine-learning environment
conda create -n research python=3.9
conda activate research
pip install -r requirements-research.txt
```

**Typical install time:** approximately 3.5 min for the `research` environment (measured on a
standard desktop computer with a broadband connection). The `deeplearning` environment additionally
downloads the CUDA-enabled PyTorch wheel, which is several GB.

---

## 3. Demo

The demo runs on a **simulated tabular dataset** (`data/example/`), so that the pipeline can be
exercised without access to patient data:

- `features_demo.csv` — 120 simulated cases × 84 features, with the same structure as the four
  feature sets of the fusion model (6 pre-nICT radiomic + 8 post-nICT radiomic + 22 pre-nICT
  deep-learning + 48 post-nICT deep-learning features; all values synthetic)
- `labels_demo.csv` — cohort assignment (train / val / test1 / test2), pCR label and simulated
  survival times

All values in the simulated dataset are random and carry no biological meaning; the dataset exists
so that the pipeline can be run end to end without access to patient data.

```bash
conda activate research

# a) pre-nICT radiomic features (manuscript Figure 1A)
python 02_ml_models/train_ml_classifiers.py --features data/example/features_demo.csv \
                                           --labels data/example/labels_demo.csv \
                                           --config 02_ml_models/ml_config.yaml \
                                           --feature-prefix PRE_RF_ \
                                           --output data/expected_output/ml_pre

# b) post-nICT radiomic features (manuscript Figure 1B)
python 02_ml_models/train_ml_classifiers.py --features data/example/features_demo.csv \
                                           --labels data/example/labels_demo.csv \
                                           --config 02_ml_models/ml_config.yaml \
                                           --feature-prefix POST_RF_ \
                                           --output data/expected_output/ml_post
# optional, to include the two WEKA models (RotationForest, LogitBoost):
#   --java-home <path to JDK> --weka-home <path to wekafiles>
```

**Expected output**

```
data/expected_output/
├── ml_pre/                # 30 models on the pre-nICT radiomic features
│   ├── metrics.csv        # AUC (95% CI), accuracy, sensitivity, specificity, PPV, NPV
│   ├── predictions.csv    # predicted probabilities
│   ├── run_summary.json
│   └── models/            # fitted models (regenerated locally, never committed)
└── ml_post/               # the same for the post-nICT radiomic features
```

**Expected run time for demo:** approximately 30 s (measured: 29 s with Python 3.9.21 and
scikit-learn 1.0.2 on a standard desktop computer, CPU only). The trained classifiers are written to
the output directory and are not distributed.

A smoke test of the deep-learning code, which needs neither images nor a GPU, can be run with:

```bash
conda activate deeplearning
python 03_dl_models/train_dl.py --smoke-test --config 03_dl_models/dl_config.yaml \
                               --output runs/smoke
```

Running all 32 candidate networks through the smoke test takes about 3 min on an RTX 5090; the
resulting comparison table is in `data/expected_output/dl_smoke/`.

---

## 4. Instructions for use

1. Prepare contrast-enhanced chest CT series in DICOM format, one folder per patient and per time
   point (pre-nICT and post-nICT).
2. Run `01_preprocessing/` to resample and normalise the images.
3. Obtain tumour masks and radiomic features for your own data (see the manuscript Methods).
4. Run `02_ml_models/` to train and apply the conventional radiomics classifiers.
5. Run `03_dl_models/` to train the 2.5D networks and to extract the deep-learning features.

All scripts take input and output paths as command-line arguments; no paths are hard-coded.

---

## Where the code is described in the manuscript

The functionality of the code and the architecture of the models are described in the **Methods
section** and the **Supplementary Methods** of the manuscript.

---

## License

Released under the MIT License — see `LICENSE`.

## Citation

If you use this code, please cite the manuscript above. See `CITATION.cff`.
