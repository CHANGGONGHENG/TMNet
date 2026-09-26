# Expected output

Reference output for the demo commands in the top-level `README.md`, so that a reviewer can compare
their own run against a known-good result.

**The demo data are simulated.** All values below are properties of that simulation and carry no
biological meaning.

---

## 1. Machine learning — pre-nICT radiomic features

```
ml_pre/
├── metrics.csv          # AUC (with a bootstrap 95% CI), accuracy, sensitivity, specificity, PPV, NPV
├── predictions.csv      # predicted probability for every case and model
├── run_summary.json     # number of models fitted, features, cases and the random seed
├── weka_tmp/            # intermediate ARFF files written by the WEKA backend
├── r_tmp/               # intermediate csv files written by the R backend
└── models/              # fitted scikit-learn models -- regenerated locally, never committed
```

```bash
python train_ml_classifiers.py --features ../data/example/features_demo.csv \
                               --labels ../data/example/labels_demo.csv \
                               --config ml_config.yaml --feature-prefix PRE_RF_ \
                               --output ../data/expected_output/ml_pre \
                               --java-home <JDK> --weka-home <WEKA_HOME>
```

| | |
| --- | --- |
| Data | 120 simulated cases, 6 features (`PRE_RF_`), seed 42 |
| Models | all 30 models of the manuscript |
| Best internal validation / test 1 / test 2 AUC | 0.792 / 0.713 / 0.833 |
| Run time | ~30 s on a standard desktop computer, CPU only |

## 2. Machine learning — post-nICT radiomic features

Same command with `--feature-prefix POST_RF_` and `--output ../data/expected_output/ml_post`.

| | |
| --- | --- |
| Data | 120 simulated cases, 8 features (`POST_RF_`), seed 42 |
| Models | all 30 models of the manuscript |
| Best internal validation / test 1 / test 2 AUC | 0.750 / 0.825 / 0.861 |
| Run time | ~30 s on a standard desktop computer, CPU only |

**Environment for both runs**

Python 3.9.21, scikit-learn 1.0.2, xgboost 1.5.2, lightgbm 3.3.2, catboost 1.2.10,
python-weka-wrapper3 with OpenJDK 17, R 4.5.0 with partykit.
Output: 120 rows in each `metrics.csv` (30 models × 4 cohorts).

---

## 3. Deep learning — smoke test of the 32 candidate networks

The deep-learning pipeline needs image volumes and no image data are distributed, so the 32
candidate 2.5D networks are exercised on synthetic volumes:

```bash
python 03_dl_models/run_candidate_comparison.py --smoke-test \
        --config 03_dl_models/dl_config.yaml --output data/expected_output/dl_smoke
```

```
dl_smoke/
└── candidate_comparison.csv   # one row per candidate: status, best epoch, validation AUC, device
```

| | |
| --- | --- |
| Data | 16 synthetic cases (12 train / 4 validation), random 224 × 224 volumes |
| Models | **all 32 candidate networks of the manuscript** (Figure 1C/1D) trained and evaluated |
| Output | `candidate_comparison.csv` with 32 rows, all `status = ok` |
| Run time | 3.2 min on an NVIDIA GeForce RTX 5090 (32 GB), CUDA 12.8 |
| Checkpoints | written to the run directory during the smoke test and **not** distributed |

The smoke test only verifies that every candidate trains and predicts; the validation AUCs in that
table come from 4 synthetic validation cases and carry no meaning. Per-model training histories are
regenerated locally rather than committed.
