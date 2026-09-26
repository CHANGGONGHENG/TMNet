# Demo data (simulated)

Simulated tabular dataset used by the demo commands in the top-level `README.md`. All values are
synthetic; no real patient data, images or measurements are used.

**Files**

| File | Content |
| --- | --- |
| `features_demo.csv` | 120 simulated cases × 84 features |
| `labels_demo.csv` | `ID`, `cohort` (train 72 / val 18 / test1 18 / test2 12), `centre`, `pCR`, and simulated OS / DFS times and event indicators |

**Feature structure**

The column structure mirrors the four feature sets used by the fusion model in the manuscript after
feature selection (Figure S4); the values and the feature names are simulated:

| Prefix | Number of features | Manuscript feature set |
| --- | --- | --- |
| `PRE_RF_` | 6 | pre-nICT radiomic features after LASSO selection |
| `POST_RF_` | 8 | post-nICT radiomic features after LASSO selection |
| `PRE_DLF_` | 22 | pre-nICT deep-learning features after LASSO selection |
| `POST_DLF_` | 48 | post-nICT deep-learning features after LASSO selection |

Before selection the manuscript started from 1,834 radiomic features per time point and from
1,024-dimensional (pre-nICT, SimpleViT) and 25,088-dimensional (post-nICT, vgg19_bn) deep-learning
feature vectors; the selection pipeline is not part of this release.

**Simulated data.** No real patient data, images or measurements are used; the values are random,
the feature names are generic, and the numbers produced on this dataset carry no biological
meaning.

**Regenerating**

```bash
python data/generate_demo_data.py
```

The generator uses a fixed random seed, so the dataset is fully reproducible.
