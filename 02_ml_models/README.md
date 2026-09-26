# 02 — Conventional radiomics machine-learning models

Training and evaluation of the **30 machine-learning models** compared in the manuscript
(Figure 1A/1B and Supplementary Fig. S2A/S2B), which include the multilayer perceptron (MLP) and
support vector machine (SVM) models that performed best at the pre- and post-nICT time points.

## The 30 models

`ml_config.yaml` lists the 30 models in the same order as the pre-nICT panel of Figure 1A.
**All 30 models are implemented**, using four backends: 27 through scikit-learn / boosting libraries
(including GentleBoost and RDA, which are provided by this repository), 2 through WEKA and 1 through
R. Missing backends are skipped automatically and reported.

```
MLP, XGBoost, CatBoost, SVM, ExtraTrees, RandomForest, RidgeLR, GradientBoosting, ElasticNet,
NuSVM, HistGradientBoosting, LR, LinearSVM, LightGBM, RotationForest, Bagging, AdaBoost,
ConditionalInferenceTree, LDA, LASSO, KNN, CART, GentleBoost, QDA, GaussianNB, PolynomialSVM,
SGDClassifier, LogitBoost, DecisionTree, RDA
```

**25 of the 30 are implemented directly**, with a scikit-learn or boosting-library equivalent:

| Manuscript model | Implementation |
| --- | --- |
| MLP | `MLPClassifier` |
| XGBoost / LightGBM / CatBoost | `XGBClassifier`, `LGBMClassifier`, `CatBoostClassifier` |
| SVM | `SVC(kernel="rbf")` |
| NuSVM | `NuSVC` |
| LinearSVM | `SVC(kernel="linear")` |
| PolynomialSVM | `SVC(kernel="poly")` |
| ExtraTrees / RandomForest | `ExtraTreesClassifier`, `RandomForestClassifier` |
| RidgeLR | `RidgeClassifier` |
| GradientBoosting / HistGradientBoosting | `GradientBoostingClassifier`, `HistGradientBoostingClassifier` |
| ElasticNet | `LogisticRegression(penalty="elasticnet")` |
| LR | `LogisticRegression(penalty="l2")` |
| LASSO | `LogisticRegression(penalty="l1")` |
| Bagging / AdaBoost | `BaggingClassifier`, `AdaBoostClassifier` |
| LDA / QDA | `LinearDiscriminantAnalysis`, `QuadraticDiscriminantAnalysis` |
| KNN | `KNeighborsClassifier` |
| CART | `DecisionTreeClassifier(criterion="gini")` |
| DecisionTree | `DecisionTreeClassifier(criterion="entropy")` |
| GaussianNB | `GaussianNB` |
| SGDClassifier | `SGDClassifier` |

**3 more are provided through other backends** (GentleBoost and RDA are also available as ordinary
scikit-learn style estimators in this repository):

| Manuscript model | How it is run |
| --- | --- |
| RDA | `LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto")` — Friedman's regularized discriminant analysis |
| GentleBoost | `GentleBoostClassifier` in `boosting.py` (Friedman, Hastie & Tibshirani 2000, regression stumps) |
| RotationForest | WEKA, class `weka.classifiers.meta.RotationForest` |
| LogitBoost | WEKA, class `weka.classifiers.meta.LogitBoost` |
| ConditionalInferenceTree | R, `partykit::ctree` |

The two models selected in the manuscript (MLP for the pre-nICT time point and SVM for the post-nICT
time point) are among the 27 models that run with scikit-learn alone.

## The WEKA backend

RotationForest and LogitBoost are provided by WEKA (Java) and are called through
`python-weka-wrapper3` (`weka_backend.py`): the feature table is written to ARFF, the classifier is
trained inside the JVM, and the predicted probabilities are returned to Python.

```bash
# 1. a Java runtime (JDK or JRE 8+); set JAVA_HOME or pass --java-home

# 2. the wrapper (bundles weka.jar)
pip install python-weka-wrapper3

# 3. the WEKA package that provides RotationForest
python -c "import weka.core.jvm as j; from weka.core import packages; j.start(); packages.install_package('rotationForest'); j.stop()"
```

Then run with the WEKA home directory that holds the installed packages:

```bash
python train_ml_classifiers.py --features ../data/example/features_demo.csv \
                               --labels ../data/example/labels_demo.csv \
                               --config ml_config.yaml --output ../data/expected_output/ml_demo \
                               --java-home /path/to/jdk --weka-home /path/to/wekafiles
```

`JAVA_HOME` and `WEKA_HOME` can be set as environment variables instead of the two options. If the
Java runtime or WEKA is not available, the two WEKA models are skipped and the remaining 28 models
still run.

## The R backend

ConditionalInferenceTree is provided by `partykit::ctree` (conditional inference trees) and is called
through `Rscript` (`r_backend.py`, `r_models.R`).

```bash
# R with the partykit package
Rscript -e "install.packages('partykit', repos='https://cloud.r-project.org')"
```

`Rscript` must be on the PATH, or its path passed with `--rscript`. If R is not available, that
model is skipped and the remaining 29 models still run.

## Inputs

- A feature table (csv): first column `ID`, all other columns numeric features. The demo uses
  `data/example/features_demo.csv`, whose columns mirror the four feature sets of the fusion model
  after selection (`PRE_RF_` 6, `POST_RF_` 8, `PRE_DLF_` 22, `POST_DLF_` 48 features).
  In the manuscript the models were trained on the same four selected sets: radiomic features
  started from 1,834 features per time point, and the deep-learning features from
  1,024-dimensional (pre-nICT, SimpleViT) and 25,088-dimensional (post-nICT, vgg19_bn) vectors,
  all reduced by univariable testing, Pearson redundancy removal and LASSO. The selection pipeline
  itself is not part of this release.
- A label table (csv): `ID`, `cohort` (`train`, `val`, `test1`, `test2`), the label column
  (`pCR` by default) and optionally survival columns. Outcome columns are never used as features.

## Outputs

- `metrics.csv` — AUC with a bootstrap 95% CI, accuracy, sensitivity, specificity, PPV and NPV,
  for every model and cohort.
- `predictions.csv` — predicted probability for every case and model.
- `run_summary.json` — number of models fitted, features, cases and the random seed.
- `models/` — the fitted scikit-learn models (`joblib`). Written to the output directory only;
  never committed to the repository.

## Scripts

| Script | Purpose |
| --- | --- |
| `train_ml_classifiers.py` | Fit and evaluate every enabled model (scikit-learn and WEKA backends) |
| `predict_ml_classifiers.py` | Apply saved scikit-learn models to a new feature table |
| `weka_backend.py` | ARFF conversion and JVM handling for the WEKA classifiers |
| `ml_config.yaml` | The 30 models, backends, hyper-parameters and seeds |
| `utils.py` | Table loading, cohort handling and performance metrics |

## Usage

```bash
python train_ml_classifiers.py --features ../data/example/features_demo.csv \
                               --labels ../data/example/labels_demo.csv \
                               --config ml_config.yaml \
                               --output ../data/expected_output/ml_demo

python predict_ml_classifiers.py --features ../data/example/features_demo.csv \
                                 --model-dir ../data/expected_output/ml_demo/models \
                                 --output ../data/expected_output/ml_demo/predictions_new.csv
```

Useful options: `--models LR,SVM` to run a subset, `--seed` to override the seed, and
`--feature-prefix PRE_RF_` (or `POST_RF_`, or several prefixes separated by commas) to train on one
feature set only — this is how the two panels of manuscript Figure 1 are reproduced:

```bash
# pre-nICT radiomic features (Figure 1A)
python train_ml_classifiers.py --features ../data/example/features_demo.csv \
                               --labels ../data/example/labels_demo.csv \
                               --config ml_config.yaml --feature-prefix PRE_RF_ \
                               --output ../data/expected_output/ml_pre

# post-nICT radiomic features (Figure 1B)
python train_ml_classifiers.py --features ../data/example/features_demo.csv \
                               --labels ../data/example/labels_demo.csv \
                               --config ml_config.yaml --feature-prefix POST_RF_ \
                               --output ../data/expected_output/ml_post
```

## Notes

- Feature standardisation is fitted on the training cohort only.
- Random seeds are fixed, so the comparison is reproducible.
- A model whose fitting assumptions are not met on a given dataset is reported and skipped without
  stopping the run (for example QDA needs more cases than features per class).
- The optional boosting packages (`xgboost`, `lightgbm`, `catboost`) are enabled in the config; if a
  package is not installed the corresponding model is skipped automatically.
