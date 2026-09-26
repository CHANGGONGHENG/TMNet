"""Generate the simulated tabular demo dataset used by the example commands.

All data are simulated: no real patient data, images or measurements are used, and the feature
names are generic. The column structure mirrors the feature sets of the manuscript after feature
selection (pre-RFs, post-RFs, pre-DLFs, post-DLFs), and the signal strength was chosen so that the
models reach a realistic, non-trivial performance.

Run:
    python data/generate_demo_data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N = 120
OUT = Path(__file__).resolve().parent / "example"
OUT.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(SEED)

# Number of retained features per set, as reported for the manuscript's fusion model.
N_PRE_RF, N_POST_RF, N_PRE_DLF, N_POST_DLF = 6, 8, 22, 48

# Loading of each feature set on the shared latent tumour signal, and the outcome model.
# Chosen so that every feature set predicts pCR with a realistic, non-trivial AUC.
LOADING = {"PRE_RF": 1.00, "POST_RF": 1.08, "PRE_DLF": 1.05, "POST_DLF": 1.35}
SHAPES = {"PRE_RF": N_PRE_RF, "POST_RF": N_POST_RF, "PRE_DLF": N_PRE_DLF, "POST_DLF": N_POST_DLF}
OUTCOME_INTERCEPT = -0.7
OUTCOME_SIGNAL = 0.9
OUTCOME_NOISE = 1.2

ids = [f"DEMO{i:03d}" for i in range(1, N + 1)]
centres = rng.choice(["CTR-A", "CTR-B", "CTR-C", "CTR-D"], size=N)

# latent tumour signal that drives the imaging features and the outcome
z = rng.normal(size=N)
# independent, unobserved component of the outcome
e = rng.normal(size=N)

logit = OUTCOME_INTERCEPT + OUTCOME_SIGNAL * z + OUTCOME_NOISE * e
p_pcr = 1.0 / (1.0 + np.exp(-logit))
pcr = (rng.random(N) < p_pcr).astype(int)

df = pd.DataFrame({"ID": ids})
for prefix, n_features in SHAPES.items():
    block = LOADING[prefix] * z[:, None] + rng.normal(size=(N, n_features))
    for j in range(n_features):
        df[f"{prefix}_{j + 1:03d}"] = np.round(block[:, j], 6)
df.to_csv(OUT / "features_demo.csv", index=False)

# survival times: longer for patients who achieved pCR
os_time = np.round(rng.exponential(30.0, N) * np.where(pcr == 1, 1.6, 1.0), 2)
dfs_time = np.round(rng.exponential(24.0, N) * np.where(pcr == 1, 1.7, 1.0), 2)
os_event = (rng.random(N) < 0.55).astype(int)
dfs_event = (rng.random(N) < 0.60).astype(int)

cohort = np.array(["train"] * 72 + ["val"] * 18 + ["test1"] * 18 + ["test2"] * 12)
assert len(cohort) == N

labels = pd.DataFrame(
    {
        "ID": ids,
        "cohort": cohort,
        "centre": centres,
        "pCR": pcr,
        "OS_time_months": os_time,
        "OS_event": os_event,
        "DFS_time_months": dfs_time,
        "DFS_event": dfs_event,
    }
)
labels.to_csv(OUT / "labels_demo.csv", index=False)

n_features = sum(SHAPES.values())
print(f"wrote {OUT / 'features_demo.csv'}  ({df.shape[0]} rows x {df.shape[1]} columns, "
      f"{n_features} features: {N_PRE_RF} pre-RF + {N_POST_RF} post-RF + "
      f"{N_PRE_DLF} pre-DLF + {N_POST_DLF} post-DLF)")
print(f"wrote {OUT / 'labels_demo.csv'}    ({labels.shape[0]} rows x {labels.shape[1]} columns)")
print(f"pCR prevalence: {pcr.mean():.3f}")
print(labels["cohort"].value_counts().to_dict())
