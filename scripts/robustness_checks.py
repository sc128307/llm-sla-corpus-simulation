"""Robustness checks for Phase 4 conclusions.

Two alternative settings:
  R1: human reference = test split only (1,120 essays, never seen by models)
  R2: equalized subsample = 100 essays per (region x topic) cell (Biber
      down-sampling), applied to BOTH human and generated before JSD.

Both recompute JSD vs the alternative human reference for key features and
report whether the "LLM significantly deviates from human" conclusion holds.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from pathlib import Path

RES = "data/results"

df = pd.read_parquet(os.path.join(RES, "features_all_with_neosca.parquet"))
human = df[df["model"] == "HUMAN"]

# test split keys
sm = pd.read_csv("data/interim/split_manifest.csv")
test_keys = set(sm.loc[sm["split"] == "test"].apply(
    lambda r: f"{r['id']}|{r['topic']}", axis=1))
print(f"test-split keys: {len(test_keys)}")

human_test = human[human["key"].isin(test_keys)]
print(f"human test subset: {len(human_test)} rows")

def jsd(hv, gv):
    if len(hv) == 0 or len(gv) == 0:
        return np.nan
    lo, hi = min(hv.min(), gv.min()), max(hv.max(), gv.max())
    bins = np.linspace(lo, hi, 21)
    ph = np.histogram(hv, bins=bins, density=True)[0] + 1e-12
    pg = np.histogram(gv, bins=bins, density=True)[0] + 1e-12
    ph /= ph.sum(); pg /= pg.sum()
    m = 0.5 * (ph + pg)
    return 0.5 * (np.sum(ph * np.log2(ph / m)) + np.sum(pg * np.log2(pg / m)))

KEY_FEATS = ["AVD_Combined", "MTLD", "Grammar_Error_Rate", "MLT",
             "Clause_per_Sentence", "MDD"]

print("\n=== R1: JSD vs test-split human reference ===")
r1 = []
for feat in KEY_FEATS:
    hv = human_test[feat].dropna().values
    for model in [m for m in df["model"].unique() if m != "HUMAN"]:
        for cond in ["zero_shot", "cot"]:
            gv = df[(df["model"] == model) & (df["condition"] == cond)][feat].dropna().values
            r1.append({"feature": feat, "model": model, "condition": cond,
                       "JSD_test": jsd(hv, gv)})
r1_df = pd.DataFrame(r1)
r1_df.to_csv(os.path.join(RES, "robustness_R1_jsd.csv"), index=False)
print(f"R1 saved: {len(r1_df)} rows")

print("\n=== R2: equalized 100/cell subsample JSD ===")
def subsample(dfr, per_cell=100):
    """Sample per_cell essays per (region x topic) cell, seed 42."""
    return (dfr.groupby(["region", "topic"], group_keys=False)
            .apply(lambda g: g.sample(n=min(per_cell, len(g)), random_state=42)))

r2 = []
for feat in KEY_FEATS:
    # human subsample
    hs = subsample(human)
    hv = hs[feat].dropna().values
    for model in [m for m in df["model"].unique() if m != "HUMAN"]:
        for cond in ["zero_shot", "cot"]:
            gs = subsample(df[(df["model"] == model) & (df["condition"] == cond)])
            gv = gs[feat].dropna().values
            r2.append({"feature": feat, "model": model, "condition": cond,
                       "JSD_100": jsd(hv, gv)})
r2_df = pd.DataFrame(r2)
r2_df.to_csv(os.path.join(RES, "robustness_R2_jsd.csv"), index=False)
print(f"R2 saved: {len(r2_df)} rows")

print("\nDONE robustness checks")
