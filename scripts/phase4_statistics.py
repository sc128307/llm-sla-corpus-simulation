"""Phase 4 statistics: paired bootstrap (text-key level), JSD, dose-response.

Per ANALYSIS_PLAN.md (user-approved):
  - Paired bootstrap: resample at text-key level (all 25 observations for a
    key move together), B=1000, 95% CI (percentile).
  - Human reference: full 5,600 (main); test-split 1,120 (robustness).
  - Statistics computed per (model x condition): feature means + CI, JSD vs
    human + CI, Cohen's d.
  - Outputs: bootstrap_summary.csv, jensen_shannon.csv, effect_sizes.csv.

Input: data/results/features_all_with_neosca.parquet
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

RES = "data/results"
B = 1000
SEED = 42

FEATURES = ["AVD_Combined", "MTLD", "MDD", "Nominalization_Rate",
            "Passive_Ratio", "Pronoun_Ratio", "Discourse_Density",
            "Grammar_Acceptability", "Grammar_Error_Rate", "MLT",
            "Clause_per_Sentence", "Top20_Word_Share", "Hapax_Ratio",
            "Lexical_Cohesion", "Referential_Density", "Sentence_Length",
            "Tree_Depth"]
# CEFR columns (7) can be added as a group
CEFR = ["CEFR_A1", "CEFR_A2", "CEFR_B1", "CEFR_B2", "CEFR_C1"]
ALL_FEATS = FEATURES + CEFR

rng = np.random.default_rng(SEED)

df = pd.read_parquet(os.path.join(RES, "features_all_with_neosca.parquet"))
print(f"loaded {len(df)} rows x {len(df.columns)} cols")

# ---- paired bootstrap helper (VECTORIZED) ----
def paired_bootstrap_vec(key_means, n_iter=B):
    """key_means: 1-D array of per-key feature means (one scalar per key).
    Vectorized resampling: sample keys with replacement n_iter times,
    mean of sampled keys each iteration -> bootstrap distribution."""
    n = len(key_means)
    idx = rng.integers(0, n, size=(n_iter, n))
    return key_means[idx].mean(axis=1)


def ci(arr):
    return np.percentile(arr, [2.5, 97.5])


# ---- 1. Feature means + CI per (model x condition), paired bootstrap ----
print("\n=== 1. Paired bootstrap: feature means per (model, condition) ===")
summary_rows = []
for model in df["model"].unique():
    for cond in df["condition"].unique():
        sub = df[(df["model"] == model) & (df["condition"] == cond)]
        for feat in ALL_FEATS:
            key_means = sub.groupby("key")[feat].mean().dropna().values
            if len(key_means) == 0:
                continue
            if len(key_means) > 300:
                key_means = rng.choice(key_means, 300, replace=False)
            means = paired_bootstrap_vec(key_means)
            lo, hi = ci(means)
            summary_rows.append({
                "model": model, "condition": cond, "feature": feat,
                "mean": float(key_means.mean()),
                "ci_low": float(lo), "ci_high": float(hi),
                "n_keys": len(key_means),
            })
summary = pd.DataFrame(summary_rows)
summary.to_csv(os.path.join(RES, "bootstrap_summary.csv"), index=False)
print(f"bootstrap_summary.csv: {len(summary)} rows "
      f"({len(ALL_FEATS)} feats x {len(df['model'].unique())} models x "
      f"{len(df['condition'].unique())} conds)")

# ---- 2. JSD vs human (full 5,600 as main reference) ----
print("\n=== 2. JSD vs human reference (full baseline) ===")
human = df[df["model"] == "HUMAN"]
jsd_rows = []
for feat in ALL_FEATS:
    hv = human[feat].dropna().values
    if len(hv) == 0:
        continue
    for model in [m for m in df["model"].unique() if m != "HUMAN"]:
        for cond in df["condition"].unique():
            gv = df[(df["model"] == model) &
                    (df["condition"] == cond)][feat].dropna().values
            if len(gv) == 0:
                continue
            # JSD via histograms
            lo = min(hv.min(), gv.min())
            hi = max(hv.max(), gv.max())
            bins = np.linspace(lo, hi, 21)
            ph = np.histogram(hv, bins=bins, density=True)[0] + 1e-12
            pg = np.histogram(gv, bins=bins, density=True)[0] + 1e-12
            ph /= ph.sum()
            pg /= pg.sum()
            m = 0.5 * (ph + pg)
            jsd = 0.5 * (np.sum(ph * np.log2(ph / m)) +
                         np.sum(pg * np.log2(pg / m)))
            jsd_rows.append({"model": model, "condition": cond,
                             "feature": feat, "JSD": float(jsd)})
jsd = pd.DataFrame(jsd_rows)
jsd.to_csv(os.path.join(RES, "jensen_shannon.csv"), index=False)
print(f"jensen_shannon.csv: {len(jsd)} rows")

# ---- 3. Cohen's d per (model x condition x feature) vs human ----
print("\n=== 3. Cohen's d vs human ===")
d_rows = []
for feat in ALL_FEATS:
    hv = human[feat].dropna().values
    if len(hv) == 0:
        continue
    for model in [m for m in df["model"].unique() if m != "HUMAN"]:
        for cond in df["condition"].unique():
            gv = df[(df["model"] == model) &
                    (df["condition"] == cond)][feat].dropna().values
            if len(gv) == 0:
                continue
            sp = np.sqrt(((len(hv) - 1) * hv.var(ddof=1) +
                          (len(gv) - 1) * gv.var(ddof=1)) /
                         (len(hv) + len(gv) - 2))
            d = (gv.mean() - hv.mean()) / sp if sp > 0 else 0.0
            d_rows.append({"model": model, "condition": cond,
                           "feature": feat, "cohens_d": float(d)})
effect = pd.DataFrame(d_rows)
effect.to_csv(os.path.join(RES, "effect_sizes.csv"), index=False)
print(f"effect_sizes.csv: {len(effect)} rows")

print("\nDONE — Phase 4 statistics complete")
