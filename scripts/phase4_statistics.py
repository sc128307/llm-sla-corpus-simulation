"""Phase 4 statistics for the frozen matched corpus.

Primary estimand: ten L2 regions, with 100 unique (region, topic) keys per
cell (200 keys per region), followed by paired bootstrap with replacement
within each region x topic stratum.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from src.paths import FEATURES_PATH, RESULTS_DIR

RES = str(RESULTS_DIR)
B = 1000
SEED = 42
KEYS_PER_REGION_TOPIC = int(os.getenv("BOOTSTRAP_KEYS_PER_REGION_TOPIC", "100"))

FEATURES = ["AVD_Combined", "MTLD", "MDD", "Nominalization_Rate",
            "Passive_Ratio", "Pronoun_Ratio", "Discourse_Density",
            "Grammar_Acceptability", "Grammar_Error_Rate", "MLT",
            "Clause_per_Sentence", "Top20_Word_Share", "Hapax_Ratio",
            "Lexical_Cohesion", "Referential_Density", "Sentence_Length",
            "Tree_Depth"]
CEFR = ["CEFR_A1", "CEFR_A2", "CEFR_B1", "CEFR_B2", "CEFR_C1"]
ALL_FEATS = FEATURES + CEFR


def feature_family(feature):
    if feature in {"AVD_Combined", "CEFR_A1", "CEFR_A2", "CEFR_B1",
                   "CEFR_B2", "CEFR_C1", "MTLD", "Top20_Word_Share",
                   "Hapax_Ratio", "Lexical_Cohesion", "FKGL"}:
        return "lexical"
    if feature in {"Discourse_Density", "Referential_Density"}:
        return "discourse"
    return "syntactic_register"


df = pd.read_parquet(FEATURES_PATH)
print(f"loaded {len(df)} rows x {len(df.columns)} cols")
human = df[df["model"] == "HUMAN"].copy()
all_regions = sorted(human["region"].dropna().unique())
l2_regions = [r for r in all_regions if r != "ENS"]
topics = sorted(human["topic"].dropna().unique())


def build_strata(regions):
    """Choose a fixed topic-balanced key set for each analysis."""
    rng = np.random.default_rng(SEED)
    strata = {}
    for region in regions:
        for topic in topics:
            keys = np.array(sorted(human.loc[
                (human["region"] == region) &
                (human["topic"] == topic), "key"].dropna().unique()))
            if len(keys) < KEYS_PER_REGION_TOPIC:
                raise ValueError(
                    f"{region}/{topic}: {len(keys)} keys available, "
                    f"need {KEYS_PER_REGION_TOPIC}")
            if len(keys) > KEYS_PER_REGION_TOPIC:
                keys = rng.choice(keys, KEYS_PER_REGION_TOPIC, replace=False)
            strata[f"{region}|{topic}"] = set(keys.tolist())
    return strata


def stratified_bootstrap(groups, rng, n_iter=B):
    """Bootstrap a mean while giving every stratum equal weight."""
    if not groups:
        return np.array([], dtype=float)
    out = np.zeros(n_iter, dtype=float)
    for values in groups:
        values = np.asarray(values, dtype=float)
        if len(values) == 0:
            continue
        idx = rng.integers(0, len(values), size=(n_iter, len(values)))
        out += values[idx].mean(axis=1)
    return out / len(groups)


def key_groups(sub, feat, strata):
    keyed = sub.groupby("key")[feat].mean()
    groups = []
    for keys in strata.values():
        values = keyed.reindex(sorted(keys)).dropna().to_numpy()
        if len(values):
            groups.append(values)
    return groups


def paired_delta_groups(sub, feat, strata):
    h = human.groupby("key")[feat].mean().rename("h")
    g = sub.groupby("key")[feat].mean().rename("g")
    paired = pd.concat([h, g], axis=1).dropna()
    paired["delta"] = paired["g"] - paired["h"]
    groups = []
    for keys in strata.values():
        values = paired["delta"].reindex(sorted(keys)).dropna().to_numpy()
        if len(values):
            groups.append(values)
    return groups


def percentile_ci(values):
    return np.percentile(values, [2.5, 97.5])


def bootstrap_pvalue(values):
    """Two-sided sign probability from the paired bootstrap distribution."""
    if len(values) == 0:
        return np.nan
    p = 2.0 * min(np.mean(values <= 0), np.mean(values >= 0))
    return float(min(1.0, max(1.0 / len(values), p)))


def bh_adjust(values):
    values = np.asarray(values, dtype=float)
    out = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(values)
    p = values[valid]
    if len(p) == 0:
        return out
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    ranked = np.clip(ranked, 0, 1)
    tmp = np.empty(len(p), dtype=float)
    tmp[order] = ranked
    out[valid] = tmp
    return out


def jsd(ph, pg):
    ph = np.asarray(ph, dtype=float) + 1e-12
    pg = np.asarray(pg, dtype=float) + 1e-12
    ph /= ph.sum()
    pg /= pg.sum()
    m = 0.5 * (ph + pg)
    return float(0.5 * (np.sum(ph * np.log2(ph / m)) +
                        np.sum(pg * np.log2(pg / m))))


def run_analysis(label, regions, summary_path, effect_path, jsd_path,
                 region_jsd_path):
    print(f"\n=== {label}: {len(regions)} regions ===")
    strata = build_strata(regions)
    print("strata keys:", len(strata), "unique keys:",
          sum(len(x) for x in strata.values()))
    sources = [("HUMAN", "baseline")]
    models = sorted(m for m in df["model"].unique() if m != "HUMAN")
    conditions = sorted(c for c in df["condition"].unique()
                        if c != "baseline")
    sources.extend((m, c) for m in models for c in conditions)

    summary_rows = []
    for model, cond in sources:
        sub = df[(df["model"] == model) & (df["condition"] == cond)]
        for feat in ALL_FEATS:
            groups = key_groups(sub, feat, strata)
            if not groups:
                continue
            values = np.concatenate(groups)
            rng = np.random.default_rng(SEED + len(summary_rows))
            boots = stratified_bootstrap(groups, rng)
            lo, hi = percentile_ci(boots)
            row = {
                "analysis_set": label, "model": model, "condition": cond,
                "feature": feat, "feature_family": feature_family(feat),
                "mean": float(values.mean()), "ci_low": float(lo),
                "ci_high": float(hi),
                "n_keys": int(sum(len(x) for x in groups)),
                "n_strata": len(groups),
            }
            if model != "HUMAN":
                deltas = paired_delta_groups(sub, feat, strata)
                dvals = np.concatenate(deltas) if deltas else np.array([])
                drng = np.random.default_rng(SEED + 100000 +
                                             len(summary_rows))
                dboot = stratified_bootstrap(deltas, drng)
                dlo, dhi = percentile_ci(dboot)
                row.update({
                    "paired_delta": float(dvals.mean()),
                    "paired_delta_ci_low": float(dlo),
                    "paired_delta_ci_high": float(dhi),
                    "bootstrap_p": bootstrap_pvalue(dboot),
                })
            summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    generated = summary[summary["model"] != "HUMAN"].copy()
    if not generated.empty:
        generated["q_bh"] = generated.groupby(
            "feature_family", group_keys=False)["bootstrap_p"].transform(
                bh_adjust)
        summary = summary.merge(
            generated[["analysis_set", "model", "condition", "feature",
                       "q_bh"]],
            on=["analysis_set", "model", "condition", "feature"],
            how="left")
    summary.to_csv(os.path.join(RES, summary_path), index=False)
    print(summary_path, len(summary), "rows")

    effect_rows = []
    for model, cond in sources:
        if model == "HUMAN":
            continue
        sub = df[(df["model"] == model) & (df["condition"] == cond)]
        for feat in ALL_FEATS:
            groups = paired_delta_groups(sub, feat, strata)
            if not groups:
                continue
            delta = np.concatenate(groups)
            sd = delta.std(ddof=1) if len(delta) > 1 else 0.0
            effect_rows.append({
                "analysis_set": label, "model": model, "condition": cond,
                "feature": feat, "feature_family": feature_family(feat),
                "d_z": float(delta.mean() / sd) if sd > 0 else 0.0,
                "n_keys": int(len(delta)), "effect_definition": "paired_dz",
            })
    pd.DataFrame(effect_rows).to_csv(os.path.join(RES, effect_path),
                                     index=False)
    print(effect_path, len(effect_rows), "rows")

    scoped = df[df["region"].isin(regions)]
    jsd_rows = []
    region_rows = []
    for feat in ALL_FEATS:
        values = scoped[feat].dropna().to_numpy()
        if len(values) == 0:
            continue
        lo, hi = float(values.min()), float(values.max())
        bins = np.linspace(lo, hi, 21) if lo != hi else None
        hv = scoped[scoped["model"] == "HUMAN"][feat].dropna().to_numpy()
        for model in models:
            for cond in conditions:
                gv = scoped[(scoped["model"] == model) &
                            (scoped["condition"] == cond)][feat].dropna().to_numpy()
                if len(hv) and len(gv):
                    value = 0.0 if bins is None else jsd(
                        np.histogram(hv, bins=bins)[0],
                        np.histogram(gv, bins=bins)[0])
                    jsd_rows.append({
                        "analysis_set": label, "model": model,
                        "condition": cond, "feature": feat, "JSD": value,
                        "bin_rule": "20 global feature-range bins",
                    })
                for region in regions:
                    hr = scoped[(scoped["model"] == "HUMAN") &
                                (scoped["region"] == region)][feat].dropna().to_numpy()
                    gr = scoped[(scoped["model"] == model) &
                                (scoped["condition"] == cond) &
                                (scoped["region"] == region)][feat].dropna().to_numpy()
                    if len(hr) and len(gr):
                        value = 0.0 if bins is None else jsd(
                            np.histogram(hr, bins=bins)[0],
                            np.histogram(gr, bins=bins)[0])
                        region_rows.append({
                            "analysis_set": label, "model": model,
                            "condition": cond, "region": region,
                            "feature": feat, "JSD": value,
                            "n_human": int(len(hr)),
                            "n_generated": int(len(gr)),
                            "bin_rule": "20 global feature-range bins",
                        })
    pd.DataFrame(jsd_rows).to_csv(os.path.join(RES, jsd_path), index=False)
    pd.DataFrame(region_rows).to_csv(os.path.join(RES, region_jsd_path),
                                     index=False)
    print(jsd_path, len(jsd_rows), "rows")
    print(region_jsd_path, len(region_rows), "rows")


run_analysis("l2_primary", l2_regions, "bootstrap_summary.csv",
             "effect_sizes.csv", "jensen_shannon.csv",
             "jensen_shannon_region.csv")

print("\nDONE — Phase 4 statistics complete")
