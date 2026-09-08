"""Null-based paired inference for the frozen phase-4 feature analysis.

This script deliberately leaves the original phase-4 outputs untouched.  It
reuses the same sampled keys, strata, feature definitions, and paired deltas as
``phase4_statistics.py`` but replaces the descriptive bootstrap sign
probability with a paired sign-flip randomisation p-value.  The sign-flip null
tests whether the mean paired difference is zero while preserving the paired
unit and the equal region-topic stratum weighting.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.chdir(Path(__file__).resolve().parents[1])

from src.paths import FEATURES_PATH, RESULTS_DIR


SEED = 42
N_NULL = int(os.getenv("SIGNFLIP_REPS", "5000"))
KEYS_PER_REGION_TOPIC = int(os.getenv("BOOTSTRAP_KEYS_PER_REGION_TOPIC", "100"))

FEATURES = [
    "AVD_Combined", "MTLD", "MDD", "Nominalization_Rate",
    "Passive_Ratio", "Pronoun_Ratio", "Discourse_Density",
    "Grammar_Acceptability", "Grammar_Error_Rate", "MLT",
    "Clause_per_Sentence", "Top20_Word_Share", "Hapax_Ratio",
    "Lexical_Cohesion", "Referential_Density", "Sentence_Length",
    "Tree_Depth",
]
CEFR = ["CEFR_A1", "CEFR_A2", "CEFR_B1", "CEFR_B2", "CEFR_C1"]
ALL_FEATS = FEATURES + CEFR


def feature_family(feature: str) -> str:
    if feature in {
        "AVD_Combined", "CEFR_A1", "CEFR_A2", "CEFR_B1", "CEFR_B2",
        "CEFR_C1", "MTLD", "Top20_Word_Share", "Hapax_Ratio",
        "Lexical_Cohesion", "FKGL",
    }:
        return "lexical"
    if feature in {"Discourse_Density", "Referential_Density"}:
        return "discourse"
    return "syntactic_register"


def bh_adjust(values: np.ndarray) -> np.ndarray:
    """Benjamini--Hochberg adjustment for one prespecified family."""
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


def build_strata(human: pd.DataFrame, regions: list[str], topics: list[str]) -> dict[str, list[str]]:
    """Match the fixed, region-topic-balanced key selection in phase 4."""
    rng = np.random.default_rng(SEED)
    strata: dict[str, list[str]] = {}
    for region in regions:
        for topic in topics:
            keys = np.array(sorted(human.loc[
                (human["region"] == region) &
                (human["topic"] == topic), "key"].dropna().unique()))
            if len(keys) < KEYS_PER_REGION_TOPIC:
                raise ValueError(
                    f"{region}/{topic}: {len(keys)} keys available, "
                    f"need {KEYS_PER_REGION_TOPIC}"
                )
            if len(keys) > KEYS_PER_REGION_TOPIC:
                keys = rng.choice(keys, KEYS_PER_REGION_TOPIC, replace=False)
            strata[f"{region}|{topic}"] = sorted(keys.tolist())
    return strata


def paired_values(
    human: pd.DataFrame,
    generated: pd.DataFrame,
    feature: str,
    strata: dict[str, list[str]],
) -> np.ndarray:
    """Return paired generated-minus-human values in stratum order."""
    h = human.groupby("key")[feature].mean().rename("h")
    g = generated.groupby("key")[feature].mean().rename("g")
    paired = pd.concat([h, g], axis=1).dropna()
    paired["delta"] = paired["g"] - paired["h"]
    values = []
    for keys in strata.values():
        group = paired["delta"].reindex(keys).dropna().to_numpy(dtype=float)
        if len(group):
            values.append(group)
    if not values:
        return np.array([], dtype=float)
    lengths = {len(group) for group in values}
    if len(lengths) != 1:
        raise ValueError("The equal-weight sign-flip test requires equal stratum sizes")
    return np.concatenate(values)


def signflip_pvalue(values: np.ndarray, signs: np.ndarray) -> tuple[float, float, float]:
    """Return observed mean, two-sided sign-flip p, and MC standard error.

    The sign matrix is generated once per analysis set and reused across tests;
    this makes the Monte Carlo procedure reproducible while avoiding any change
    to the observed paired data.
    """
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    observed = float(values.mean())
    null_stats = signs @ values / len(values)
    extreme = np.abs(null_stats) >= abs(observed) - 1e-12
    p = (1.0 + float(extreme.sum())) / (len(null_stats) + 1.0)
    mc_se = float(np.sqrt(max(p * (1.0 - p), 0.0) / (len(null_stats) + 1.0)))
    return observed, p, mc_se


def run_analysis(
    df: pd.DataFrame,
    label: str,
    regions: list[str],
    output_name: str,
    seed_offset: int,
) -> None:
    human = df[df["model"] == "HUMAN"].copy()
    topics = sorted(human["topic"].dropna().unique())
    strata = build_strata(human, regions, topics)
    all_keys = sum(len(keys) for keys in strata.values())
    rng = np.random.default_rng(SEED + seed_offset)
    signs = rng.choice(
        np.array([-1.0, 1.0], dtype=np.float32),
        size=(N_NULL, all_keys),
    )

    models = sorted(m for m in df["model"].unique() if m != "HUMAN")
    conditions = sorted(c for c in df["condition"].unique() if c != "baseline")
    rows = []
    for model in models:
        for condition in conditions:
            generated = df[(df["model"] == model) & (df["condition"] == condition)]
            for feature in ALL_FEATS:
                values = paired_values(human, generated, feature, strata)
                observed, p, mc_se = signflip_pvalue(values, signs)
                rows.append({
                    "analysis_set": label,
                    "model": model,
                    "condition": condition,
                    "feature": feature,
                    "feature_family": feature_family(feature),
                    "paired_delta": observed,
                    "signflip_p": p,
                    "signflip_mc_se": mc_se,
                    "n_keys": int(len(values)),
                    "n_strata": int(len(strata)),
                    "n_null_reps": N_NULL,
                    "null_method": "paired sign-flip; equal region-topic stratum weighting",
                })

    result = pd.DataFrame(rows)
    result["q_bh"] = np.nan
    for family, idx in result.groupby("feature_family").groups.items():
        result.loc[idx, "q_bh"] = bh_adjust(result.loc[idx, "signflip_p"].to_numpy())
    output = Path(RESULTS_DIR) / output_name
    result.to_csv(output, index=False)
    print(f"{label}: {len(result)} tests, {all_keys} paired keys, {N_NULL} sign-flips -> {output}")


def main() -> None:
    df = pd.read_parquet(FEATURES_PATH)
    human = df[df["model"] == "HUMAN"]
    all_regions = sorted(human["region"].dropna().unique())
    l2_regions = [region for region in all_regions if region != "ENS"]
    run_analysis(df, "l2_primary", l2_regions, "signflip_inference_primary.csv", 0)
    run_analysis(df, "with_ens_sensitivity", all_regions, "signflip_inference_with_ens.csv", 1)


if __name__ == "__main__":
    main()
