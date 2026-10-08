"""RP2 writer-cluster bootstrap sensitivity for the frozen L2 Primary sample.

This script is read-only with respect to frozen inputs.  It conditions on the
fixed 2,000-key selection reconstructed with phase4 seed 42, resamples writers
within region, and applies the same writer multiplicities to both topics and
all generated model-condition cells.  It reports percentile sensitivity
intervals only; no null p-values or q-values are produced.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

FEATURE_PATH = ROOT / "features" / "features_all_with_neosca.parquet"
HUMAN_PATH = ROOT / "corpus" / "human" / "human_data_full.csv"
ESSAY_BOOT_PATH = ROOT / "results" / "bootstrap_summary.csv"
OUT_PATH = ROOT / "results" / "rp2_writer_cluster_bootstrap_2026-10-06.csv"
MANIFEST_PATH = ROOT / "results" / "rp2_writer_cluster_bootstrap_manifest_2026-10-06.json"

SELECTION_SEED = 42
BOOTSTRAP_SEED = 20261006
B = 10000
KEYS_PER_STRATUM = 100

FEATURES = [
    "AVD_Combined", "MTLD", "MDD", "Nominalization_Rate",
    "Passive_Ratio", "Pronoun_Ratio", "Discourse_Density",
    "Grammar_Acceptability", "Grammar_Error_Rate", "MLT",
    "Clause_per_Sentence", "Top20_Word_Share", "Hapax_Ratio",
    "Lexical_Cohesion", "Referential_Density", "Sentence_Length",
    "Tree_Depth",
    "CEFR_A1", "CEFR_A2", "CEFR_B1", "CEFR_B2", "CEFR_C1",
]


def family(feature: str) -> str:
    if feature in {
        "AVD_Combined", "CEFR_A1", "CEFR_A2", "CEFR_B1", "CEFR_B2",
        "CEFR_C1", "MTLD", "Top20_Word_Share", "Hapax_Ratio",
        "Lexical_Cohesion", "FKGL",
    }:
        return "lexical"
    if feature in {"Discourse_Density", "Referential_Density"}:
        return "discourse"
    return "syntactic_register"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def fixed_selection(human_features: pd.DataFrame) -> dict[tuple[str, str], list[str]]:
    """Reproduce phase4's deterministic 100-key selection per stratum."""
    rng = np.random.default_rng(SELECTION_SEED)
    strata = {}
    for region in sorted(human_features["region"].dropna().unique()):
        if region == "ENS":
            continue
        for topic in sorted(human_features["topic"].dropna().unique()):
            keys = np.array(sorted(human_features.loc[
                (human_features["region"] == region)
                & (human_features["topic"] == topic), "key"
            ].dropna().unique()))
            if len(keys) < KEYS_PER_STRATUM:
                raise ValueError(f"{region}/{topic}: only {len(keys)} keys")
            if len(keys) > KEYS_PER_STRATUM:
                keys = rng.choice(keys, KEYS_PER_STRATUM, replace=False)
            strata[(str(region), str(topic))] = sorted(keys.tolist())
    if len(strata) != 20 or sum(len(v) for v in strata.values()) != 2000:
        raise AssertionError("unexpected fixed selection size")
    return strata


def main() -> None:
    features = pd.read_parquet(FEATURE_PATH)
    human_csv = pd.read_csv(HUMAN_PATH)
    human_features = features[features["model"] == "HUMAN"].copy()
    if human_features["key"].duplicated().any():
        raise AssertionError("human feature keys are not unique")

    # The corpus id is the feature key for human rows.  Validate the join
    # rather than inferring writer identity from a feature-table string.
    human_meta = human_csv[[
        "id", "region", "topic", "file_name"
    ]].copy()
    # The frozen pairing key is (id, topic), because the same writer id can
    # occur in both prompt topics.
    human_meta["key"] = (
        human_meta["id"].astype(str) + "|" + human_meta["topic"].astype(str)
    )
    human_meta = human_meta[["key", "id", "region", "topic", "file_name"]]
    if human_meta["key"].duplicated().any():
        raise AssertionError("human corpus keys are not unique")
    joined = human_features[["key", "region", "topic"]].merge(
        human_meta, on="key", how="left", validate="one_to_one",
        suffixes=("_feature", "_corpus"),
    )
    if joined[["region_corpus", "topic_corpus"]].isna().any().any():
        raise AssertionError("human feature/corpus key join is incomplete")
    if not (joined["region_feature"] == joined["region_corpus"]).all():
        raise AssertionError("region mismatch in human key join")
    if not (joined["topic_feature"] == joined["topic_corpus"]).all():
        raise AssertionError("topic mismatch in human key join")
    # The split manifest's writer id is the authoritative cluster key.  It is
    # read from the fixed classifier dataset metadata, then joined by key.
    oracle = pd.read_parquet(ROOT / "results" / "l1_dataset" / "oracle_human.parquet")
    oracle_meta = oracle[["key", "writer_id", "region", "topic"]].drop_duplicates("key")
    selected_strata = fixed_selection(human_features)
    selected_keys = [k for keys in selected_strata.values() for k in keys]
    selected = human_meta[human_meta["key"].isin(selected_keys)].merge(
        oracle_meta, on="key", how="left", validate="one_to_one",
        suffixes=("_corpus", "_oracle"),
    )
    if selected["writer_id"].isna().any():
        raise AssertionError("selected keys missing writer_id")
    if not (selected["region_corpus"] == selected["region_oracle"]).all():
        raise AssertionError("selected region mismatch against oracle metadata")
    if not (selected["topic_corpus"] == selected["topic_oracle"]).all():
        raise AssertionError("selected topic mismatch against oracle metadata")
    if selected["writer_id"].nunique() != 1534:
        raise AssertionError("unexpected selected writer count")

    # writer_index is local to region.  One draw vector is shared by both
    # topics, preserving within-writer dependence in every generated cell.
    region_state = {}
    for region in sorted(selected["region_corpus"].unique()):
        sub = selected[selected["region_corpus"] == region].copy()
        writers = sorted(sub["writer_id"].astype(str).unique())
        writer_index = {writer: i for i, writer in enumerate(writers)}
        sub["writer_index"] = sub["writer_id"].astype(str).map(writer_index)
        rng = np.random.default_rng(BOOTSTRAP_SEED + len(region_state))
        draws = rng.integers(0, len(writers), size=(B, len(writers)))
        counts = np.zeros((B, len(writers)), dtype=np.int16)
        for row, draw in enumerate(draws):
            counts[row] = np.bincount(draw, minlength=len(writers))
        region_state[region] = {"selected": sub, "counts": counts}

    human_by_key = human_features.set_index("key")
    essay_boot = pd.read_csv(ESSAY_BOOT_PATH)
    generated_cells = sorted(
        (str(model), str(condition))
        for model, condition in features.loc[features["model"] != "HUMAN",
                                             ["model", "condition"]]
        .drop_duplicates().itertuples(index=False, name=None)
    )
    if len(generated_cells) != 24:
        raise AssertionError(f"expected 24 generated cells, got {len(generated_cells)}")

    rows = []
    for model, condition in generated_cells:
        generated = features[
            (features["model"] == model) & (features["condition"] == condition)
        ].set_index("key")
        generated = generated.reindex(selected_keys)
        if generated[FEATURES].isna().any().any():
            raise AssertionError(f"missing generated features for {model}/{condition}")
        cell_essay = essay_boot[
            (essay_boot["analysis_set"] == "l2_primary")
            & (essay_boot["model"] == model)
            & (essay_boot["condition"] == condition)
        ].set_index("feature")
        for feature in FEATURES:
            delta = generated[feature].subtract(human_by_key.loc[selected_keys, feature])
            delta = delta.astype(float)
            total = np.zeros((B,), dtype=float)
            valid = np.ones((B,), dtype=bool)
            for (region, topic), keys in selected_strata.items():
                state = region_state[region]
                meta = state["selected"].set_index("key").loc[keys]
                values = delta.loc[keys].to_numpy(dtype=float)
                writer_idx = meta["writer_index"].to_numpy(dtype=int)
                weights = state["counts"][:, writer_idx]
                denominators = weights.sum(axis=1)
                valid &= denominators > 0
                numerators = weights @ values
                total += np.divide(
                    numerators, denominators,
                    out=np.full(B, np.nan, dtype=float),
                    where=denominators > 0,
                )
            boot_values = total[valid] / len(selected_strata)
            if len(boot_values) == 0:
                raise AssertionError(f"no valid writer bootstrap reps for {model}/{condition}/{feature}")
            essay_row = cell_essay.loc[feature]
            rows.append({
                "analysis_set": "l2_primary",
                "model": model,
                "condition": condition,
                "feature": feature,
                "feature_family": family(feature),
                "paired_delta": float(delta.mean()),
                "writer_bootstrap_ci_low": float(np.percentile(boot_values, 2.5)),
                "writer_bootstrap_ci_high": float(np.percentile(boot_values, 97.5)),
                "essay_bootstrap_ci_low": float(essay_row["paired_delta_ci_low"]),
                "essay_bootstrap_ci_high": float(essay_row["paired_delta_ci_high"]),
                "n_keys": 2000,
                "n_strata": 20,
                "n_writers": 1534,
                "n_writers_with_two_selected_essays": 466,
                "n_bootstrap_reps": B,
                "n_valid_reps": int(len(boot_values)),
                "n_invalid_reps": int(B - len(boot_values)),
                "bootstrap_method": "within-region writer cluster bootstrap; equal-stratum mean",
                "sampling_unit": "writer",
                "conditioning": "fixed 2,000 selected keys; fixed generated features; no null test",
                "selection_seed": SELECTION_SEED,
                "bootstrap_seed": BOOTSTRAP_SEED,
            })

    result = pd.DataFrame(rows)
    if len(result) != 528 or result[["model", "condition", "feature"]].duplicated().any():
        raise AssertionError("expected 528 unique candidate rows")
    result.to_csv(OUT_PATH, index=False)
    manifest = {
        "schema_version": 1,
        "unit": "RP2",
        "status": "complete",
        "scope": "Intrinsic writer-cluster interval sensitivity; classifier writer bootstrap blocked",
        "output_path": str(OUT_PATH.relative_to(ROOT)).replace("\\", "/"),
        "output_sha256": sha256(OUT_PATH),
        "input_sha256": {
            str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p)
            for p in [FEATURE_PATH, HUMAN_PATH, ESSAY_BOOT_PATH,
                      ROOT / "results" / "l1_dataset" / "oracle_human.parquet"]
        },
        "selection": {
            "seed": SELECTION_SEED,
            "keys": 2000,
            "strata": 20,
            "keys_per_stratum": 100,
            "writers": 1534,
            "writers_with_two_selected_essays": 466,
        },
        "bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "reps": B,
            "sampling_unit": "writer within region",
            "estimator": "equal-weight mean of 20 region-topic stratum means",
            "invalid_reps": int(result["n_invalid_reps"].max()),
        },
        "completeness": {
            "generated_cells": 24,
            "features_per_cell": 22,
            "rows": len(result),
            "family_counts": result["feature_family"].value_counts().to_dict(),
        },
        "classifier_route": {
            "status": "blocked",
            "reason": "No keyed fixed-test predictions; RP1 score discrepancies remain unreconciled",
            "frozen_checkpoints_modified": False,
        },
        "data_changed": False,
        "features_changed": False,
        "models_changed": False,
        "frozen_results_overwritten": False,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
