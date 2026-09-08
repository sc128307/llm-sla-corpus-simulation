"""Phase 1 — Build the few-shot exemplar bank with a leakage-safe split.

Creates two artifacts under corpus/manifest/:
  split_manifest.csv  — every human essay + its split assignment
                        (test | pool), stratified by (region, topic).
  exemplar_pool.csv   — the pool subset; few-shot exemplars MUST be drawn
                        from here, never from the 'test' rows.

Leakage rule (the exemplar pool and classifier test split are disjoint):
  The L1 classifier (Phase 5) is evaluated on held-out HUMAN essays. If an
  exemplar that appeared inside a few-shot prompt were also in that test
  split, the classifier could memorise it and inflate Sim-to-Real accuracy.
  => exemplar pool and classifier test split are DISJOINT by construction.

Design:
  - Default (backwards-compatible): 20% stratified holdout per (region, topic)
    -> 'test', 80% -> 'pool'.
  - Optional ``--split-unit writer`` keeps both topics by the same writer and
    is recommended as a sensitivity analysis; it changes which exemplars were
    used and therefore must not be mixed with an existing generation run.
  - Seed fixed (42) and recorded in the manifest path/name for the method
    section: "train/pool-test split, 80/20 stratified by region x topic,
    seed 42".

Usage:
  python scripts/build_exemplar_bank.py [--test-fraction 0.2] [--seed 42]
  python scripts/build_exemplar_bank.py --split-unit writer
"""
import argparse
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from src.data_loader import load_human_baseline  # noqa: E402
from src.paths import MANIFEST_DIR  # noqa: E402

OUT_DIR = str(MANIFEST_DIR)


def build(test_fraction: float, seed: int, split_unit: str = "essay") -> None:
    df = load_human_baseline()
    print(f"Loaded {len(df)} human essays "
          f"({df['id'].nunique()} unique students).")
    print("NOTE: 'id' is per-STUDENT (each student wrote both topics); "
          "the per-row unique key is (id, topic) / file_name.")

    df = df.reset_index(drop=True)
    df["split"] = "pool"
    if split_unit == "writer":
        for region, g in df.groupby("region"):
            writers = g["id"].drop_duplicates().sort_values()
            n_test = int(round(len(writers) * test_fraction))
            sampled = writers.sample(n=n_test, random_state=seed)
            df.loc[df["id"].isin(set(sampled)), "split"] = "test"
    else:
        for (region, topic), g in df.groupby(["region", "topic"]):
            n_test = int(round(len(g) * test_fraction))
            sampled = g.sample(n=n_test, random_state=seed).index
            df.loc[sampled, "split"] = "test"

    manifest_path = os.path.join(OUT_DIR, "split_manifest.csv")
    pool_path = os.path.join(OUT_DIR, "exemplar_pool.csv")

    df.to_csv(manifest_path, index=False, encoding="utf_8_sig")
    df[df["split"] == "pool"].to_csv(pool_path, index=False,
                                     encoding="utf_8_sig")

    n_test = (df["split"] == "test").sum()
    n_pool = (df["split"] == "pool").sum()
    print(f"Split: test={n_test}  pool={n_pool}  (total={len(df)})")
    print("\nPer (region, topic) cell sizes [test/pool]:")
    pivot = df.pivot_table(
        index="region", columns="topic", values="file_name",
        aggfunc="count", fill_value=0,
    )
    print(pivot.to_string())

    # Sanity checks
    assert n_test + n_pool == len(df), "split must partition the corpus"
    if split_unit == "writer":
        assert not (set(df.loc[df["split"] == "test", "id"]) &
                    set(df.loc[df["split"] == "pool", "id"])), \
            "writer IDs must be disjoint across split"
    cell_min = df.groupby(["region", "topic"])["split"].apply(
        lambda s: (s == "test").sum()
    ).min()
    assert cell_min >= 10, f"smallest test cell is only {cell_min} rows"
    print(f"\nSmallest per-cell test count: {cell_min} (>= 10 OK)")
    print(f"✅ Wrote {manifest_path}")
    print(f"✅ Wrote {pool_path}")
    print(f"Split parameters: test_fraction={test_fraction}, seed={seed}, split_unit={split_unit}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split-unit", choices=["essay", "writer"], default="essay")
    args = ap.parse_args()
    build(args.test_fraction, args.seed, args.split_unit)
