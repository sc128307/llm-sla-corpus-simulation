"""Build leak-controlled L1 classification datasets.

The primary dataset is a 10-class L2-region task. A separate 11-class
dataset, including ENS, is built for sensitivity analysis. The generated
corpus is never modified; only local dataset parquet files are written.

For each task:
  * per-condition sources are mirror-matched to the human-pool regional prior;
  * ``eqmix`` contains one quarter of the target count from each condition in
    every region (equal condition contribution and human-matched region prior);
  * ``fullmix`` contains all four conditions, so it is four times larger while
    retaining the same regional prior;
  * human test keys are removed from generated sources before sampling;
  * parquet files retain key/writer/region/topic metadata for clustered QA.
"""
import sys
import os
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

try:
    from sklearn.model_selection import train_test_split
except ModuleNotFoundError:  # allow dataset construction in the lean audit env
    def train_test_split(data, test_size, random_state, stratify):
        """Small stratified fallback for the fixed 85/15 split."""
        rng = np.random.default_rng(random_state)
        train_parts, dev_parts = [], []
        labels = pd.Series(stratify, index=data.index)
        for label in sorted(labels.unique()):
            idx = labels[labels == label].index.to_numpy().copy()
            rng.shuffle(idx)
            n_dev = max(1, int(np.ceil(len(idx) * test_size)))
            dev_parts.append(data.loc[idx[:n_dev]])
            train_parts.append(data.loc[idx[n_dev:]])
        train = pd.concat(train_parts).sample(frac=1, random_state=random_state)
        dev = pd.concat(dev_parts).sample(frac=1, random_state=random_state)
        return train, dev

from src.paths import RESULTS_DIR, SPLIT_MANIFEST_PATH, generated_path  # noqa: E402

SEED = 42
MODELS = [
    "GPT-5.6-Luna", "Gemini-2.5-Flash", "Qwen-3.7-Plus",
    "DeepSeek-V4-Flash", "Llama-4-Maverick", "MiniMax-M3",
]
CONDS = ["zero_shot", "one_shot", "few_shot", "cot"]  # cot = manuscript C3

PRIMARY_DIR = RESULTS_DIR / "l1_dataset"
ENS_DIR = RESULTS_DIR / "l1_dataset_ens"


def key_series(df):
    """Return stable essay key and writer id for human/generated rows."""
    if "source_id" in df.columns:
        writer = df["source_id"].astype(str)
    else:
        writer = df["id"].astype(str)
    key = writer + "|" + df["topic"].astype(str)
    return key, writer


def text_series(df, text_col):
    return df[text_col].astype(str)


def mirror_matched(df, target_dist, label_col="region"):
    """Downsample each region to the exact target distribution."""
    pieces = []
    for region, count in target_dist.items():
        sub = df[df[label_col] == region]
        if len(sub) < count:
            raise ValueError(
                f"source has {len(sub)} rows for {region}, "
                f"but target requires {count}; refusing silent upsampling"
            )
        pieces.append(sub.sample(n=count, random_state=SEED))
    return pd.concat(pieces, ignore_index=True).sample(
        frac=1, random_state=SEED
    ).reset_index(drop=True)


def grouped_train_dev(df, test_size=0.15):
    """Split by source writer so paired topics/conditions cannot cross splits."""
    work = df.copy()
    _, writer = key_series(work)
    work["_writer_id_for_split"] = writer
    groups = work[["_writer_id_for_split", "region"]].drop_duplicates()
    train_groups, dev_groups = train_test_split(
        groups,
        test_size=test_size,
        random_state=SEED,
        stratify=groups["region"],
    )
    train_ids = set(train_groups["_writer_id_for_split"])
    dev_ids = set(dev_groups["_writer_id_for_split"])
    train = work[work["_writer_id_for_split"].isin(train_ids)].drop(
        columns=["_writer_id_for_split"]
    )
    dev = work[work["_writer_id_for_split"].isin(dev_ids)].drop(
        columns=["_writer_id_for_split"]
    )
    return train.sample(frac=1, random_state=SEED), dev.sample(
        frac=1, random_state=SEED
    )


def equal_condition_mix(cond_dfs, target_dist):
    """Build an equal-condition mix with the human region prior."""
    if any(count % len(CONDS) for count in target_dist.values()):
        raise ValueError("target region counts must be divisible by 4 for eqmix")
    pieces = []
    for cond_idx, cond in enumerate(CONDS):
        df = cond_dfs[cond]
        for region, count in target_dist.items():
            n = count // len(CONDS)
            sub = df[df["region"] == region]
            if len(sub) < n:
                raise ValueError(
                    f"{cond} has {len(sub)} rows for {region}, needs {n}"
                )
            pieces.append(sub.sample(n=n, random_state=SEED + cond_idx))
    return pd.concat(pieces, ignore_index=True).sample(
        frac=1, random_state=SEED
    ).reset_index(drop=True)


def make_dataset(name, source_df, human_test, target_dist, text_col):
    """Return train/dev/test rows with labels and retained audit metadata."""
    matched = mirror_matched(source_df, target_dist)
    labels = sorted(target_dist)
    label2id = {region: i for i, region in enumerate(labels)}
    train, dev = grouped_train_dev(matched)

    def format_rows(df, split):
        key, writer = key_series(df)
        actual_text_col = text_col if text_col in df.columns else "text"
        return pd.DataFrame({
            "text": text_series(df, actual_text_col),
            "label": df["region"].map(label2id).astype(int),
            "split": split,
            "key": key,
            "writer_id": writer,
            "region": df["region"].astype(str),
            "topic": df["topic"].astype(str),
        }).reset_index(drop=True)

    return pd.concat([
        format_rows(train, "train"),
        format_rows(dev, "dev"),
        format_rows(human_test, "test"),
    ], ignore_index=True), label2id


def write_task(task_dir, include_ens, split_manifest, gen_cache):
    task_dir.mkdir(parents=True, exist_ok=True)
    target_regions = sorted(split_manifest["region"].unique())
    if not include_ens:
        target_regions = [r for r in target_regions if r != "ENS"]

    human_pool = split_manifest[
        (split_manifest["split"] == "pool")
        & split_manifest["region"].isin(target_regions)
    ].copy()
    human_test = split_manifest[
        (split_manifest["split"] == "test")
        & split_manifest["region"].isin(target_regions)
    ].copy()
    target_dist = human_pool["region"].value_counts().sort_index().to_dict()
    test_keys = set(
        human_test["id"].astype(str) + "|" + human_test["topic"].astype(str)
    )

    def get_gen(model, cond):
        cache_key = (model, cond, include_ens)
        if cache_key not in gen_cache:
            df = pd.read_csv(generated_path(model, cond))
            row_keys = df["source_id"].astype(str) + "|" + df["topic"].astype(str)
            df = df.loc[~row_keys.isin(test_keys)].copy()
            df = df[df["region"].isin(target_regions)].copy()
            gen_cache[cache_key] = df
        return gen_cache[cache_key]

    def save(name, source_df, source_text_col):
        rows, _ = make_dataset(
            name, source_df, human_test, target_dist, source_text_col
        )
        rows.to_parquet(task_dir / f"{name}.parquet", index=False)

    # Human reference.
    human_source = human_pool.rename(columns={"text": "generated_text"})
    save("oracle_human", human_source, "generated_text")

    # Per-condition sources plus two optional pooled source designs.
    for model in MODELS:
        for cond in CONDS:
            save(f"sim_{model}_{cond}", get_gen(model, cond), "generated_text")

        cond_dfs = {cond: get_gen(model, cond) for cond in CONDS}
        eq = equal_condition_mix(cond_dfs, target_dist)
        save(f"sim_{model}_eqmix", eq, "generated_text")

        # Fullmix is intentionally all four conditions. It is four times the
        # per-condition source size, not a second mirror-matched sample.
        full = pd.concat(list(cond_dfs.values()), ignore_index=True)
        labels = sorted(target_dist)
        label2id = {region: i for i, region in enumerate(labels)}
        train, dev = grouped_train_dev(full)

        def format_full(df, split):
            key, writer = key_series(df)
            actual_text_col = "generated_text" if "generated_text" in df.columns else "text"
            return pd.DataFrame({
                "text": text_series(df, actual_text_col),
                "label": df["region"].map(label2id).astype(int),
                "split": split,
                "key": key,
                "writer_id": writer,
                "region": df["region"].astype(str),
                "topic": df["topic"].astype(str),
            })

        full_rows = pd.concat([
            format_full(train, "train"),
            format_full(dev, "dev"),
            format_full(human_test, "test"),
        ], ignore_index=True)
        full_rows.to_parquet(
            task_dir / f"sim_{model}_fullmix.parquet", index=False
        )

    manifest = {
        "task": "11_class_with_ens_sensitivity" if include_ens else "10_class_l2_primary",
        "regions": target_regions,
        "include_ens": include_ens,
        "mirror_target": target_dist,
        "human_pool_size": int(len(human_pool)),
        "human_test_size": int(len(human_test)),
        "seed": SEED,
        "heldout_test_key_filter": True,
        "n_classifiers": 1 + 6 * 6,
        "conditions": CONDS,
        "condition_label_note": "cot is retained as the file key for manuscript C3 planning instruction",
        "eqmix_design": "per-region human target count divided equally across four conditions; total source size equals one condition",
        "fullmix_design": "all four heldout-filtered conditions concatenated; source size is four times one condition",
        "split_rule": "85/15 stratified writer-group train/dev; fixed human test",
        "metadata_columns": ["key", "writer_id", "region", "topic"],
    }
    with open(task_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"built {manifest['task']}: {task_dir}")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def main():
    sm = pd.read_csv(SPLIT_MANIFEST_PATH)
    gen_cache = {}
    # Replace only derived dataset directories. Source corpora are untouched.
    for task_dir in (PRIMARY_DIR, ENS_DIR):
        if task_dir.exists():
            shutil.rmtree(task_dir)
    write_task(PRIMARY_DIR, include_ens=False, split_manifest=sm, gen_cache=gen_cache)
    write_task(ENS_DIR, include_ens=True, split_manifest=sm, gen_cache=gen_cache)


if __name__ == "__main__":
    main()
