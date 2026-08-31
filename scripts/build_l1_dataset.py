"""Phase 5: build L1 classification datasets (36 classifiers).

Matrix design (user-approved):
   6 target LLMs
   ├─ per-condition: 24 classifiers (each model x one of C0-C3)
   ├─ equal-mix:      6 classifiers (random 1120, 4 conditions pooled)
   └─ full-mix:       6 classifiers (all 4480, 4 conditions pooled)

For each classifier:
   - TRAIN = generated texts (Sim) or human pool (Oracle), Mirror-matched to
     human per-region target counts (Biber/1-to-1 fair comparison).
   - Split 85/15 into train/dev inside (early stopping, seed 42).
   - TEST  = ICNALE test split (1120, held-out, disjoint from any training).

Mirror logic (from original paper, cell 14):
   target_distribution = human pool count per region.
   Each source is downsampled to match those EXACT per-region counts.
   => class priors identical across all sources (only data quality differs).

Output: data/results/l1_dataset/ (labels + splits), keyed by classifier name.
"""
import sys, os, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RES = "data/results"
DS = os.path.join(RES, "l1_dataset")
os.makedirs(DS, exist_ok=True)
SEED = 42

MODELS = ["GPT-5.6-Luna", "Gemini-2.5-Flash", "Qwen-3.7-Plus",
          "DeepSeek-V4-Flash", "Llama-4-Maverick", "MiniMax-M3"]
CONDS = ["zero_shot", "one_shot", "few_shot", "cot"]

# 1. human split
sm = pd.read_csv("data/interim/split_manifest.csv")
human_pool = sm[sm.split == "pool"].copy()   # 4480: Oracle train source + mirror target
human_test = sm[sm.split == "test"].copy()   # 1120: gold test (disjoint)
print(f"human pool: {len(human_pool)}, test: {len(human_test)}")

# Mirror target: per-region counts in human pool
target_dist = human_pool["region"].value_counts().to_dict()
print("mirror target (per region):", target_dist)

# map generated corpora to (model, condition) -> df with region + text
gen_cache = {}
def get_gen(model, cond):
    key = (model, cond)
    if key not in gen_cache:
        df = pd.read_csv(
            f"data/raw/LLM_Generations/Corpus_{model}_{cond}.csv")
        gen_cache[key] = df
    return gen_cache[key]


def mirror_matched(df, label_col="region"):
    """Downsample df so each region matches target_dist (class-prior mirror)."""
    pieces = []
    for region, cnt in target_dist.items():
        sub = df[df[label_col] == region]
        if len(sub) < cnt:
            sub = sub.sample(frac=1, random_state=SEED)  # use all if under
        else:
            sub = sub.sample(n=cnt, random_state=SEED)
        pieces.append(sub)
    return pd.concat(pieces).sample(frac=1, random_state=SEED).reset_index(drop=True)


def make_dataset(name, df, region_col="region", text_col="generated_text"):
    """df: full corpus for this source. Returns dict with train/dev/test splits
    as (X, y) lists + label2id."""
    matched = mirror_matched(df, region_col)
    labels = sorted(matched[region_col].unique())
    label2id = {l: i for i, l in enumerate(labels)}
    # 85/15 stratified train/dev
    train, dev = train_test_split(
        matched, test_size=0.15, random_state=SEED,
        stratify=matched[region_col])
    test = human_test  # gold test, fixed
    return {
        "name": name,
        "train_X": train[text_col].tolist(),
        "train_y": [label2id[l] for l in train[region_col]],
        "dev_X": dev[text_col].tolist(),
        "dev_y": [label2id[l] for l in dev[region_col]],
        "test_X": test["text"].tolist(),
        "test_y": [label2id[l] for l in test["region"]],
        "label2id": label2id,
    }


def save_dataset(ds):
    path = os.path.join(DS, f"{ds['name']}.parquet")
    # store as DataFrame with split column (compact, single file)
    rows = []
    for split, X, y in [("train", ds["train_X"], ds["train_y"]),
                        ("dev", ds["dev_X"], ds["dev_y"]),
                        ("test", ds["test_X"], ds["test_y"])]:
        for x, yy in zip(X, y):
            rows.append({"text": x, "label": yy, "split": split})
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


# 2. Oracle baseline (Real->Real): train on human pool
oracle_ds = make_dataset("oracle_human", human_pool.rename(
    columns={"text": "generated_text"}), region_col="region",
    text_col="generated_text")
save_dataset(oracle_ds)
print(f"oracle: train {len(oracle_ds['train_X'])}, dev {len(oracle_ds['dev_X'])}, "
      f"test {len(oracle_ds['test_X'])}")

# 3. Per-condition (24): each model x each condition, trained on that
#    condition's generated texts, mirror-matched to human pool.
for model in MODELS:
    for cond in CONDS:
        gen = get_gen(model, cond)
        # use human-test keys to EXCLUDE any test overlap (defensive; none by design)
        name = f"sim_{model}_{cond}"
        ds = make_dataset(name, gen)
        save_dataset(ds)
        print(f"{name}: train {len(ds['train_X'])}")
    # 4. Equal-mix (1 per model): random 1120 across 4 conditions
    cond_dfs = [get_gen(model, c) for c in CONDS]
    eq = pd.concat(cond_dfs).sample(n=1120, random_state=SEED).reset_index(drop=True)
    name = f"sim_{model}_eqmix"
    ds = make_dataset(name, eq)
    save_dataset(ds)
    print(f"{name}: train {len(ds['train_X'])}")
    # 5. Full-mix (1 per model): all 4480 across 4 conditions
    full = pd.concat(cond_dfs, ignore_index=True)
    name = f"sim_{model}_fullmix"
    ds = make_dataset(name, full)
    save_dataset(ds)
    print(f"{name}: train {len(ds['train_X'])}")

print(f"\nDONE — datasets in {DS}")

# manifest
manifest = {"mirror_target": target_dist, "test_size": len(human_test),
            "seed": SEED, "n_classifiers": 1 + 6*6}
with open(os.path.join(DS, "manifest.json"), "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)
print("manifest saved")
