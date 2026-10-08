"""Final merge of neosca features: A, B, C, C0-C3 CSVs -> neosca_all.csv
with (key|model|condition) dedup, then verify coverage vs features_all.

Also merges into features_all.parquet as MLT + Clause_per_Sentence columns.
"""
import sys, os, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
from src.paths import FEATURE_DIR, RESULTS_DIR

RES = str(RESULTS_DIR)
ALL = 140000

# 1. collect all neosca CSVs
files = (glob.glob(os.path.join(RES, "neosca_A.csv"))
         + glob.glob(os.path.join(RES, "neosca_B.csv"))
         + glob.glob(os.path.join(RES, "neosca_C.csv"))
         + glob.glob(os.path.join(RES, "neosca_C_C*.csv")))
print("neosca files:", [os.path.basename(f) for f in files])

dfs = []
for f in files:
    df = pd.read_csv(f)
    dfs.append(df)
    print(f"  {os.path.basename(f)}: {len(df)} rows")
raw = pd.concat(dfs, ignore_index=True)
print(f"\nraw total (before dedup): {len(raw)}")

# 2. dedup by (key|model|condition)
raw["dk"] = raw["key"] + "|" + raw["model"] + "|" + raw["condition"]
before = len(raw)
raw = raw.drop_duplicates(subset="dk", keep="last").drop(columns="dk")
print(f"after dedup: {len(raw)} (removed {before - len(raw)} dup)")

# 3. coverage check vs features_all keys
feat = pd.read_parquet(FEATURE_DIR / "features_all.parquet",
                       columns=["key", "model", "condition"])
feat["dk"] = feat["key"] + "|" + feat["model"] + "|" + feat["condition"]
raw["dk"] = raw["key"] + "|" + raw["model"] + "|" + raw["condition"]
missing = set(feat["dk"]) - set(raw["dk"])
extra = set(raw["dk"]) - set(feat["dk"])
print(f"\nfeatures_all keys: {len(feat)}")
print(f"neosca unique keys: {len(raw)}")
print(f"missing (feature keys without neosca): {len(missing)}")
if missing:
    # show first few missing by model/condition
    m = feat[feat["dk"].isin(list(missing)[:10])]
    print(m[["model", "condition"]].value_counts().head())
print(f"extra (neosca keys not in features): {len(extra)}")

# 4. save neosca_all
out = os.path.join(RES, "neosca_all.csv")
raw.to_csv(out, index=False, encoding="utf-8")
print(f"\nsaved -> {out} ({len(raw)} rows)")

# 5. merge into features_all.parquet (keep ALL original columns)
feat = pd.read_parquet(FEATURE_DIR / "features_all.parquet")
neosca_cols = raw[["key", "model", "condition", "MLT",
                   "Clause_per_Sentence"]].rename(
    columns={"key": "k2", "model": "m2", "condition": "c2"})
feat2 = feat.merge(neosca_cols,
                   left_on=["key", "model", "condition"],
                   right_on=["k2", "m2", "c2"], how="left")
feat2.drop(columns=["k2", "m2", "c2"], inplace=True)
print(f"\nmerged features_all: {len(feat2)} rows x {len(feat2.columns)} cols")
print(f"MLT coverage: {feat2['MLT'].notna().sum()} / {len(feat2)}")
print(f"C/S coverage: {feat2['Clause_per_Sentence'].notna().sum()} / {len(feat2)}")
feat2.to_parquet(FEATURE_DIR / "features_all_with_neosca.parquet",
                 index=False)
print("saved -> features_all_with_neosca.parquet")
