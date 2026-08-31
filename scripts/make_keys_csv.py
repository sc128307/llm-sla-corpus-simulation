"""Generate keys.csv from features_all.parquet (run in nlp-corpus env)."""
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

feat = pd.read_parquet("data/results/features_all.parquet",
                       columns=["key", "region", "topic", "model", "condition"])
feat.to_csv("data/results/keys.csv", index=False, encoding="utf-8")
print("keys.csv:", len(feat), "rows")

# verify human_data_full.csv columns
h = pd.read_csv("data/interim/human_data_full.csv", nrows=2)
print("human_data_full columns:", list(h.columns))
