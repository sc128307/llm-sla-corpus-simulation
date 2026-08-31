"""Merge features_A + features_B, validate full coverage."""
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

A = pd.read_parquet("data/results/features_A.parquet")
B = pd.read_parquet("data/results/features_B.parquet")
print(f"A: {len(A)} rows | B: {len(B)} rows")
assert len(A) == 72800, f"A expected 72800, got {len(A)}"
assert len(B) == 67200, f"B expected 67200, got {len(B)}"

df = pd.concat([A, B], ignore_index=True)
df.to_parquet("data/results/features_all.parquet", index=False)
print(f"\nMERGED: {len(df)} rows -> data/results/features_all.parquet")

# validation
print("\n=== coverage by model x condition ===")
ct = df.groupby(["model", "condition"]).size().unstack(fill_value=0)
print(ct.to_string())

# expected: 24 files x 5600 + human 5600 = 140,000
expected = 6 * 4 * 5600 + 5600
print(f"\ntotal: {len(df)} (expected {expected})")
print("duplicate keys:", df.duplicated(subset=["key", "model", "condition"]).sum())
print("nulls:", df.isnull().sum().sum())

# feature sanity
num = df.select_dtypes(include="number")
print("\nfeature ranges (min/max):")
for c in num.columns:
    print(f"  {c}: [{num[c].min():.3f}, {num[c].max():.3f}]")
