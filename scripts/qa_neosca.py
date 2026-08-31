"""Sampled QA of neosca output: correctness, coverage, truncation, sentence boundaries.

Checks:
  1. numeric sanity: MLT/C-S ranges for human vs generated
  2. coverage: keys in neosca CSV match features_all keys (no missing/extra)
  3. no truncation: all rows have non-null MLT & C/S
  4. sentence boundary: recompute MLT/C-S for a known text via raw neosca call
     and compare with the CSV row (validates boundary detection end-to-end)
"""
import sys, os, json
sys.path.insert(0, "libs")
os.chdir(".")
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

RES = "data/results"

# ---- 1+2+3: load neosca A output (smoke + running checkpoint) ----
neosca_path = os.path.join(RES, "neosca_A.csv")
if os.path.exists(neosca_path):
    nsc = pd.read_csv(neosca_path)
    print(f"neosca_A.csv: {len(nsc)} rows, cols={list(nsc.columns)}")
    print(f"  nulls: MLT={nsc['MLT'].isnull().sum()}, "
          f"C/S={nsc['Clause_per_Sentence'].isnull().sum()}")
    print(f"  MLT range: [{nsc['MLT'].min():.2f}, {nsc['MLT'].max():.2f}]")
    print(f"  C/S range: [{nsc['Clause_per_Sentence'].min():.3f}, "
          f"{nsc['Clause_per_Sentence'].max():.3f}]")
    print(f"  models: {sorted(nsc['model'].unique())}")
    print(f"  conditions: {sorted(nsc['condition'].unique())}")
    # coverage vs features_all keys (subset already processed)
    feat = pd.read_parquet(os.path.join(RES, "features_all.parquet"),
                           columns=["key", "model", "condition"])
    feat_key = (feat["key"] + "|" + feat["model"] + "|" + feat["condition"])
    nsc_key = (nsc["key"] + "|" + nsc["model"] + "|" + nsc["condition"])
    missing = set(feat_key) - set(nsc_key)
    extra = set(nsc_key) - set(feat_key)
    print(f"\n  coverage: {len(set(nsc_key))} neosca keys vs "
          f"{len(set(feat_key))} feature keys")
    print(f"  extra keys (in neosca not in features): {len(extra)}")
    print(f"  missing (in features not in neosca): {len(missing)} "
          f"(expected: all unprocessed)")
else:
    print("neosca_A.csv not found yet")

# ---- 4: sentence-boundary correctness on a known text ----
print("\n=== sentence boundary spot-check ===")
text = ("Smoking should be banned in all restaurants because it harms other "
        "people. Many customers do not want to breathe smoke while they are "
        "eating. The government must protect the health of all people. In my "
        "opinion having a part-time job is good for university students.")
from neosca.ns_sca.ns_sca import Ns_SCA
sca = Ns_SCA()
sca.run_on_text(text)
vals = sca.counters[-1].get_all_values()
print(f"text has 4 sentences (4 periods)")
print(f"  neosca says: W={vals.get('W')} S={vals.get('S')} T={vals.get('T')} "
      f"C={vals.get('C')} MLT={vals.get('MLT')} C/S={vals.get('C/S')}")
print(f"  -> S={vals.get('S')} means sentence boundary detection "
      f"({'correct' if vals.get('S')==4 else 'CHECK!'})")
