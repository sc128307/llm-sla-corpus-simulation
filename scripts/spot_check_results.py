"""Spot-check Phase 4 results: dose-response patterns for key features."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

bs = pd.read_csv("data/results/bootstrap_summary.csv")
jsd = pd.read_csv("data/results/jensen_shannon.csv")
eff = pd.read_csv("data/results/effect_sizes.csv")

print("=== Human baseline feature values (main reference) ===")
h = bs[bs["model"] == "HUMAN"]
for _, r in h[h["feature"].isin(["AVD_Combined", "MTLD", "MDD",
                                 "Grammar_Error_Rate", "MLT",
                                 "Clause_per_Sentence"])].iterrows():
    print(f"  {r['feature']:<22} mean={r['mean']:.2f} "
          f"CI[{r['ci_low']:.2f}, {r['ci_high']:.2f}]")

print("\n=== Dose-response: GPT (C0->C3) vs Human ===")
gpt = bs[(bs["model"] == "GPT-5.6-Luna")]
for feat in ["AVD_Combined", "MTLD", "Grammar_Error_Rate", "MLT"]:
    row_h = h[h["feature"] == feat].iloc[0]
    print(f"\n  {feat} (human={row_h['mean']:.2f}):")
    for cond in ["zero_shot", "one_shot", "few_shot", "cot"]:
        r = gpt[(gpt["condition"] == cond) & (gpt["feature"] == feat)]
        if len(r):
            r = r.iloc[0]
            outside = "✓偏离" if (r["ci_low"] > row_h["ci_high"] or
                                   r["ci_high"] < row_h["ci_low"]) else "·重叠"
            print(f"    {cond:<10} mean={r['mean']:.2f} "
                  f"CI[{r['ci_low']:.2f},{r['ci_high']:.2f}] {outside}")

print("\n=== JSD summary (GPT across conditions, 4 key feats) ===")
gjsd = jsd[jsd["model"] == "GPT-5.6-Luna"]
for feat in ["AVD_Combined", "MTLD", "Grammar_Error_Rate"]:
    vals = gjsd[gjsd["feature"] == feat]
    line = "  ".join(f"{c}:{v:.3f}" for c, v in
                     zip(vals["condition"], vals["JSD"]))
    print(f"  {feat:<20} {line}")

print("\n=== Cohen's d: which models deviate most (Grammar_Error_Rate) ===")
ge = eff[eff["feature"] == "Grammar_Error_Rate"]
for cond in ["zero_shot", "cot"]:
    sub = ge[ge["condition"] == cond].sort_values("cohens_d")
    line = "  ".join(f"{m.split('-')[0]}:{d:.1f}" for m, d in
                     zip(sub["model"], sub["cohens_d"]) if m != "HUMAN")
    print(f"  {cond}: {line}")
