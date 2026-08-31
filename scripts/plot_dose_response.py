"""Dose-response figures: feature mean vs condition (C0->C3), per model,
with 95% bootstrap CI, human baseline as horizontal line.

Output: data/results/figures/dose_response_<feature>.png
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = "data/results"
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)

bs = pd.read_csv(os.path.join(RES, "bootstrap_summary.csv"))
MODELS = ["GPT-5.6-Luna", "Gemini-2.5-Flash", "Qwen-3.7-Plus",
          "DeepSeek-V4-Flash", "Llama-4-Maverick", "MiniMax-M3"]
CONDS = ["zero_shot", "one_shot", "few_shot", "cot"]
COND_LABEL = {"zero_shot": "C0\nzero", "one_shot": "C1\none",
              "few_shot": "C2\nfew", "cot": "C3\nCoT"}
COLORS = {"GPT-5.6-Luna": "#1f77b4", "Gemini-2.5-Flash": "#ff7f0e",
          "Qwen-3.7-Plus": "#2ca02c", "DeepSeek-V4-Flash": "#d62728",
          "Llama-4-Maverick": "#9467bd", "MiniMax-M3": "#8c564b"}

FEATURES = ["AVD_Combined", "MTLD", "MDD", "Nominalization_Rate",
            "Passive_Ratio", "Pronoun_Ratio", "Discourse_Density",
            "Grammar_Acceptability", "Grammar_Error_Rate", "MLT",
            "Clause_per_Sentence", "Top20_Word_Share", "Hapax_Ratio",
            "Lexical_Cohesion", "Referential_Density", "Sentence_Length",
            "Tree_Depth", "CEFR_A1", "CEFR_B1", "CEFR_C1"]

human = bs[bs["model"] == "HUMAN"].set_index("feature")

for feat in FEATURES:
    if feat not in human.index:
        continue
    hrow = human.loc[feat]
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(CONDS))
    for m in MODELS:
        sub = bs[(bs["model"] == m) & (bs["feature"] == feat)]
        if sub.empty:
            continue
        sub = sub.set_index("condition").reindex(CONDS)
        ax.errorbar(x, sub["mean"], yerr=[
            sub["mean"] - sub["ci_low"], sub["ci_high"] - sub["mean"]],
            marker="o", capsize=3, lw=1.5, color=COLORS[m],
            label=m.replace("-5.6-Luna", "").replace("-2.5-Flash", "")
                     .replace("-3.7-Plus", "").replace("-V4-Flash", "")
                     .replace("-4-Maverick", "").replace("-M3", ""))
    # human baseline
    ax.axhline(hrow["mean"], color="black", ls="--", lw=2,
               label=f"Human ({hrow['mean']:.2f})")
    ax.fill_between(x, hrow["ci_low"], hrow["ci_high"], color="black",
                    alpha=0.08)
    ax.set_xticks(x)
    ax.set_xticklabels([COND_LABEL[c] for c in CONDS])
    ax.set_xlabel("Prompt intervention")
    ax.set_ylabel(feat)
    ax.set_title(f"{feat} — dose response")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"dose_response_{feat}.png"), dpi=150)
    plt.close(fig)
    print(f"saved dose_response_{feat}.png", flush=True)

print("\nDONE — figures in", FIG)
