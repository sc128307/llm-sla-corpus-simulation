"""Sim-to-Real gap figure: L1 classification acc per model across conditions."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = "data/results"
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)

df = pd.read_csv(os.path.join(RES, "l1_classification_summary.csv"))
MODELS = ["GPT-5.6-Luna", "Gemini-2.5-Flash", "Qwen-3.7-Plus",
          "DeepSeek-V4-Flash", "Llama-4-Maverick", "MiniMax-M3"]
CONDS = ["zero_shot", "one_shot", "few_shot", "cot"]
COND_LABEL = {"zero_shot": "C0\nzero", "one_shot": "C1\none",
              "few_shot": "C2\nfew", "cot": "C3\nCoT"}
COLORS = {"GPT-5.6-Luna": "#1f77b4", "Gemini-2.5-Flash": "#ff7f0e",
          "Qwen-3.7-Plus": "#2ca02c", "DeepSeek-V4-Flash": "#d62728",
          "Llama-4-Maverick": "#9467bd", "MiniMax-M3": "#8c564b"}

oracle = df[df.classifier == "oracle_human"].iloc[0]["accuracy"] * 100
fig, ax = plt.subplots(figsize=(8, 5.5))
x = range(len(CONDS))
for m in MODELS:
    accs = []
    for c in CONDS:
        r = df[df.classifier == f"sim_{m}_{c}"]
        if len(r):
            accs.append(r.iloc[0]["accuracy"] * 100)
        else:
            accs.append(0.0)
    ax.plot(x, accs, marker="o", lw=1.8, color=COLORS[m],
            label=m.replace("-5.6-Luna", "").replace("-2.5-Flash", "")
                    .replace("-3.7-Plus", "").replace("-V4-Flash", "")
                    .replace("-4-Maverick", "").replace("-M3", ""))
ax.axhline(oracle, color="black", ls="--", lw=2,
           label=f"Oracle (Human {oracle:.1f}%)")
ax.axhline(100/11, color="gray", ls=":", lw=1.5, label="Chance (9.1%)")
ax.set_xticks(x)
ax.set_xticklabels([COND_LABEL[c] for c in CONDS])
ax.set_xlabel("Prompt intervention")
ax.set_ylabel("L1 classification accuracy (%)")
ax.set_title("Sim-to-Real gap: L1 identification across conditions")
ax.legend(fontsize=7, ncol=2)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "sim_to_real_gap.png"), dpi=150)
print(f"saved sim_to_real_gap.png (oracle={oracle:.1f}%)")

# also summary table
print("\n=== Sim-to-Real gap vs Oracle (percentage points) ===")
oracle_dec = df[df.classifier == "oracle_human"].iloc[0]["accuracy"]
for m in MODELS:
    cm = [c for c in df.classifier if c.startswith(f"sim_{m}_")]
    sub = df[df.classifier.isin(cm)]
    sim_max = sub["accuracy"].max()
    sim_cond = sub.loc[sub["accuracy"].idxmax(), "classifier"].replace(
        f"sim_{m}_", "")
    gap_pts = (oracle_dec - sim_max) * 100
    print(f"  {m:<20} best sim acc={sim_max*100:.1f}% "
          f"({sim_cond}) gap={gap_pts:.1f} pts")
