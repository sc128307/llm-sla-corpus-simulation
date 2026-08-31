"""Strict refusal-scan across ALL texts (any length): model broke character."""
import pandas as pd
import re
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

OUT = "data/raw/LLM_Generations"
MODELS = ["Gemini-2.5-Flash", "Qwen-3.7-Plus", "DeepSeek-V4-Flash",
          "Llama-4-Maverick", "MiniMax-M3", "GPT-5.6-Luna"]
CONDS = ["zero_shot", "one_shot", "few_shot", "cot"]

REFUSAL = re.compile(
    r"\b(as an ai|i am an ai|i'm an ai|as a language model|"
    r"i (?:am|'m) a language model|i cannot (?:write|provide|assist|help|"
    r"complete|fulfill|answer)|i can'?t (?:write|provide|assist|help|"
    r"complete|fulfill|answer)|i won'?t write|i will not write|"
    r"i apologize,? but|i'm sorry,? but i (?:can'?t|cannot|won'?t)|"
    r"sorry,? i (?:can'?t|cannot)|unfortunately,? i (?:can'?t|cannot)|"
    r"it would be inappropriate for me|i am not able to (?:write|provide)|"
    r"i'?m not able to (?:write|provide)|against (?:my|our) (?:policy|"
    r"guidelines)|i cannot engage in|not (?:possible|allowed) for me to "
    r"(?:write|provide|complete))\b",
    re.I)

rows = []
for m in MODELS:
    for c in CONDS:
        df = pd.read_csv(os.path.join(OUT, f"Corpus_{m}_{c}.csv"))
        for _, r in df.iterrows():
            t = str(r["generated_text"])
            if REFUSAL.search(t):
                rows.append({"model": m, "cond": c, "region": r["region"],
                             "topic": r["topic"], "id": r["source_id"],
                             "n_words": len(t.split()),
                             "text": t[:300]})

dfr = pd.DataFrame(rows)
print(f"TOTAL strict refusals: {len(dfr)}")
if len(dfr):
    print(dfr.groupby(["model", "cond"]).size().unstack(fill_value=0))
    for _, r in dfr.iterrows():
        print(f"[{r['model']}/{r['cond']}] {r['region']} {r['id']} "
              f"{r['n_words']}w: {r['text']!r}")
