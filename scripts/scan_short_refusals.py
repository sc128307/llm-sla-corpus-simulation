"""Scan for one-sentence error/refusal outputs (no essay body).

A model may answer with a short refusal/error instead of the essay, e.g.
"I'm sorry, I cannot fulfill this request." or "I apologize, but I can't
write this essay." These are usually a single short sentence.
Detection: text is SHORT (<= 60 words) AND contains refusal/error signals.
"""
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

# refusal / error signals (anywhere in text)
SIGNAL = re.compile(
    r"\b(can'?t|cannot|won'?t|unable to|apolog|sorry|unfortunately|"
    r"refus|as an ai|language model|error|invalid|not able to|"
    r"i am not (?:able|allowed)|i'?m not (?:able|allowed)|"
    r"against (?:my|our) (?:policy|guidelines)|violates|"
    r"i don'?t (?:have|think i can)|exceeds (?:my|the)|"
    r"not (?:possible|allowed) for me|fail(?:ed|ure)?)\b",
    re.I)

rows = []
for m in MODELS:
    for c in CONDS:
        df = pd.read_csv(os.path.join(OUT, f"Corpus_{m}_{c}.csv"))
        for _, r in df.iterrows():
            t = str(r["generated_text"]).strip()
            nw = len(t.split())
            if nw == 0:
                continue
            # candidate: short text that is NOT an essay opening
            if nw <= 60 and SIGNAL.search(t):
                rows.append({"model": m, "cond": c, "region": r["region"],
                             "topic": r["topic"], "id": r["source_id"],
                             "n_words": nw, "text": t[:250]})

dfr = pd.DataFrame(rows)
print(f"TOTAL short-refusal candidates: {len(dfr)}")
if len(dfr):
    print("\nby model x cond:")
    print(dfr.groupby(["model", "cond"]).size().unstack(fill_value=0))
    print("\nby region:")
    print(dfr.groupby("region").size())
    print("\nall texts:")
    for _, r in dfr.iterrows():
        print(f"[{r['model']}/{r['cond']}] {r['region']} {r['id']} "
              f"{r['n_words']}w: {r['text']!r}")
