"""Precise artifact scan v2 — low false-positive:
  R: true refusal (model refuses to role-play) — exact phrases
  T: truncated (last sentence has no terminal punctuation AND text ends
     mid-clause) — heuristic: ends with lowercase letter/word, no .!?"
  J: JSON/metadata leak at start
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

# true refusal: model breaks character
REFUSAL_RE = re.compile(
    r"\b(?:as an ai|as a language model|as an ai assistant|i am an ai|"
    r"i'm an ai|i am a language model|i cannot (?:write|provide|assist|help|"
    r"complete|fulfill)|i can'?t (?:write|provide|assist|help|complete|"
    r"fulfill)|i won'?t write|sorry,? i (?:can'?t|cannot|won'?t) (?:write|"
    r"help)|i don'?t (?:write|feel comfortable (?:writing|providing))|"
    r"unfortunately,? i (?:can'?t|cannot) (?:write|provide|help)|"
    r"i apologize,? but i (?:can'?t|cannot) (?:write|help|provide)|"
    r"it would be inappropriate for me|i cannot engage in)\b",
    re.I)

# truncated: ends with lowercase letter or word (no terminal punctuation),
# and is longer than 15 words (real essays end with punctuation)
TRUNC_RE = re.compile(r"[a-z]$")

# JSON/metadata leak at very start
JSON_RE = re.compile(r'^\s*[\{\["]')

rows = []
for m in MODELS:
    for c in CONDS:
        df = pd.read_csv(os.path.join(OUT, f"Corpus_{m}_{c}.csv"))
        for _, r in df.iterrows():
            t = str(r["generated_text"]).strip()
            if not t:
                continue
            refusal = bool(REFUSAL_RE.search(t))
            nw = len(t.split())
            trunc = bool(TRUNC_RE.search(t)) and nw > 15
            json_leak = bool(JSON_RE.match(t)) and nw > 3
            if refusal or trunc or json_leak:
                rows.append({"model": m, "cond": c, "region": r["region"],
                             "topic": r["topic"], "id": r["source_id"],
                             "refusal": refusal, "trunc": trunc,
                             "json": json_leak, "n_words": nw,
                             "tail": t[-120:]})

dfr = pd.DataFrame(rows)
print(f"TOTAL: {len(dfr)}")
if len(dfr):
    print("\nby type:", "refusal", int(dfr.refusal.sum()),
          "| trunc", int(dfr.trunc.sum()), "| json", int(dfr.json.sum()))
    print("\nby model x cond:")
    print(dfr.groupby(["model", "cond"]).size().unstack(fill_value=0))
    print("\nby region:")
    print(dfr.groupby("region").size())
    print("\nsamples:")
    for _, r in dfr.head(40).iterrows():
        k = "R" if r.refusal else ("T" if r.trunc else "J")
        print(f"[{k}] {r['model']}/{r['cond']} {r['region']} {r['id']} "
              f"{r['n_words']}w ...{r['tail']!r}")
