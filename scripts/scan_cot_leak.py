"""Scan cot files for reasoning leakage in output (violates C3 no-reasoning
requirement and the corpus-wide no-internal-reasoning rule)."""
import pandas as pd
import re
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

OUT = "data/raw/LLM_Generations"
MODELS = ["Gemini-2.5-Flash", "Qwen-3.7-Plus", "DeepSeek-V4-Flash",
          "Llama-4-Maverick", "MiniMax-M3", "GPT-5.6-Luna"]

# reasoning-leak markers: chain-of-thought scaffolding in output
LEAK = re.compile(
    r"\b(constraint checklist|confidence score|strategiz|step \d|"
    r"reasoning|i will now (?:write|consider)|let'?s (?:think|consider|"
    r"reason)|thinking step|first,? (?:let'?s|i will) consider|"
    r"to do this well|i need to consider|plan:|reasoning process|"
    r"internal reasoning|as i reason|before writing|draft:|"
    r"outline:|checklist|score: \d/\d|here'?s my (?:plan|reasoning)|"
    r"my reasoning|analysis:|assess(?:ment)?|considerations:|"
    r"step-by-step|chain of thought|reasoning steps)\b",
    re.I)

rows = []
for m in MODELS:
    df = pd.read_csv(os.path.join(OUT, f"Corpus_{m}_cot.csv"))
    for _, r in df.iterrows():
        t = str(r["generated_text"])
        mch = LEAK.findall(t)
        if mch:
            rows.append({"model": m, "region": r["region"],
                         "id": r["source_id"], "n_words": len(t.split()),
                         "markers": sorted(set(x.lower() for x in mch))[:4],
                         "head": t[:200]})

dfr = pd.DataFrame(rows)
print(f"TOTAL cot reasoning-leak candidates: {len(dfr)}")
if len(dfr):
    print("\nby model:")
    print(dfr.groupby("model").size())
    print("\nsamples:")
    for _, r in dfr.head(30).iterrows():
        print(f"[{r['model']}] {r['region']} {r['id']} {r['n_words']}w "
              f"<{','.join(r['markers'])}>: {r['head']!r}")
