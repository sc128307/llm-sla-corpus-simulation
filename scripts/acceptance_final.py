"""FINAL acceptance: E1-E5 + artifact + cot-leak on all 24 files.

E1 completeness: 5600 rows, unique (id,topic)
E2 no empty text
E3 length: 50-500 words
E4 exemplar plagiarism (one_shot/few_shot)
E5 no reasoning leakage in cot outputs (C3 hard requirement)
E6 no refusal/error-message artifacts (any length)
"""
import pandas as pd
import re
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

POOL = pd.read_csv("data/interim/exemplar_pool.csv")
OUT = "data/raw/LLM_Generations"
MODELS = ["Gemini-2.5-Flash", "Qwen-3.7-Plus", "DeepSeek-V4-Flash",
          "Llama-4-Maverick", "MiniMax-M3", "GPT-5.6-Luna"]
CONDS = ["zero_shot", "one_shot", "few_shot", "cot"]
NGRAM_N = 8
OVERLAP = 0.30
MIN_W, MAX_W = 50, 500

LEAK = re.compile(
    r"\b(constraint checklist|confidence score|strategiz|step \d+:|"
    r"i will now write|let'?s (?:think|consider|reason)|to do this well|"
    r"i need to consider|reasoning process|internal reasoning|"
    r"here'?s my (?:plan|reasoning)|my reasoning|checklist|"
    r"score: \d/\d|chain of thought)\b", re.I)

REFUSAL = re.compile(
    r"\b(as an ai|i am an ai|i'm an ai|as a language model|"
    r"i cannot (?:write|provide|assist|help|complete|fulfill|answer)|"
    r"i can'?t (?:write|provide|assist|help|complete|fulfill|answer)|"
    r"i apologize,? but|against (?:my|our) (?:policy|guidelines)|"
    r"not (?:possible|allowed) for me)\b", re.I)

def ngrams(tokens, n=NGRAM_N):
    return set(zip(*[tokens[i:] for i in range(n)])) if len(tokens) >= n else set()

def ov_vs_ex(text, region, topic, k):
    if k == 0:
        return 0.0
    cell = POOL[(POOL["region"] == region) & (POOL["topic"] == topic)]
    cell = cell.sort_values("file_name")
    exs = cell["text"].tolist()[:k]
    if not exs:
        return 0.0
    gn = ngrams(list(str(text).lower().split()))
    if not gn:
        return 1.0
    en = set()
    for ex in exs:
        en |= ngrams(list(str(ex).lower().split()))
    return len(gn & en) / len(gn)

issues = 0
for m in MODELS:
    for c in CONDS:
        path = os.path.join(OUT, f"Corpus_{m}_{c}.csv")
        if not os.path.exists(path):
            print(f"MISSING {m}/{c}")
            issues += 1
            continue
        df = pd.read_csv(path)
        k = {"zero_shot": 0, "one_shot": 1, "few_shot": 3, "cot": 0}[c]
        n = len(df)
        keys = df["source_id"].astype(str) + "|" + df["topic"].astype(str)
        dup = int(keys.duplicated().sum())
        empty = int(df["generated_text"].fillna("").str.strip().eq("").sum())
        nw = df["generated_text"].fillna("").str.split().str.len()
        short = int((nw < MIN_W).sum())
        long = int((nw > MAX_W).sum())
        e4 = 0
        if k > 0:
            for _, r in df.iterrows():
                if ov_vs_ex(r["generated_text"], r["region"], r["topic"], k) > OVERLAP:
                    e4 += 1
        leak = int(df["generated_text"].fillna("").str.contains(LEAK).sum()) if c == "cot" else 0
        refus = int(df["generated_text"].fillna("").str.contains(REFUSAL).sum())
        bad = dup + empty + short + long + e4 + leak + refus
        if bad or n != 5600:
            issues += 1
        tag = "OK " if bad == 0 and n == 5600 else "!! "
        print(f"{tag}{m}/{c:<10} n={n:5d} dup={dup} empty={empty} "
              f"short={short} long={long} e4={e4} leak={leak} refus={refus}")
print(f"\n{'ALL 24 PASS' if issues == 0 else f'{issues} ISSUE(S)'}")
