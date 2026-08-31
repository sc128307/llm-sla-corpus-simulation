"""Data-cleaning note (reproducible): remove reasoning preamble from
Gemini-2.5-Flash cot PAK_175.

Rationale: C3 (cot) prompt requires reasoning to stay internal and the
output to contain ONLY the essay. Gemini emitted a visible reasoning
preamble before the essay body. Per the C3 contract, the preamble is
removed; the essay body is kept verbatim (270 words, complete, valid).
This is compliant post-processing of the model output, not regeneration;
documented in the paper's corpus-cleaning section.

Detection rule: text before the first line starting with the essay's
first sentence ("Part-time job is a common thing") is the preamble.
"""
import pandas as pd
import re
import sys, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8")

PATH = "data/raw/LLM_Generations/Corpus_Gemini-2.5-Flash_cot.csv"
ESSAY_START = "Part-time job is a common thing"

df = pd.read_csv(PATH)
mask = (df["source_id"] == "PAK_175") & (df["topic"] == "Part-time Job")
idx = df.index[mask]
assert len(idx) == 1, f"expected 1 row, got {len(idx)}"
i = idx[0]
t = str(df.at[i, "generated_text"])

pos = t.find(ESSAY_START)
assert pos > 0, "essay start not found"

preamble = t[:pos]
body = t[pos:]

print(f"row {i}: total {len(t.split())} words -> preamble "
      f"{len(preamble.split())} words removed, body {len(body.split())} words")

# keep a trace of the cleaning
log_path = "data/outputs/corpus_cleaning_log.csv"
log = []
if os.path.exists(log_path):
    log = pd.read_csv(log_path).to_dict("records")
log.append({"file": "Corpus_Gemini-2.5-Flash_cot.csv",
            "source_id": "PAK_175", "topic": "Part-time Job",
            "action": "remove_reasoning_preamble",
            "words_before": len(t.split()),
            "words_after": len(body.split()),
            "note": "C3 reasoning preamble removed; essay body kept verbatim"})
pd.DataFrame(log).to_csv(log_path, index=False, encoding="utf-8")

df.at[i, "generated_text"] = body
if "e4_resampled" not in df.columns:
    df["e4_resampled"] = ""
df.at[i, "e4_resampled"] = "preamble_removed"
df.to_csv(PATH, index=False, encoding="utf-8")
print("saved; cleaning log ->", log_path)
