"""E4 — exemplar plagiarism check: n-gram overlap between generated texts
and their few-shot exemplars (the 'do not copy' compliance gate).

Also the canonical quality checker skeleton (E1–E5) for the full run.
Usage:
  python scripts/check_corpus.py --dir data/outputs/validation [--dir ...]
"""
import argparse
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

POOL = "data/interim/exemplar_pool.csv"
NGRAM_N = 8
OVERLAP_THRESHOLD = 0.30  # max share of generated n-grams also in exemplars


def ngrams(tokens, n=NGRAM_N):
    return set(zip(*[tokens[i:] for i in range(n)])) if len(tokens) >= n \
        else set()


def exemplars_for_cell(pool, region, topic, k):
    cell = pool[(pool["region"] == region) & (pool["topic"] == topic)]
    cell = cell.sort_values("file_name")
    return cell["text"].tolist()[:k]


def check_file(path, pool, k):
    df = pd.read_csv(path)
    df = df[df["generated_text"].notna()]
    flagged = []
    overlaps = []
    for _, r in df.iterrows():
        region, topic = r["region"], r["topic"]
        exs = exemplars_for_cell(pool, region, topic, k)
        if not exs:
            continue
        gen = set(r["generated_text"].lower().split())
        if len(gen) < NGRAM_N:
            continue
        gen_ng = ngrams(list(r["generated_text"].lower().split()))
        ex_ng = set()
        for ex in exs:
            ex_ng |= ngrams(list(ex.lower().split()))
        inter = len(gen_ng & ex_ng)
        ov = inter / len(gen_ng) if gen_ng else 0.0
        overlaps.append(ov)
        if ov > OVERLAP_THRESHOLD:
            flagged.append((r["source_id"], r["topic"], round(ov, 3)))
    if overlaps:
        import numpy as np
        print(f"{os.path.basename(path):46s} n={len(df):4d} "
              f"mean_overlap={np.mean(overlaps):.3f} "
              f"max={np.max(overlaps):.3f} flagged={len(flagged)}")
        for f in flagged[:5]:
            print(f"    FLAG {f}")
        return len(flagged)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    args = ap.parse_args()
    pool = pd.read_csv(POOL)
    total = 0
    for d in args.dirs:
        for cond, k in [("one_shot", 1), ("few_shot", 3)]:
            for f in sorted(glob.glob(os.path.join(d, f"Corpus_*_{cond}.csv"))):
                total += check_file(f, pool, k)
    print(f"\nE4 total flagged: {total}")


if __name__ == "__main__":
    main()
