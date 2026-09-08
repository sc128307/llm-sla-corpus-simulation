"""Audit explicit region/nationality markers in human and generated essays.

This is a leakage diagnostic, not a text-cleaning operation. It reports the
frequency of marker families by labeled region so a later anonymised sensitivity
analysis can use a fixed, documented vocabulary.
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.paths import CORPUS_DIR, GENERATED_DIR, RESULTS_DIR


MARKERS = {
    "CHN": ["china", "chinese", "beijing", "shanghai", "guangzhou"],
    "HKG": ["hong kong", "hongkong", "hong konger", "hong kongers", "cantonese"],
    "IDN": ["indonesia", "indonesian", "indonesians", "jakarta"],
    "JPN": ["japan", "japanese", "tokyo", "osaka"],
    "KOR": ["korea", "korean", "koreans", "south korea", "seoul", "busan"],
    "PAK": ["pakistan", "pakistani", "pakistanis", "islamabad", "karachi"],
    "PHL": ["philippines", "filipino", "filipinos", "manila"],
    "SIN": ["singapore", "singaporean", "singaporeans"],
    "THA": ["thailand", "thai", "thais", "bangkok"],
    "TWN": ["taiwan", "taiwanese", "taipei"],
    "ENS": [],
}


def compile_patterns():
    words = sorted({word for vals in MARKERS.values() for word in vals},
                   key=len, reverse=True)
    group_to_word = {f"m{i}": word for i, word in enumerate(words)}
    combined = re.compile(
        r"(?<!\w)(?:" + "|".join(
            f"(?P<m{i}>{re.escape(word)})" for i, word in enumerate(words)
        ) + r")(?!\w)",
        re.I,
    )
    return combined, group_to_word


def audit_frame(df, text_col, source, patterns):
    combined, group_to_word = patterns
    counts = {}
    totals = df.groupby("region").size().to_dict()
    for region, text in zip(df["region"].astype(str),
                            df[text_col].fillna("").astype(str)):
        key = str(region)
        row = counts.setdefault(key, {
            word: {"occurrences": 0, "texts": 0}
            for word in group_to_word.values()
        })
        seen = set()
        for match in combined.finditer(text):
            word = group_to_word[match.lastgroup]
            row[word]["occurrences"] += 1
            seen.add(word)
        for word in seen:
            row[word]["texts"] += 1
    rows = []
    for region, total in totals.items():
        for word in group_to_word.values():
            occurrences = counts[str(region)][word]["occurrences"]
            hits = counts[str(region)][word]["texts"]
            rows.append({
                "source": source,
                "region": str(region),
                "marker": word,
                "n_texts": int(total),
                "n_occurrences": int(occurrences),
                "n_texts_with_marker": int(hits),
                "pct_texts": float(hits / total * 100) if total else 0.0,
            })
    return rows


def main():
    patterns = compile_patterns()
    rows = []
    human_path = CORPUS_DIR / "human" / "human_data_full.csv"
    human = pd.read_csv(human_path)
    rows.extend(audit_frame(human, "text", "human", patterns))
    for path in sorted(GENERATED_DIR.glob("Corpus_*.csv")):
        df = pd.read_csv(path, usecols=["region", "generated_text"])
        rows.extend(audit_frame(df, "generated_text", path.stem, patterns))

    out = pd.DataFrame(rows)
    csv_path = RESULTS_DIR / "l1_region_marker_audit.csv"
    json_path = RESULTS_DIR / "l1_region_marker_audit.json"
    out.to_csv(csv_path, index=False)
    summary = {
        "marker_dictionary": MARKERS,
        "sources": int(out["source"].nunique()),
        "rows": int(len(out)),
        "max_pct_by_marker": (
            out.groupby("marker")["pct_texts"].max().sort_values(ascending=False).to_dict()
        ),
        "audit_csv": str(csv_path),
    }
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
