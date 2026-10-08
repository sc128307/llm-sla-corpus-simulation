"""Phase 4 full feature extraction — metrics.py (21 spaCy features).

Design (per user contract):
  - 2 processes, each handling half the work (human baseline only in A).
  - INCREMENTAL SAVE: append to a shard parquet every CHECKPOINT rows.
  - RESUME: a progress log records completed keys; restart skips them.
  - Low memory: single process, one spaCy model + RoBERTa-CoLA per process.

Usage:
  python scripts/extract_features_v2.py A   # human + first half of files
  python scripts/extract_features_v2.py B   # second half of files
"""
import sys, os, time, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
from src.metrics import LinguisticAuditor
from src.data_loader import load_human_baseline
from src.paths import GENERATED_DIR, FEATURE_DIR, generated_path

OUT = str(GENERATED_DIR)
RES = str(FEATURE_DIR)
MODELS = ["Gemini-2.5-Flash", "Qwen-3.7-Plus", "DeepSeek-V4-Flash",
          "Llama-4-Maverick", "MiniMax-M3", "GPT-5.6-Luna"]
CONDS = ["zero_shot", "one_shot", "few_shot", "cot"]
CHECKPOINT = 200

FEAT_COLS = ["Word_Count", "Sentence_Length", "AVD_Combined",
             "CEFR_A1", "CEFR_A2", "CEFR_B1", "CEFR_B2", "CEFR_C1",
             "MTLD", "MDD", "Nominalization_Rate", "Passive_Ratio",
             "Discourse_Density", "Pronoun_Ratio", "FKGL",
             "Grammar_Acceptability", "Grammar_Error_Rate",
             "Top20_Word_Share", "Hapax_Ratio", "Lexical_Cohesion",
             "Referential_Density"]


def load_done(progress_path):
    if os.path.exists(progress_path):
        with open(progress_path, encoding="utf-8") as f:
            return set(json.load(f))
    return set()


def save_done(progress_path, done):
    with open(progress_path, "w", encoding="utf-8") as f:
        json.dump(sorted(done), f)


def append_rows(shard_path, rows):
    """Append rows to shard parquet, creating it if missing."""
    df = pd.DataFrame(rows)
    if os.path.exists(shard_path):
        old = pd.read_parquet(shard_path)
        df = pd.concat([old, df], ignore_index=True)
    df.to_parquet(shard_path, index=False)


def analyze_row(aud, text, key, region, topic, model, condition):
    f = aud.analyze_sample(str(text))
    if f is None:
        return None
    f["key"] = key
    f["region"] = region
    f["topic"] = topic
    f["model"] = model
    f["condition"] = condition
    return f


def done_key(key, model, condition):
    """Resume key must include model+condition: the same (id|topic) appears
    under HUMAN and every model×condition, so key alone is ambiguous."""
    return f"{key}|{model}|{condition}"


def process_corpus_file(aud, path, model, condition, done, shard_path,
                        progress_path, buf):
    df = pd.read_csv(path)
    n_done = n_skip = 0
    for _, r in df.iterrows():
        key = f"{r['source_id']}|{r['topic']}"
        dk = done_key(key, model, condition)
        if dk in done:
            n_skip += 1
            continue
        f = analyze_row(aud, r["generated_text"], key, r["region"],
                        r["topic"], model, condition)
        if f:
            buf.append(f)
            done.add(dk)
            n_done += 1
        if len(buf) >= CHECKPOINT:
            append_rows(shard_path, buf)
            save_done(progress_path, done)
            buf.clear()
    return n_done, n_skip


def main():
    which = sys.argv[1]  # "A" or "B"
    aud = LinguisticAuditor()
    print(f"engine: {aud.grammar_engine}", flush=True)

    shard_path = os.path.join(RES, f"features_{which}.parquet")
    progress_path = os.path.join(RES, f"features_{which}_done.json")
    done = load_done(progress_path)
    print(f"[{which}] resume: {len(done)} keys already done", flush=True)
    buf = []
    t0 = time.time()
    total_new = 0

    # human baseline -> A only
    if which == "A":
        human = load_human_baseline()
        n_done = n_skip = 0
        for _, r in human.iterrows():
            key = f"{r['id']}|{r['topic']}"
            dk = done_key(key, "HUMAN", "baseline")
            if dk in done:
                n_skip += 1
                continue
            f = analyze_row(aud, r["text"], key, r["region"], r["topic"],
                            "HUMAN", "baseline")
            if f:
                buf.append(f)
                done.add(dk)
                n_done += 1
            if len(buf) >= CHECKPOINT:
                append_rows(shard_path, buf)
                save_done(progress_path, done)
                buf.clear()
        total_new += n_done
        print(f"[A] human: +{n_done} new, {n_skip} skipped, "
              f"{time.time()-t0:.0f}s", flush=True)

    # corpus files: A takes even indices, B takes odd
    files = [(m, c) for m in MODELS for c in CONDS]
    my_files = files[0::2] if which == "A" else files[1::2]
    for m, c in my_files:
        path = str(generated_path(m, c))
        if not os.path.exists(path):
            print(f"[{which}] MISSING {path}", flush=True)
            continue
        n_done, n_skip = process_corpus_file(
            aud, path, m, c, done, shard_path, progress_path, buf)
        total_new += n_done
        print(f"[{which}] {m}/{c}: +{n_done} new, {n_skip} skipped, "
              f"{time.time()-t0:.0f}s", flush=True)

    # final flush
    if buf:
        append_rows(shard_path, buf)
        save_done(progress_path, done)
        buf.clear()
    print(f"[{which}] DONE: {total_new} new rows, total {len(done)} keys, "
          f"{(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
