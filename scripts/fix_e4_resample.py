"""E4/validity fix: re-sample flagged rows under IDENTICAL generation
conditions (same prompt, same temperature=0.7, same exemplar).

Fixes two defect classes:
  (a) exemplar plagiarism (8-gram overlap > 0.30 vs same-cell exemplars)
  (b) truncated/garbled outputs (word count outside [MIN_WORDS, MAX_WORDS])

Methodological basis: LLM decoding at temperature>0 is stochastic; calling
the same configuration again is a legitimate re-sampling of the output
distribution and does not change the operational definition of the C1/C2
treatment. We loop until no flagged row remains or a round cap is hit;
residual copies are kept and marked (never deleted), to be reported as
limitation / excluded in analysis.

Gemini: realtime is geo-blocked (403) -> OpenRouter batch lane.
MiniMax / DeepSeek / Qwen: realtime lane.

Usage: python scripts/fix_e4_resample.py [--rounds 5] [--models DISP,...]
"""
import argparse
import os
import sys
import time
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
from dotenv import load_dotenv  # noqa: E402
load_dotenv(override=True)

from src.generate import (get_client, prepare_rows, build_request_body,  # noqa: E402
                          MAX_TOKENS, TEMPERATURE, _batch_model_slug)
from src.data_loader import load_human_baseline  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
POOL = pd.read_csv("data/interim/exemplar_pool.csv")
OUT = "data/raw/LLM_Generations"
NGRAM_N = 8
OVERLAP_THRESHOLD = 0.30
MIN_WORDS, MAX_WORDS = 50, 500
MARK_COL = "e4_resampled"  # marker column added when re-sampling touched a row

with open("configs/models.yaml", encoding="utf-8") as f:
    MODELS = {m["name"]: m for m in yaml.safe_load(f)["models"]}

# display -> (yaml name, condition, k)
TARGETS = [
    ("MiniMax-M3", "minimax/minimax-m3", "zero_shot", 0),
    ("MiniMax-M3", "minimax/minimax-m3", "one_shot", 1),
    ("MiniMax-M3", "minimax/minimax-m3", "few_shot", 3),
    ("MiniMax-M3", "minimax/minimax-m3", "cot", 0),
    ("Gemini-2.5-Flash", "google/gemini-2.5-flash:batch", "cot", 0),
    ("Gemini-2.5-Flash", "google/gemini-2.5-flash:batch", "one_shot", 1),
    ("DeepSeek-V4-Flash", "deepseek-v4-flash", "one_shot", 1),
    ("Qwen-3.7-Plus", "qwen3.7-plus", "one_shot", 1),
]


def ngrams(tokens, n=NGRAM_N):
    return set(zip(*[tokens[i:] for i in range(n)])) if len(tokens) >= n else set()


def overlap_vs_exemplars(text, region, topic, k):
    cell = POOL[(POOL["region"] == region) & (POOL["topic"] == topic)]
    cell = cell.sort_values("file_name")
    exs = cell["text"].tolist()[:k]
    if not exs:
        return 0.0
    gen_ng = ngrams(list(str(text).lower().split()))
    if not gen_ng:
        return 1.0
    ex_ng = set()
    for ex in exs:
        ex_ng |= ngrams(list(str(ex).lower().split()))
    return len(gen_ng & ex_ng) / len(gen_ng)


def flag_rows(df, k):
    """Flag rows failing plagiarism (if k>0) OR length sanity."""
    out = []
    for i, r in df.iterrows():
        text = str(r["generated_text"])
        n_words = len(text.split())
        bad_len = n_words < MIN_WORDS or n_words > MAX_WORDS
        bad_plag = False
        if k > 0 and not bad_len:
            bad_plag = overlap_vs_exemplars(text, r["region"], r["topic"],
                                            k) > OVERLAP_THRESHOLD
        if bad_len or bad_plag:
            out.append(i)
    return out


def call_identical(client, model_cfg, msgs):
    """One call under the SAME config as the original run.
    NOTE: api_extra must go through extra_body for realtime (SDK behaviour),
    exactly like src.generate.call_model does."""
    kwargs = dict(model=model_cfg["name"], messages=msgs,
                  temperature=TEMPERATURE)
    kwargs["max_completion_tokens" if model_cfg["provider"] == "openai"
           else "max_tokens"] = MAX_TOKENS
    if model_cfg.get("api_extra"):
        kwargs["extra_body"] = model_cfg["api_extra"]
    resp = client.chat.completions.create(**kwargs)
    return (resp.choices[0].message.content or "").strip()


def _passes(text, r, k):
    n_words = len(str(text).split())
    if n_words < MIN_WORDS or n_words > MAX_WORDS:
        return False
    if k > 0 and overlap_vs_exemplars(
            text, r["region"], r["topic"], k) > OVERLAP_THRESHOLD:
        return False
    return True


def fix_realtime(model_cfg, rows, k, rounds):
    client = get_client(model_cfg["provider"])
    done_ok = {}  # key -> text
    for rnd in range(1, rounds + 1):
        remaining = [(r, msgs) for r, msgs in rows
                     if f"{r['id']}|{r['topic']}" not in done_ok]
        if not remaining:
            break
        print(f"    round {rnd}: {len(remaining)} rows "
              f"(temp={TEMPERATURE}, prompt unchanged)", flush=True)
        for r, msgs in remaining:
            key = f"{r['id']}|{r['topic']}"
            try:
                text = call_identical(client, model_cfg, msgs)
                if text and _passes(text, r, k):
                    done_ok[key] = text
                else:
                    time.sleep(0.3)
            except Exception as e:
                print(f"      ✗ {key}: {str(e)[:90]}", flush=True)
                time.sleep(1.0)
    return done_ok


def fix_gemini_batch(model_cfg, rows, k):
    import requests
    api_key = os.getenv("OPENROUTER_API_KEY")
    slug = _batch_model_slug(model_cfg)
    lines = []
    for r, msgs in rows:
        key = f"{r['id']}|{r['topic']}"
        body = build_request_body(slug, msgs, model_cfg.get("api_extra"))
        body["temperature"] = TEMPERATURE
        lines.append({"custom_id": key, "body": body})
    resp = requests.post(
        "https://openrouter.ai/api/beta/batches",
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json"},
        json={"endpoint": "/v1/chat/completions",
              "model": slug, "requests": lines},
        timeout=120)
    if resp.status_code not in (200, 202):
        raise RuntimeError(f"OR batch submit failed ({resp.status_code}): "
                           f"{resp.text[:200]}")
    jid = resp.json().get("id") or resp.json().get("batch_id")
    print(f"    gemini batch {jid[:20]}… ({len(lines)} rows)", flush=True)
    t0 = time.time()
    while time.time() - t0 < 6 * 3600:
        r = requests.get(
            f"https://openrouter.ai/api/beta/batches/{jid}",
            headers={"Authorization": f"Bearer {api_key}"}, timeout=60)
        data = r.json()
        st = data.get("status")
        print(f"      batch status: {st}", flush=True)
        if st == "completed":
            break
        if st in ("failed", "cancelled", "expired"):
            raise RuntimeError(f"gemini batch {st}")
        time.sleep(120)
    done_ok = {}
    for it in data.get("results", []):
        if it.get("error"):
            continue
        rr = it.get("response", {})
        if rr.get("status_code") != 200:
            continue
        try:
            text = (rr["body"]["choices"][0]["message"]["content"]
                    or "").strip()
        except (KeyError, IndexError):
            continue
        cid = it.get("custom_id", "")
        sid, _, topic = cid.partition("|")
        row = human[(human["id"] == sid) & (human["topic"] == topic)]
        if len(row) == 1:
            r0 = row.iloc[0]
            if text and _passes(text, r0, k):
                done_ok[cid] = text
    return done_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--models", default=None,
                    help="comma-separated display names to process")
    args = ap.parse_args()
    global human
    human = load_human_baseline()

    for disp, yname, cond, k in TARGETS:
        if args.models and disp not in args.models.split(","):
            continue
        fname = f"Corpus_{disp}_{cond}.csv"
        path = os.path.join(OUT, fname)
        df = pd.read_csv(path)
        flagged = flag_rows(df, k)
        if not flagged:
            print(f"{fname}: clean ({len(df)} rows)")
            continue
        print(f"\n{fname}: {len(flagged)} flagged -> re-sampling "
              f"(identical config, max {args.rounds} rounds)", flush=True)

        keys = set(df.loc[flagged].apply(
            lambda r: f"{r['source_id']}|{r['topic']}", axis=1))
        sub = human[human.apply(
            lambda r: f"{r['id']}|{r['topic']}" in keys, axis=1)]
        rows = prepare_rows(sub, POOL, cond, k)
        model_cfg = MODELS[yname]

        if model_cfg["provider"] == "openrouter" and ":batch" in yname:
            done_ok = fix_gemini_batch(model_cfg, rows, k)
        else:
            done_ok = fix_realtime(model_cfg, rows, k, args.rounds)

        df = pd.read_csv(path)
        if MARK_COL not in df.columns:
            df[MARK_COL] = ""
        replaced = 0
        for i in df.index:
            key = f"{df.at[i, 'source_id']}|{df.at[i, 'topic']}"
            if key in done_ok:
                df.at[i, "generated_text"] = done_ok[key]
                df.at[i, MARK_COL] = "resampled"
                replaced += 1
        df.to_csv(path, index=False, encoding="utf-8")
        still = len(flag_rows(df, k))
        print(f"  -> {fname}: replaced {replaced}, still flagged {still}",
              flush=True)

    print("\nDONE")


if __name__ == "__main__":
    main()
