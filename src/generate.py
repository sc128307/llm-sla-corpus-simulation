"""Phase 3 — condition-parameterised corpus generation (6 models x 4 conditions).

Roster and routing are defined in ``configs/models.yaml``. The frozen run uses
the six configured model IDs
(GPT-5.6-Luna, Gemini-2.5-Flash, Qwen-3.7-Plus, DeepSeek-V4-Flash,
Llama-4-Maverick, and MiniMax-M3).

Batch lane (OpenAI-compatible /v1/batches, used by OpenAI official and
DashScope compatible-mode):
  python -m src.generate --batch-action submit    # build JSONL + submit, save job ids
  python -m src.generate --batch-action retrieve  # poll jobs, download, convert to CSV
The command-line lane is selected from the model configuration; provider and
model identity are never inferred from code defaults.

Design invariants (also summarised in the release README):
  - Canonical baseline via src.data_loader.load_human_baseline()
    (5,600 rows; pairing key = (id, topic)).
  - Exemplars (one/few-shot) fixed per (region, topic) cell, drawn ONLY from
    corpus/manifest/exemplar_pool.csv (never the L1-probe test split).
  - temperature 0.7, max_tokens 2000, provider-specific reasoning/thinking
    channels disabled (see configs/models.yaml).  This disables hidden
    reasoning telemetry; it is not a request to expose chain-of-thought.
  - Output: corpus/generated/Corpus_<Model>_<Condition>.csv with
    columns source_id, region, topic, proficiency, generated_text, model_id,
    condition, latency_s, prompt_tokens, completion_tokens.
  - Per-row usage is recorded in each generated CSV. A final run manifest must
    be created separately before reporting costs or provenance.
"""
import argparse
import os
import sys
import time
import json

# Windows GBK consoles choke on the emoji in progress prints; force UTF-8
# output so the script never depends on the caller's console encoding.
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if sys.stderr:
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_human_baseline  # noqa: E402
from src.paths import (EXEMPLAR_POOL_PATH, GENERATED_DIR, RESULTS_DIR,
                       generated_path)  # noqa: E402

load_dotenv(override=True)

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_PATH = BASE_DIR / "configs" / "models.yaml"
PROMPTS_PATH = BASE_DIR / "configs" / "prompts.yaml"
POOL_PATH = EXEMPLAR_POOL_PATH
OUT_DIR = GENERATED_DIR
BATCH_DIR = RESULTS_DIR / "batch_jobs"
MANIFEST_PATH = RESULTS_DIR / "generation_manifest.csv"

# OpenAI-compatible endpoints per provider
PROVIDERS = {
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "deepseek": ("https://api.deepseek.com", "DEEPSEEK_API_KEY"),
    "aliyun": ("https://dashscope.aliyuncs.com/compatible-mode/v1",
               "DASHSCOPE_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "together": ("https://api.together.xyz/v1", "TOGETHER_API_KEY"),
}

# Approximate USD per 1M tokens (input, output) — batch prices where the
# lane is batch, list prices otherwise. Estimates only; exact cost comes
# from provider invoices. Prices quoted by user 2026-08-22 where noted.
PRICING = {
    "gpt-5.6-luna": (0.10, 0.60),
    "google/gemini-2.5-flash:batch": (0.15, 1.25),
    "deepseek-v4-flash": (0.06, 0.12),
    "qwen3.7-plus": (0.16, 0.64),
    "meta-llama/llama-4-maverick": (0.20, 0.80),
    "minimax/minimax-m3": (0.30, 1.20),
}

TEMPERATURE = 0.7
# A uniform 2,000-token completion cap is used for all six configured models.
# Provider-side reasoning/thinking is disabled in configs/models.yaml; retries
# therefore handle empty or malformed responses as ordinary API failures rather
# than assuming that tokens were consumed by an undisclosed reasoning trace.
MAX_TOKENS = 2000
MAX_RETRIES = 4
BATCH_POLL_SEC = 60


# ----------------------------------------------------------------------
# Prompt construction
# ----------------------------------------------------------------------
def load_prompts():
    with open(PROMPTS_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_messages(prompts_cfg, condition: str, row, exemplars):
    """[system, user] for one (condition, essay) pair. Only the intervention
    block varies across conditions."""
    region = row["region"]
    cond = prompts_cfg["conditions"][condition]
    if region == "ENS":
        identity_block = prompts_cfg["identity_definitions"]["Native"]
    else:
        country = prompts_cfg["persona_mapping"].get(region, region)
        identity_block = prompts_cfg["identity_definitions"]["Learner"].format(
            country=country)
    fmt = {"identity_block": identity_block, "topic": row["topic"],
           "region": region,
           "region_name": prompts_cfg["persona_mapping"].get(region, region),
           "exemplar_source_desc": (
               "who is a native English speaker" if region == "ENS"
               else "whose English is their second language")}
    if condition == "one_shot":
        fmt["exemplar"] = exemplars[0]
    elif condition == "few_shot":
        fmt["exemplars"] = "\n\n".join(exemplars)
    system = cond["system"].format(**fmt)
    user = cond["user"].format(**fmt)
    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]


def load_exemplars_by_cell(pool: pd.DataFrame, region: str, topic: str,
                           k: int) -> list:
    """Fixed, deterministic exemplars per (region, topic) cell from the pool
    split only (never the classifier test split)."""
    cell = pool[(pool["region"] == region) & (pool["topic"] == topic)]
    cell = cell.sort_values("file_name")
    return cell["text"].tolist()[:k]


# ----------------------------------------------------------------------
# Clients
# ----------------------------------------------------------------------
def get_client(provider: str):
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider: {provider}")
    base_url, env_key = PROVIDERS[provider]
    api_key = os.getenv(env_key)
    if not api_key:
        raise ValueError(f"Missing {env_key} in .env")
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError("openai package required: pip install openai") from e
    return OpenAI(api_key=api_key, base_url=base_url, timeout=120.0)


def call_model(client, model_id: str, messages, temperature, max_tokens,
               api_extra=None, use_max_completion_tokens=False):
    """One chat completion with retries. Returns (text, latency, pt, ct).

    Empty/whitespace-only content counts as a failed attempt and is retried:
    reasoning models (e.g. DeepSeek-V4-Flash) can spend the whole token
    budget on reasoning and return an empty message; re-sampling (temperature
    0.7) converges because the reasoning length is stochastic.
    """
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            t0 = time.time()
            kwargs = dict(model=model_id, messages=messages,
                          temperature=temperature)
            # GPT-5.6-family (OpenAI official) requires max_completion_tokens
            kwargs["max_completion_tokens" if use_max_completion_tokens
                   else "max_tokens"] = max_tokens
            if api_extra:
                kwargs["extra_body"] = api_extra
            resp = client.chat.completions.create(**kwargs)
            latency = time.time() - t0
            text = (resp.choices[0].message.content or "").strip()
            if not text:
                last_err = RuntimeError(
                    "empty content (reasoning likely consumed the budget)")
                time.sleep(2 ** attempt)
                continue
            usage = resp.usage
            return (text, latency,
                    getattr(usage, "prompt_tokens", 0) or 0,
                    getattr(usage, "completion_tokens", 0) or 0)
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"call failed after {MAX_RETRIES} tries: {last_err}")


# ----------------------------------------------------------------------
# Shared: per-cell exemplars + row preparation
# ----------------------------------------------------------------------
def prepare_rows(human: pd.DataFrame, pool: pd.DataFrame, condition: str,
                 k: int):
    prompts_cfg = load_prompts()
    exemplar_map = {}
    for (region, topic), _ in human.groupby(["region", "topic"]):
        exemplar_map[(region, topic)] = load_exemplars_by_cell(
            pool, region, topic, k)
    rows = []
    for _, r in human.iterrows():
        ex = exemplar_map[(r["region"], r["topic"])]
        msgs = build_messages(prompts_cfg, condition, r, ex)
        rows.append((r, msgs))
    return rows


def build_request_body(model_id: str, messages, api_extra=None,
                       use_max_completion_tokens=False):
    body = {"model": model_id, "messages": messages,
            "temperature": TEMPERATURE}
    # GPT-5.6-family (OpenAI official) requires max_completion_tokens
    body["max_completion_tokens" if use_max_completion_tokens
         else "max_tokens"] = MAX_TOKENS
    if api_extra:
        body.update(api_extra)
    return body


# ----------------------------------------------------------------------
# Realtime lane
# ----------------------------------------------------------------------
def generate_realtime(model_cfg, condition, human, pool, out_dir, limit,
                      workers, k):
    model_id = model_cfg["name"]
    display = model_cfg["display_name"]
    api_extra = model_cfg.get("api_extra")
    client = get_client(model_cfg["provider"])
    out_file = out_dir / f"Corpus_{display.replace(' ', '_')}_{condition}.csv"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if out_file.exists():
        try:
            existing = pd.read_csv(out_file)
            done = set(existing["source_id"].astype(str) + "|"
                       + existing["topic"].astype(str))
        except Exception:
            done = set()
    if done:
        print(f"  [resume] {display}/{condition}: {len(done)} (id,topic) done")

    rows = human.head(limit) if limit is not None else human
    prepared = prepare_rows(rows, pool, condition, k)

    results = []
    errors = 0
    total_pt = total_ct = 0
    t_start = time.time()

    def _flush(results, out_file, force=False):
        """Incremental checkpoint: append collected results to the CSV so a
        killed process loses at most the last un-flushed batch, and resume
        re-runs never re-pay for already-persisted essays."""
        if not results:
            return
        new_df = pd.DataFrame(results)
        if out_file.exists():
            try:
                old = pd.read_csv(out_file)
                if len(old):
                    new_df = pd.concat([old, new_df], ignore_index=True)
            except Exception:
                pass  # stale/empty file — start fresh
        new_df.to_csv(out_file, index=False, encoding="utf-8")
        results.clear()
        if force:
            print(f"    [checkpoint] {display}/{condition}: saved "
                  f"{len(new_df)} rows to {out_file.name}", flush=True)

    def work(item):
        row, msgs = item
        key = f"{row['id']}|{row['topic']}"
        if key in done:
            return None
        text, latency, pt, ct = call_model(
            client, model_id, msgs, TEMPERATURE, MAX_TOKENS, api_extra,
            use_max_completion_tokens=(
                model_cfg["provider"] == "openai"))
        return {"source_id": row["id"], "region": row["region"],
                "topic": row["topic"], "proficiency": row["proficiency"],
                "generated_text": text, "model_id": model_id,
                "condition": condition, "latency_s": round(latency, 2),
                "prompt_tokens": pt, "completion_tokens": ct}

    print(f"  {display}/{condition}: {len(prepared)} essays, "
          f"{workers} workers, temp={TEMPERATURE}")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(work, it): it for it in prepared}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                res = fut.result()
                if res is not None:
                    results.append(res)
                    total_pt += res["prompt_tokens"]
                    total_ct += res["completion_tokens"]
            except Exception as e:
                errors += 1
                print(f"  ✗ error: {e}")
            if i % 200 == 0:
                print(f"    ... {i}/{len(prepared)} (ok={len(results)})",
                      flush=True)
                _flush(results, out_file)  # incremental checkpoint

    # Final flush: persist anything not yet checkpointed (or create an empty
    # header file so a completed-but-empty run leaves a valid CSV).
    _flush(results, out_file, force=True)
    if not out_file.exists():
        new_df = pd.DataFrame(
            columns=["source_id", "region", "topic", "proficiency",
                     "generated_text", "model_id", "condition", "latency_s",
                     "prompt_tokens", "completion_tokens"])
        new_df.to_csv(out_file, index=False, encoding="utf-8")

    try:
        existing = pd.read_csv(out_file)
        n_rows = len(existing)
    except Exception:
        n_rows = len(results)

    summary = {"model": display, "model_id": model_id, "condition": condition,
               "rows_requested": len(prepared), "rows_written": n_rows,
               "errors": errors,
               "elapsed_min": round((time.time() - t_start) / 60, 1),
               "prompt_tokens": total_pt, "completion_tokens": total_ct}
    _add_cost(summary, model_id)
    print(f"  ✅ {display}/{condition}: {n_rows} rows, "
          f"${summary.get('est_cost_usd', 0):.2f} (est), "
          f"{summary['elapsed_min']} min")
    return summary


# ----------------------------------------------------------------------
# Batch lane
#   openrouter -> native Batch API: POST /api/beta/batches (JSON, 24h window)
#   openai/aliyun -> OpenAI-style files + /v1/batches (best effort;
#                    DashScope compatible-mode support TBD at submit time)
# ----------------------------------------------------------------------
def batch_jsonl_path(model_cfg, condition):
    return BATCH_DIR / (f"{model_cfg['display_name'].replace(' ', '_')}_"
                        f"{condition}.jsonl")


def batch_jobid_path(model_cfg, condition):
    return BATCH_DIR / (f"{model_cfg['display_name'].replace(' ', '_')}_"
                        f"{condition}.jobid")


def _batch_model_slug(model_cfg):
    # OpenRouter beta/batches takes the plain slug; the ':batch' suffix is
    # a catalog/pricing marker, not part of the request model id.
    return model_cfg["name"].replace(":batch", "")


def batch_jobids(model_cfg, condition):
    """All submitted job ids for (model, condition) — one per line."""
    p = batch_jobid_path(model_cfg, condition)
    if not p.exists():
        return []
    return [l.strip() for l in p.read_text(encoding="utf-8").splitlines()
            if l.strip()]


def batch_submit(model_cfg, condition, human, pool, k, limit=None,
                 chunk_size=None, sequential=False):
    """Submit the (model x condition) batch, optionally split into chunks.

    OpenAI's Batch API limits queued prompt tokens per model (see Platform
    Settings); chunk_size splits the requests so we can stay under the
    limit. With sequential=True each chunk is submitted only after the
    previous one completes. All job ids are stored one per line in the
    .jobid file; batch_retrieve merges them. Existing .jobid entries are
    kept (a chunk that already completed keeps its id so its results are
    not lost), and keys already present in the output CSV are skipped
    (resume, so re-runs never double-pay for completed essays).
    """
    rows = human.head(limit) if limit is not None else human
    prepared = prepare_rows(rows, pool, condition, k)
    slug = _batch_model_slug(model_cfg)
    display = model_cfg["display_name"]

    # Resume: skip (id, topic) already written to the output CSV.
    done = set()
    out_file = OUT_DIR / f"Corpus_{display.replace(' ', '_')}_{condition}.csv"
    if out_file.exists():
        try:
            existing = pd.read_csv(out_file)
            done = set(existing["source_id"].astype(str) + "|"
                       + existing["topic"].astype(str))
        except Exception:
            done = set()
    if done:
        print(f"  [resume] {display}/{condition}: {len(done)} (id,topic) "
              f"already done, skipping")

    lines = []
    for row, msgs in prepared:
        body = build_request_body(
            slug, msgs, model_cfg.get("api_extra"),
            use_max_completion_tokens=(model_cfg["provider"] == "openai"))
        key = f"{row['id']}|{row['topic']}"
        if key in done:
            continue
        lines.append({"custom_id": key, "body": body})

    if not lines:
        print(f"  [batch] {display}/{condition}: nothing to submit "
              f"(all {len(done)} keys already done)")
        return []

    chunks = ([lines[i:i + chunk_size] for i in range(0, len(lines),
               chunk_size)] if chunk_size else [lines])
    print(f"  [batch] {display}/{condition}: {len(lines)} requests -> "
          f"{len(chunks)} chunk(s)")
    jobids = []
    for idx, chunk in enumerate(chunks):
        if model_cfg["provider"] == "openrouter":
            jid = _submit_chunk_openrouter(model_cfg, condition, chunk, idx)
        else:
            jid = _submit_chunk_openai_style(model_cfg, condition, chunk, idx)
        jobids.append(jid)
        print(f"    chunk {idx + 1}/{len(chunks)} -> job {jid[:24]}…")
        if sequential and idx < len(chunks) - 1:
            print(f"    waiting for chunk {idx + 1} to complete...",
                  flush=True)
            st = _wait_job_complete(model_cfg, jid)
            if st in ("failed", "cancelled", "expired", "timeout"):
                raise RuntimeError(
                    f"chunk {idx + 1} {st}; aborting sequential submit")
    # Merge with previously recorded job ids (never drop a completed chunk).
    p = batch_jobid_path(model_cfg, condition)
    p.parent.mkdir(parents=True, exist_ok=True)
    existing_ids = []
    if p.exists():
        existing_ids = [l.strip() for l in
                        p.read_text(encoding="utf-8").splitlines()
                        if l.strip()]
    merged = list(dict.fromkeys(existing_ids + jobids))
    p.write_text("\n".join(merged) + "\n", encoding="utf-8")
    print(f"  [batch] {display}/{condition}: {len(jobids)} new job id(s) "
          f"saved -> {p.name} (total {len(merged)})")
    return jobids


def _submit_chunk_openrouter(model_cfg, condition, chunk, idx):
    import requests
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Missing OPENROUTER_API_KEY in .env")
    payload = {"endpoint": "/v1/chat/completions",
               "model": _batch_model_slug(model_cfg), "requests": chunk}
    resp = requests.post(
        "https://openrouter.ai/api/beta/batches",
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json"},
        json=payload, timeout=120)
    if resp.status_code not in (200, 202):
        raise RuntimeError(f"OpenRouter batch submit failed "
                           f"({resp.status_code}): {resp.text[:300]}")
    return resp.json().get("id") or resp.json().get("batch_id")


def _submit_chunk_openai_style(model_cfg, condition, chunk, idx):
    """OpenAI official / DashScope compatible-mode: files + /v1/batches.
    NOTE: OpenAI has a file-visibility propagation delay — a batch created
    immediately after files.create can fail with "Cannot find file / org
    does not have access" (observed 2026-08-28, wave A). We wait + retry."""
    client = get_client(model_cfg["provider"])
    base = batch_jsonl_path(model_cfg, condition)
    path = base.with_name(f"{base.stem}_c{idx}{base.suffix}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for it in chunk:
            f.write(json.dumps(
                {"custom_id": it["custom_id"], "method": "POST",
                 "url": "/v1/chat/completions", "body": it["body"]},
                ensure_ascii=False) + "\n")
    last_err = None
    for attempt in range(3):
        with open(path, "rb") as f:
            file_resp = client.files.create(file=f, purpose="batch")
        time.sleep(10)  # file-visibility propagation delay
        batch = client.batches.create(
            input_file_id=file_resp.id, endpoint="/v1/chat/completions",
            completion_window="24h")
        time.sleep(15)  # early validation check
        try:
            check = client.batches.retrieve(batch.id)
            if check.status != "failed":
                return batch.id
            last_err = getattr(check, "errors", None)
            print(f"    [batch] attempt {attempt} failed: {last_err}; "
                  f"retrying...", flush=True)
        except Exception as e:
            last_err = e
            time.sleep(5)
    raise RuntimeError(f"batch submit failed after 3 tries: {last_err}")


def _poll_job_status(model_cfg, jid):
    if model_cfg["provider"] == "openrouter":
        import requests
        api_key = os.getenv("OPENROUTER_API_KEY")
        resp = requests.get(
            f"https://openrouter.ai/api/beta/batches/{jid}",
            headers={"Authorization": f"Bearer {api_key}"}, timeout=60)
        if resp.status_code != 200:
            return "unknown"
        return resp.json().get("status")
    client = get_client(model_cfg["provider"])
    return client.batches.retrieve(jid).status


def _wait_job_complete(model_cfg, jid, poll_sec=300, deadline_h=24):
    import time
    t0 = time.time()
    while time.time() - t0 < deadline_h * 3600:
        st = _poll_job_status(model_cfg, jid)
        print(f"      chunk status: {st}", flush=True)
        if st in ("completed", "succeeded", "failed", "cancelled",
                  "expired"):
            return st
        time.sleep(poll_sec)
    return "timeout"


def batch_retrieve(model_cfg, condition, out_dir):
    """Fetch and merge all chunk jobs for (model, condition)."""
    display = model_cfg["display_name"]
    jobids = batch_jobids(model_cfg, condition)
    if not jobids:
        print(f"  [batch] no job ids for {display}/{condition} — submit "
              f"first")
        return None
    all_items = []
    total_pt = total_ct = 0
    pending = []
    actual_cost = 0.0
    for jid in jobids:
        if model_cfg["provider"] == "openrouter":
            items, pt, ct, st, cost = _fetch_openrouter_job(
                model_cfg, condition, jid)
        else:
            items, pt, ct, st = _fetch_openai_job(model_cfg, condition, jid)
            cost = 0.0
        print(f"  [batch] {display}/{condition}: job {jid[:16]}… -> {st}")
        if st in ("completed", "succeeded"):
            all_items.extend(items)
            total_pt += pt
            total_ct += ct
            actual_cost += cost
        else:
            pending.append((jid, st))
    if pending:
        print(f"  [batch] {display}/{condition}: {len(pending)} job(s) not "
              f"done: {pending}")
    if not all_items:
        return None
    out_file = out_dir / f"Corpus_{display.replace(' ', '_')}_{condition}.csv"
    summary = _write_batch_csv(model_cfg, condition, out_file, all_items,
                               total_pt, total_ct)
    if summary is not None and actual_cost:
        summary["actual_cost_usd"] = round(actual_cost, 4)
        print(f"  💰 [batch] {display}/{condition}: OpenRouter actual cost "
              f"${summary['actual_cost_usd']}")
    return summary


def _fetch_openrouter_job(model_cfg, condition, jid):
    import requests
    api_key = os.getenv("OPENROUTER_API_KEY")
    resp = requests.get(
        f"https://openrouter.ai/api/beta/batches/{jid}",
        headers={"Authorization": f"Bearer {api_key}"}, timeout=60)
    if resp.status_code != 200:
        return [], 0, 0, f"poll-fail-{resp.status_code}", 0.0
    data = resp.json()
    status = data.get("status")
    if status != "completed":
        return [], 0, 0, status, 0.0
    items = []
    skipped_empty = 0
    for it in data.get("results", []):
        err = it.get("error")
        if err:
            print(f"  ⚠️ {it.get('custom_id')}: {str(err)[:110]}")
            continue
        r = it.get("response", {})
        if r.get("status_code") != 200:
            continue
        body = r.get("body", {})
        try:
            text = (body["choices"][0]["message"]["content"] or "").strip()
        except (KeyError, IndexError):
            continue
        if not text:
            skipped_empty += 1
            continue
        sid, _, topic = it["custom_id"].partition("|")
        items.append({"source_id": sid, "topic": topic,
                      "generated_text": text,
                      "model_id": model_cfg["name"], "condition": condition,
                      "latency_s": None, "prompt_tokens": 0,
                      "completion_tokens": 0})
    usage = data.get("usage") or {}
    pt = usage.get("prompt_tokens", 0) or 0
    ct = usage.get("completion_tokens", 0) or 0
    cost = usage.get("cost", 0) or 0
    if skipped_empty:
        print(f"  ⚠️ [batch] {condition}: {skipped_empty} empty-content "
              f"items skipped")
    return items, pt, ct, status, cost


def _fetch_openai_job(model_cfg, condition, jid):
    client = get_client(model_cfg["provider"])
    try:
        batch = client.batches.retrieve(jid)
    except Exception as e:
        # e.g. batch deleted server-side or id truncated — report, skip
        print(f"  [batch] {jid[:20]}… retrieve error: {str(e)[:120]}")
        return [], 0, 0, "unretrievable"
    status = batch.status
    if status == "failed":
        print(f"  [batch] JOB FAILED: {getattr(batch, 'errors', None)}")
        return [], 0, 0, status
    if status not in ("completed", "succeeded"):
        return [], 0, 0, status
    try:
        content = client.files.content(batch.output_file_id)
    except Exception as e:
        # output file already deleted (we prune retrieved outputs) — the
        # rows, if any, are already merged into the CSV by an earlier run
        print(f"  [batch] {jid[:20]}… output file gone: {str(e)[:120]}")
        return [], 0, 0, "output_deleted"
    items = []
    total_pt = total_ct = 0
    for line in content.text.splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        resp = obj.get("response", {})
        if resp.get("status_code") != 200:
            continue
        body = resp.get("body", {})
        text = (body["choices"][0]["message"]["content"] or "").strip()
        if not text:
            continue
        usage = body.get("usage", {})
        pt = usage.get("prompt_tokens", 0) or 0
        ct = usage.get("completion_tokens", 0) or 0
        total_pt += pt
        total_ct += ct
        # OpenAI batch output: 'id' is a system id; the request key is in
        # 'custom_id' (observed 2026-08-28: id = 'batch_req_...').
        cid = obj.get("custom_id") or obj.get("id") or ""
        sid, _, topic = cid.partition("|")
        items.append({"source_id": sid, "topic": topic,
                      "generated_text": text,
                      "model_id": model_cfg["name"], "condition": condition,
                      "latency_s": None, "prompt_tokens": pt,
                      "completion_tokens": ct})
    return items, total_pt, total_ct, status


def _write_batch_csv(model_cfg, condition, out_file, results, total_pt,
                     total_ct):
    """Attach region/proficiency from the baseline and write the CSV."""
    out_file.parent.mkdir(parents=True, exist_ok=True)
    human = load_human_baseline()
    meta = human.set_index(["id", "topic"])[["region", "proficiency"]]
    df = pd.DataFrame(results)
    df = df.merge(meta, left_on=["source_id", "topic"],
                  right_index=True, how="left")
    df = df[["source_id", "region", "topic", "proficiency", "generated_text",
             "model_id", "condition", "latency_s", "prompt_tokens",
             "completion_tokens"]]
    # Merge with any rows already on disk (resume across chunk retrievals /
    # already-retrieved chunks whose output files were pruned) — never
    # overwrite previously fetched essays with a partial set.
    if out_file.exists():
        try:
            old = pd.read_csv(out_file)
            if len(old):
                df = pd.concat([old, df], ignore_index=True)
                df = df.drop_duplicates(subset=["source_id", "topic"],
                                        keep="last")
        except Exception:
            pass  # stale/empty file — start fresh
    df.to_csv(out_file, index=False, encoding="utf-8")
    summary = {"model": model_cfg["display_name"],
               "model_id": model_cfg["name"], "condition": condition,
               "rows_requested": None, "rows_written": len(df), "errors": 0,
               "elapsed_min": None, "prompt_tokens": total_pt,
               "completion_tokens": total_ct}
    _add_cost(summary, model_cfg["name"])
    print(f"  ✅ [batch] {model_cfg['display_name']}/{condition}: {len(df)} "
          f"rows -> {out_file.name}, "
          f"${summary.get('est_cost_usd', 0):.2f} (est)")
    return summary


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _add_cost(summary, model_id):
    price = PRICING.get(model_id)
    if price and (summary["prompt_tokens"] or summary["completion_tokens"]):
        p_in, p_out = price
        summary["est_cost_usd"] = round(
            (summary["prompt_tokens"] * p_in
             + summary["completion_tokens"] * p_out) / 1e6, 2)


def main():
    ap = argparse.ArgumentParser(description="Phase 3 generation (6x4 matrix)")
    ap.add_argument("--models", default=None,
                    help="comma-separated model names (models.yaml `name`); "
                         "default: all")
    ap.add_argument("--conditions", default=None,
                    help="comma-separated: zero_shot,one_shot,few_shot,cot; "
                         "default: all")
    ap.add_argument("--limit", type=int, default=None,
                    help="generate only the first N essays (smoke test)")
    ap.add_argument("--per-cell", type=int, default=None,
                    help="stratified validation: sample N essays from EACH "
                         "(region, topic) cell (seed-fixed), so all configured "
                         "regions x 2 topics are represented — needed to "
                         "test the L1-signal claim (--limit would only give "
                         "the first region).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--batch-action", choices=["submit", "retrieve", "all"],
                    default=None,
                    help="batch lane action (openai/aliyun models); "
                         "'all' = submit then poll+retrieve in one go")
    ap.add_argument("--lane", choices=["auto", "batch", "realtime"],
                    default="auto",
                    help="override the lane from models.yaml. 'realtime' "
                         "lets free-quota models (deepseek-v4-flash, "
                         "qwen3.8-max) run on the free tier (pilot); "
                         "'batch' forces the batch lane (full run).")
    ap.add_argument("--batch-chunk", type=int, default=None,
                    help="split each (model x condition) batch into chunks "
                         "of N requests (OpenAI Batch API queued-prompt-token "
                         "limits); job ids stored one per line, retrieve "
                         "merges chunks")
    ap.add_argument("--batch-sequential", action="store_true",
                    help="with --batch-chunk: wait for each chunk to "
                         "complete before submitting the next (respects "
                         "per-model queued-token limits)")
    args = ap.parse_args()

    with open(MODELS_PATH, encoding="utf-8") as f:
        models_cfg = yaml.safe_load(f)["models"]
    if args.models:
        wanted = set(args.models.split(","))
        models_cfg = [m for m in models_cfg if m["name"] in wanted]
    conditions = (args.conditions.split(",") if args.conditions
                  else ["zero_shot", "one_shot", "few_shot", "cot"])

    batch_models = [m for m in models_cfg if m.get("mode") == "batch"]
    rt_models = [m for m in models_cfg if m.get("mode") != "batch"]
    if args.lane == "batch":
        batch_models, rt_models = models_cfg, []
    elif args.lane == "realtime":
        batch_models, rt_models = [], models_cfg

    human = load_human_baseline()
    pool = pd.read_csv(POOL_PATH)
    if args.per_cell:
        # Stratified per (region, topic) cell — every region/topic is
        # represented so the L1-signal claim is actually testable.
        human = human.groupby(["region", "topic"], group_keys=False).apply(
            lambda g: g.sample(n=args.per_cell, random_state=args.seed))
        args.limit = None  # use the full stratified sample
        print(f"[validation] stratified sample: {len(human)} essays "
              f"({args.per_cell} per region x topic cell, seed {args.seed})")
    print(f"Human baseline: {len(human)} | pool: {len(pool)} | "
          f"matrix: {len(models_cfg)} models x {len(conditions)} conditions "
          f"(batch={len(batch_models)}, realtime={len(rt_models)})")

    summaries = []
    fatal = []
    for cond in conditions:
        for model_cfg in rt_models:
            try:
                summaries.append(generate_realtime(
                    model_cfg, cond, human, pool, args.out,
                    limit=args.limit, workers=args.workers, k=args.k))
            except Exception as e:
                print(f"✗ FAILED {model_cfg['display_name']}/{cond}: {e}")

    if args.batch_action:
        for cond in conditions:
            for model_cfg in batch_models:
                try:
                    if args.batch_action in ("submit", "all"):
                        batch_submit(model_cfg, cond, human, pool, args.k,
                                     limit=args.limit,
                                     chunk_size=args.batch_chunk,
                                     sequential=args.batch_sequential)
                    if args.batch_action in ("retrieve", "all"):
                        s = batch_retrieve(model_cfg, cond, args.out)
                        if s:
                            summaries.append(s)
                except Exception as e:
                    print(f"✗ BATCH FAILED {model_cfg['display_name']}/"
                          f"{cond}: {e}")
                    fatal.append((model_cfg["display_name"], cond))

    if summaries:
        MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(summaries).to_csv(MANIFEST_PATH, index=False,
                                       encoding="utf-8")
        print(f"\n📋 Manifest -> {MANIFEST_PATH}")
        if any("est_cost_usd" in s for s in summaries):
            total = sum(s.get("est_cost_usd", 0) for s in summaries)
            print(f"💰 Estimated total cost so far: ${total:.2f}")

    if fatal:
        print(f"\n✗ {len(fatal)} batch cell(s) FAILED: {fatal}")
        sys.exit(1)


if __name__ == "__main__":
    main()
