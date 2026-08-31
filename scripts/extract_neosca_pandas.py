"""Phase 4 neosca feature extraction: MLT + Clause_per_Sentence on ALL texts.

Runs in neosca_clean env (CUDA torch + stanza 1.14.0 + neosca in libs/ + pandas).
  - input:  data/results/features_all.parquet (keys from nlp-corpus run)
  - output: data/results/neosca_{A,B}.csv (appended every CHECKPOINT rows)
  - resume: data/results/neosca_{A,B}_done.json
Design: 2 GPU processes, incremental save, checkpoint resume, gc after file.
Usage:  python scripts/extract_neosca_pandas.py A
"""
import sys, os, time, json, gc
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "libs"))
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

RES = "data/results"
CHECKPOINT = 100


def load_done(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return set(json.load(f))
    return set()


def save_done(path, done):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sorted(done), f)


def append_csv(path, rows):
    df = pd.DataFrame(rows)
    if os.path.exists(path):
        old = pd.read_csv(path)
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(path, index=False, encoding="utf-8")


def analyze_with_neosca(sca, text):
    sca.run_on_text(str(text)[:3000])
    vals = sca.counters[-1].get_all_values()
    return (vals.get("MLT"), vals.get("C/S"))


def main():
    which = sys.argv[1]
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    import torch
    from neosca.ns_sca.ns_sca import Ns_SCA
    sca = Ns_SCA()
    print(f"[neosca {which}] cuda={torch.cuda.is_available()}", flush=True)

    out_path = os.path.join(RES, f"neosca_{which}.csv")
    progress_path = os.path.join(RES, f"neosca_{which}_done.json")
    # C0-C3 reuse the C namespace but with own files to avoid write races
    if which in ("C0", "C1", "C2", "C3"):
        out_path = os.path.join(RES, f"neosca_C_{which}.csv")
        progress_path = os.path.join(RES, f"neosca_C_{which}_done.json")
    done = load_done(progress_path)
    print(f"[neosca {which}] resume: {len(done)} done", flush=True)

    # keys from feature matrix
    feat = pd.read_parquet(os.path.join(RES, "features_all.parquet"),
                           columns=["key", "region", "topic", "model",
                                    "condition"])
    if limit:
        feat = feat.head(limit)
        print(f"[neosca {which}] LIMIT {limit} rows (smoke test)", flush=True)
    # 3-shard split: A/B/C by row index % 3
    shard_ids = {"A": 0, "B": 1, "C": 2, "C0": 0, "C1": 1, "C2": 2, "C3": 3}
    shard = shard_ids.get(which, 0)
    if which in ("C0", "C1", "C2", "C3"):
        # C sub-splits: first take C's 1/3 (index %3 == 2), then %4 within it
        c_rows = feat.iloc[2::3].reset_index(drop=True)
        sub = {"C0": 0, "C1": 1, "C2": 2, "C3": 3}[which]
        my_feat = c_rows.iloc[sub::4].reset_index(drop=True)
        print(f"[neosca {which}] {len(my_feat)} rows assigned "
              f"(C-sub {sub}/4)", flush=True)
    else:
        my_feat = feat.iloc[shard::3].reset_index(drop=True)
        print(f"[neosca {which}] {len(my_feat)} rows assigned "
              f"(shard {shard}/3)", flush=True)

    buf = []
    n_new = 0
    t0 = time.time()

    # group by (model, condition); read each file's texts once
    groups = my_feat.groupby(["model", "condition"])
    for (model, cond), g in groups:
        if model == "HUMAN":
            h = pd.read_csv("data/interim/human_data_full.csv")
            text_by_key = {f"{r['id']}|{r['topic']}": str(r["text"])
                           for _, r in h.iterrows()}
            label = "HUMAN"
        else:
            path = os.path.join("data/raw/LLM_Generations",
                                f"Corpus_{model}_{cond}.csv")
            df = pd.read_csv(path, usecols=["source_id", "topic",
                                            "generated_text"])
            text_by_key = {f"{r['source_id']}|{r['topic']}":
                           str(r["generated_text"])
                           for _, r in df.iterrows()}
            label = f"{model}/{cond}"
        print(f"[neosca {which}] loaded {label}: {len(text_by_key)} texts, "
              f"group {len(g)} rows", flush=True)

        for _, r in g.iterrows():
            dk = f"{r['key']}|{r['model']}|{r['condition']}"
            if dk in done:
                continue
            text = text_by_key.get(r["key"], "")
            if not text:
                continue
            try:
                mlt, cps = analyze_with_neosca(sca, text)
            except Exception as e:
                print(f"  ✗ {dk}: {str(e)[:80]}", flush=True)
                continue
            if mlt is not None:
                buf.append({"key": r["key"], "region": r["region"],
                            "topic": r["topic"], "model": r["model"],
                            "condition": r["condition"],
                            "MLT": mlt, "Clause_per_Sentence": cps})
                done.add(dk)
                n_new += 1
            if len(buf) >= CHECKPOINT:
                append_csv(out_path, buf)
                save_done(progress_path, done)
                buf.clear()
        del text_by_key
        gc.collect()
        print(f"[neosca {which}] after {label}: +{n_new} new, "
              f"{time.time()-t0:.0f}s", flush=True)

    if buf:
        append_csv(out_path, buf)
        save_done(progress_path, done)
        buf.clear()
    print(f"[neosca {which}] DONE: {n_new} new, total {len(done)}, "
          f"{(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
