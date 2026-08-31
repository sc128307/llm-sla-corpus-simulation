"""Fine-tune a grammatical-acceptability classifier (RoBERTa on CoLA).

Theory (see docs/METHODOLOGY_JUSTIFICATION.md §5.1 + Zotero notes):
  Lau et al. 2017 — acceptability is a probabilistic continuum, not a
  discrete right/wrong; Warstadt et al. 2019 — neural classifiers can judge
  acceptability (CoLA). The grammar probe of the paper therefore uses a
  CoLA-fine-tuned RoBERTa outputting P(acceptable) per sentence, replacing
  the old 7-rule heuristic engine.

Artifacts (gitignored, under models/):
  models/grammar_cola/            model + tokenizer (AutoModelForSequenceClassification)
  models/grammar_cola/metadata.json  version info for the method section:
                                    base model, CoLA MCC, train/val sizes,
                                    epochs, seed, date, hardware.

Usage:
  python scripts/train_grammar_cola.py [--epochs 5] [--batch-size 32]
      [--lr 1e-5] [--seed 42] [--output models/grammar_cola]
"""
import argparse
import json
import os
import sys
import time
import urllib.request
import zipfile

os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402
from sklearn.metrics import matthews_corrcoef  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForSequenceClassification, AutoTokenizer, Trainer,
    TrainingArguments,
)
from transformers import EarlyStoppingCallback  # noqa: E402

COLA_URL = "https://nyu-mll.github.io/CoLA/cola_public_1.1.zip"
COLA_DIR = os.path.join(ROOT, "data", "interim", "cola")


def ensure_cola():
    """CoLA raw TSVs (nyu-mll, canonical source) -> data/interim/cola/.
    Columns: sentence_source, label, label_name, sentence."""
    raw = os.path.join(COLA_DIR, "cola_public", "raw")
    train_f = os.path.join(raw, "in_domain_train.tsv")
    if os.path.exists(train_f):
        return
    os.makedirs(COLA_DIR, exist_ok=True)
    zip_path = os.path.join(COLA_DIR, "cola_public_1.1.zip")
    if not os.path.exists(zip_path):
        print("downloading CoLA from nyu-mll...", flush=True)
        urllib.request.urlretrieve(COLA_URL, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(COLA_DIR)
    print("CoLA extracted to", COLA_DIR, flush=True)


def load_cola_split(tsv_path):
    df = pd.read_csv(tsv_path, sep="\t", header=None,
                     names=["sentence_source", "label", "label_name",
                            "sentence"])
    return df[["label", "sentence"]]


def ensure_model_local(model_name):
    """Copy the base model into models/.hf_models/ (local_dir mode — no
    symlinks, which the sandbox blocks)."""
    local = os.path.join(ROOT, "models", ".hf_models",
                         model_name.replace("/", "--"))
    if not os.path.exists(os.path.join(local, "config.json")):
        os.makedirs(os.path.dirname(local), exist_ok=True)
        print(f"downloading {model_name} -> {local} (local_dir, no "
              f"symlinks)", flush=True)
        snapshot_download(model_name, local_dir=local)
    return local


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="roberta-base")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default=os.path.join(ROOT, "models",
                                                     "grammar_cola"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ensure_cola()
    raw = os.path.join(COLA_DIR, "cola_public", "raw")
    train = load_cola_split(os.path.join(raw, "in_domain_train.tsv"))
    val = load_cola_split(os.path.join(raw, "in_domain_dev.tsv"))
    print(f"train: {len(train)} | val: {len(val)}", flush=True)

    model_local = ensure_model_local(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_local)

    def tok(ex):
        return tokenizer(ex["sentence"], truncation=True,
                         max_length=128, padding="max_length")

    import datasets as hf_datasets
    train_ds = hf_datasets.Dataset.from_pandas(train).map(tok, batched=True)
    val_ds = hf_datasets.Dataset.from_pandas(val).map(tok, batched=True)
    train_ds = train_ds.select_columns(["input_ids", "attention_mask",
                                        "label"])
    val_ds = val_ds.select_columns(["input_ids", "attention_mask", "label"])
    train_ds.set_format("torch")
    val_ds.set_format("torch")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_local, num_labels=2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"mcc": matthews_corrcoef(labels, preds)}

    out_dir = args.output
    os.makedirs(out_dir, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=0.06,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="mcc",
        save_total_limit=2,
        logging_steps=100,
        seed=args.seed,
        report_to=[],
    )
    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    t0 = time.time()
    trainer.train()
    train_min = round((time.time() - t0) / 60, 1)

    final = trainer.evaluate()
    mcc = float(final["eval_mcc"])
    print(f"CoLA MCC = {mcc:.4f} | train time {train_min} min", flush=True)

    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    meta = {
        "engine": "roberta-cola",
        "base_model": args.model_name,
        "cola_mcc": round(mcc, 4),
        "cola_val_n": len(val),
        "train_n": len(train),
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "train_min": train_min,
        "date": time.strftime("%Y-%m-%d"),
        "device": device,
        "note": "sentence-level P(acceptable); grammar probe = mean over "
                "sentences. See docs/METHODOLOGY_JUSTIFICATION.md",
    }
    with open(os.path.join(out_dir, "metadata.json"), "w",
              encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"model + metadata -> {out_dir}")


if __name__ == "__main__":
    main()
