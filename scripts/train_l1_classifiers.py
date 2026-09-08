"""Phase 5: train DeBERTa-v3-base L1 classifiers on the Sim datasets + Oracle.

Each classifier: fine-tune on the dataset's train split, early-stop on dev,
evaluate on the FIXED gold test (human test split, disjoint from all training).
Reports accuracy + macro-F1 per classifier.

Config: model=microsoft/deberta-v3-base, batch=4, grad_acc=4 (eff 16),
  lr=2e-5, epochs=3, seed=42, fp16 (CUDA). The required max length is passed
  from the tokenizer audit; essays longer than the model window are chunked
  and evaluated by essay-level logit aggregation rather than truncated.

Output: results/l1_classification_summary.csv
"""
import sys, os, gc, json, argparse, re
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import torch
from transformers import (AutoConfig, AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments, DataCollatorWithPadding,
                          set_seed)
from sklearn.metrics import accuracy_score, f1_score
from src.paths import MODEL_DIR, RESULTS_DIR

PARSER = argparse.ArgumentParser()
PARSER.add_argument("--dataset-dir", default=str(RESULTS_DIR / "l1_dataset"))
PARSER.add_argument("--output-dir", default=str(RESULTS_DIR / "l1_models"))
PARSER.add_argument("--summary-path", default=str(RESULTS_DIR / "l1_classification_summary.csv"))
PARSER.add_argument("--max-length", type=int, required=True,
                    help="token audit result; do not use an unverified default")
PARSER.add_argument("--seed", type=int, default=42)
PARSER.add_argument("--skip-existing", action="store_true",
                    help="skip classifiers with a complete checkpoint and prior metrics")
PARSER.add_argument("--prior-log", default=None,
                    help="stdout log containing metrics from a previous interrupted run")
ARGS = PARSER.parse_args()

DS = ARGS.dataset_dir
MODEL_PATH = str(MODEL_DIR / "deberta-v3-base" / "deberta-v3-base")
if not os.path.isdir(MODEL_PATH):
    MODEL_PATH = str(MODEL_DIR / "deberta-v3-base")
os.environ["HF_HOME"] = str(MODEL_DIR / ".hf_home")

MAX_LEN = ARGS.max_length
# Conservative settings for an 8-GB RTX 4060 Laptop GPU.  The effective
# optimisation batch remains 16 while the per-device activation footprint is
# halved relative to the original batch=8 configuration.
BATCH = 4
GRAD_ACC = 4
LR = 2e-5
EPOCHS = 3
SEED = ARGS.seed
set_seed(SEED)

MODEL_CONFIG = AutoConfig.from_pretrained(MODEL_PATH, local_files_only=True)
MODEL_MAX_LEN = int(getattr(MODEL_CONFIG, "max_position_embeddings", 512))
if MAX_LEN > MODEL_MAX_LEN:
    raise ValueError(
        f"--max-length={MAX_LEN} exceeds model max_position_embeddings={MODEL_MAX_LEN}; "
        "use the model limit and essay-level overflow chunking"
    )

# get all classifier datasets
ds_names = [f[:-8] for f in os.listdir(DS) if f.endswith(".parquet")
            and f != "manifest.json"]
print(f"classifiers to train: {len(ds_names)}")


class L1ChunkDataset(torch.utils.data.Dataset):
    """Chunk long essays without dropping tokens; metrics aggregate by essay."""
    def __init__(self, texts, labels, groups, tokenizer):
        self.items = []
        self.labels = []
        self.group_ids = []
        special_count = tokenizer.num_special_tokens_to_add(pair=False)
        window = MAX_LEN - special_count
        if window <= 0:
            raise ValueError("max length is too small for tokenizer special tokens")
        for text, label, group in zip(texts, labels, groups):
            ids = tokenizer.encode(
                str(text), add_special_tokens=False, truncation=False
            )
            if not ids:
                ids = [tokenizer.unk_token_id or tokenizer.pad_token_id]
            for start in range(0, len(ids), window):
                chunk = ids[start:start + window]
                encoded = tokenizer.prepare_for_model(
                    chunk,
                    add_special_tokens=True,
                    truncation=False,
                    return_attention_mask=True,
                    return_token_type_ids=True,
                )
                self.items.append(encoded)
                self.labels.append(int(label))
                self.group_ids.append(str(group))

    def __getitem__(self, idx):
        item = {k: torch.tensor(v) for k, v in self.items[idx].items()
                if k in {"input_ids", "attention_mask", "token_type_ids"}}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def essay_metrics(prediction, group_ids):
    """Aggregate chunk logits to essay-level accuracy and macro-F1."""
    logits = prediction.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    logits = np.asarray(logits)
    labels = np.asarray(prediction.label_ids)
    sums, counts, gold = {}, {}, {}
    for i, group in enumerate(group_ids):
        sums[group] = sums.get(group, 0) + logits[i]
        counts[group] = counts.get(group, 0) + 1
        gold[group] = int(labels[i])
    groups = list(sums)
    y_pred = [int(np.argmax(sums[g] / counts[g])) for g in groups]
    y_true = [gold[g] for g in groups]
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="macro"),
    }


def train_one(name, df):
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, local_files_only=True, use_fast=False
    )
    # label mapping from dataset (assume 11 classes stored as ints)
    n_labels = df["label"].max() + 1
    # Transformers 4.57.1 forwards unknown constructor kwargs to the
    # DeBERTa class, which rejects ``num_labels``.  Set it on the config and
    # explicitly allow replacement of the two-label base classification head.
    model_config = AutoConfig.from_pretrained(MODEL_PATH, local_files_only=True)
    model_config.num_labels = int(n_labels)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,
        config=model_config,
        local_files_only=True,
        ignore_mismatched_sizes=True,
    )
    if torch.cuda.is_available():
        model.cuda()

    train_df = df[df.split == "train"]
    dev_df = df[df.split == "dev"]
    test_df = df[df.split == "test"]

    train_ds = L1ChunkDataset(
        train_df.text.tolist(), train_df.label.tolist(), train_df.key.tolist(), tokenizer
    )
    dev_ds = L1ChunkDataset(
        dev_df.text.tolist(), dev_df.label.tolist(), dev_df.key.tolist(), tokenizer
    )
    test_ds = L1ChunkDataset(
        test_df.text.tolist(), test_df.label.tolist(), test_df.key.tolist(), tokenizer
    )

    args = TrainingArguments(
        output_dir=os.path.join(ARGS.output_dir, name),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH,
        per_device_eval_batch_size=BATCH * 2,
        gradient_accumulation_steps=GRAD_ACC,
        learning_rate=LR,
        fp16=torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="none",
        seed=SEED,
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=dev_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=lambda p: essay_metrics(p, dev_ds.group_ids),
    )
    resume_checkpoint = None
    if ARGS.skip_existing and name not in prior:
        resume_checkpoint = latest_checkpoint(name)
        if resume_checkpoint:
            print(f"  {name}: resuming from {resume_checkpoint}", flush=True)
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    trainer.compute_metrics = lambda p: essay_metrics(p, test_ds.group_ids)
    metrics = trainer.evaluate(test_ds)
    del model, trainer, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return metrics.get("eval_accuracy"), metrics.get("eval_f1")


def complete_checkpoint(name):
    """Return true only when a saved model and trainer state are both present."""
    root = os.path.join(ARGS.output_dir, name)
    if not os.path.isdir(root):
        return False
    for current, _, files in os.walk(root):
        if "model.safetensors" in files and "trainer_state.json" in files:
            return True
    return False


def latest_checkpoint(name):
    """Return the latest resumable checkpoint, if an interrupted run exists."""
    root = os.path.join(ARGS.output_dir, name)
    candidates = []
    for current, _, files in os.walk(root):
        if "model.safetensors" in files and "trainer_state.json" in files:
            state_path = os.path.join(current, "trainer_state.json")
            try:
                with open(state_path, encoding="utf-8") as handle:
                    state = json.load(handle)
                candidates.append((int(state.get("global_step", -1)), current))
            except (OSError, ValueError, TypeError):
                continue
    return max(candidates, default=(-1, None))[1]


def prior_results():
    """Recover completed metrics so an interrupted matrix can be resumed."""
    found = {}
    if os.path.exists(ARGS.summary_path):
        old = pd.read_csv(ARGS.summary_path)
        for row in old.to_dict("records"):
            found[str(row["classifier"])] = {
                "classifier": str(row["classifier"]),
                "accuracy": float(row["accuracy"]),
                "f1": float(row["f1"]),
            }
    # The first run was intentionally stopped before its final CSV write; use
    # the flushed per-classifier lines in the persistent stdout log.
    log_path = ARGS.prior_log or os.path.join(
        ROOT, "results", "l1_training_logs", "primary_stdout.log"
    )
    if not os.path.exists(ARGS.summary_path) and os.path.exists(log_path):
        pattern = re.compile(r"^  (.+): acc=([0-9.]+) f1=([0-9.]+)$")
        with open(log_path, encoding="utf-8") as handle:
            for line in handle:
                match = pattern.match(line.rstrip())
                if match:
                    name, acc, f1 = match.groups()
                    found[name] = {"classifier": name, "accuracy": float(acc), "f1": float(f1)}
    return found


results = list(prior_results().values()) if ARGS.skip_existing else []
prior = {row["classifier"] for row in results}
if ARGS.skip_existing:
    print(f"prior completed metrics: {len(prior)}")
for name in ds_names:
    if ARGS.skip_existing and name in prior and complete_checkpoint(name):
        print(f"  {name}: skipped (complete checkpoint)", flush=True)
        continue
    df = pd.read_parquet(os.path.join(DS, f"{name}.parquet"))
    acc, f1 = train_one(name, df)
    results.append({"classifier": name, "accuracy": acc, "f1": f1})
    print(f"  {name}: acc={acc:.4f} f1={f1:.4f}", flush=True)

out = pd.DataFrame(results).drop_duplicates("classifier", keep="last")
order = {name: i for i, name in enumerate(ds_names)}
out["_order"] = out["classifier"].map(order)
out = out.sort_values("_order").drop(columns="_order")
out.to_csv(ARGS.summary_path, index=False)
print(f"\nDONE -> {ARGS.summary_path} ({len(out)} rows)")
print(out[["classifier", "accuracy"]].round(4).to_string())
