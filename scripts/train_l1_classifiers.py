"""Phase 5: train DeBERTa-v3-base L1 classifiers on the 36 Sim datasets + Oracle.

Each classifier: fine-tune on the dataset's train split, early-stop on dev,
evaluate on the FIXED gold test (human test split, disjoint from all training).
Reports accuracy + macro-F1 per classifier.

Config (from original paper, cell 14):
  model=microsoft/deberta-v3-base, max_len=256, batch=8, grad_acc=2 (eff 16),
  lr=2e-5, epochs=3, seed=42, fp16 (CUDA).

Output: data/results/l1_classification_summary.csv
"""
import sys, os, gc, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import torch
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments, DataCollatorWithPadding,
                          set_seed)
from sklearn.metrics import accuracy_score, f1_score

DS = "data/results/l1_dataset"
MODEL_PATH = "models/deberta-v3-base"
os.environ["HF_HOME"] = ROOT + "/models/.hf_home"

MAX_LEN = 256
BATCH = 8
GRAD_ACC = 2
LR = 2e-5
EPOCHS = 3
SEED = 42
set_seed(SEED)

# get all classifier datasets
ds_names = [f[:-8] for f in os.listdir(DS) if f.endswith(".parquet")
            and f != "manifest.json"]
print(f"classifiers to train: {len(ds_names)}")


class L1Dataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(texts, truncation=True, padding=True,
                                   max_length=MAX_LEN)
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def train_one(name, df):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    # label mapping from dataset (assume 11 classes stored as ints)
    n_labels = df["label"].max() + 1
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH, num_labels=n_labels)
    if torch.cuda.is_available():
        model.cuda()

    train_df = df[df.split == "train"]
    dev_df = df[df.split == "dev"]
    test_df = df[df.split == "test"]

    train_ds = L1Dataset(train_df.text.tolist(), train_df.label.tolist(),
                         tokenizer)
    dev_ds = L1Dataset(dev_df.text.tolist(), dev_df.label.tolist(), tokenizer)
    test_ds = L1Dataset(test_df.text.tolist(), test_df.label.tolist(),
                        tokenizer)

    args = TrainingArguments(
        output_dir=f"data/results/l1_models/{name}",
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
        compute_metrics=lambda p: {
            "accuracy": accuracy_score(p.label_ids,
                                       np.argmax(p.predictions, axis=1)),
            "f1": f1_score(p.label_ids, np.argmax(p.predictions, axis=1),
                           average="macro")},
    )
    trainer.train()
    metrics = trainer.evaluate(test_ds)
    del model, trainer, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return metrics.get("eval_accuracy"), metrics.get("eval_f1")


results = []
for name in ds_names:
    df = pd.read_parquet(os.path.join(DS, f"{name}.parquet"))
    acc, f1 = train_one(name, df)
    results.append({"classifier": name, "accuracy": acc, "f1": f1})
    print(f"  {name}: acc={acc:.4f} f1={f1:.4f}", flush=True)

out = pd.DataFrame(results)
out.to_csv("data/results/l1_classification_summary.csv", index=False)
print(f"\nDONE -> l1_classification_summary.csv ({len(out)} rows)")
print(out[["classifier", "accuracy"]].round(4).to_string())
