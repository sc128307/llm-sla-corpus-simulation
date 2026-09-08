"""Paired comparison of raw and explicit-region-masked Primary L1 results."""
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "l1_classification_summary_primary.csv"
ANON = ROOT / "results" / "l1_classification_summary_primary_anonymized.csv"
OUT_CSV = ROOT / "results" / "l1_primary_raw_vs_anonymized.csv"
OUT_COND = ROOT / "results" / "l1_primary_raw_vs_anonymized_condition_summary.csv"
OUT_MODEL = ROOT / "results" / "l1_primary_raw_vs_anonymized_model_summary.csv"
OUT_JSON = ROOT / "results" / "l1_primary_raw_vs_anonymized_comparison.json"

CONDITIONS = ["zero_shot", "one_shot", "few_shot", "cot", "eqmix", "fullmix"]
MODELS = {
    "DeepSeek-V4-Flash", "Gemini-2.5-Flash", "GPT-5.6-Luna",
    "Llama-4-Maverick", "MiniMax-M3", "Qwen-3.7-Plus"
}


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


def parse(name):
    if name == "oracle_human":
        return "human", "oracle"
    value = name.removeprefix("sim_")
    for condition in sorted(CONDITIONS, key=len, reverse=True):
        suffix = "_" + condition
        if value.endswith(suffix):
            model = value[:-len(suffix)]
            if model not in MODELS:
                raise ValueError(f"unexpected model: {name}")
            return model, condition
    raise ValueError(f"cannot parse classifier: {name}")


def add_grouping(df):
    parsed = df["classifier"].map(parse)
    df["model"] = [x[0] for x in parsed]
    df["condition"] = [x[1] for x in parsed]
    return df


def summary(grouped):
    out = grouped.agg(
        n=("classifier", "size"),
        raw_accuracy_mean=("raw_accuracy", "mean"),
        anonymized_accuracy_mean=("anonymized_accuracy", "mean"),
        accuracy_delta_mean=("accuracy_delta", "mean"),
        accuracy_delta_sd=("accuracy_delta", "std"),
        accuracy_delta_min=("accuracy_delta", "min"),
        accuracy_delta_max=("accuracy_delta", "max"),
        raw_f1_mean=("raw_f1", "mean"),
        anonymized_f1_mean=("anonymized_f1", "mean"),
        f1_delta_mean=("f1_delta", "mean"),
        f1_delta_sd=("f1_delta", "std"),
        f1_delta_min=("f1_delta", "min"),
        f1_delta_max=("f1_delta", "max"),
    ).reset_index()
    return out


def main():
    raw = add_grouping(pd.read_csv(RAW).rename(columns={
        "accuracy": "raw_accuracy", "f1": "raw_f1"}))
    anon = pd.read_csv(ANON).rename(columns={
        "accuracy": "anonymized_accuracy", "f1": "anonymized_f1"})
    if set(raw.classifier) != set(anon.classifier):
        raise ValueError("raw and anonymized classifier sets differ")
    merged = raw.merge(anon, on="classifier", validate="one_to_one")
    merged["accuracy_delta"] = merged["anonymized_accuracy"] - merged["raw_accuracy"]
    merged["f1_delta"] = merged["anonymized_f1"] - merged["raw_f1"]
    merged["accuracy_drop"] = -merged["accuracy_delta"]
    merged["f1_drop"] = -merged["f1_delta"]
    merged.to_csv(OUT_CSV, index=False)

    condition = summary(merged[merged.condition != "oracle"].groupby("condition"))
    condition.to_csv(OUT_COND, index=False)
    model = summary(merged[merged.condition != "oracle"].groupby("model"))
    model.to_csv(OUT_MODEL, index=False)

    synth = merged[merged.condition != "oracle"]
    oracle = merged[merged.condition == "oracle"]
    result = {
        "status": "passed",
        "raw_summary_sha256": sha256(RAW),
        "anonymized_summary_sha256": sha256(ANON),
        "n_classifiers": int(len(merged)),
        "n_synthetic": int(len(synth)),
        "n_oracle": int(len(oracle)),
        "overall_synthetic": {
            "raw_accuracy_mean": float(synth.raw_accuracy.mean()),
            "anonymized_accuracy_mean": float(synth.anonymized_accuracy.mean()),
            "accuracy_delta_mean": float(synth.accuracy_delta.mean()),
            "raw_f1_mean": float(synth.raw_f1.mean()),
            "anonymized_f1_mean": float(synth.anonymized_f1.mean()),
            "f1_delta_mean": float(synth.f1_delta.mean()),
            "accuracy_drop_positive_count": int((synth.accuracy_drop > 0).sum()),
            "f1_drop_positive_count": int((synth.f1_drop > 0).sum()),
        },
        "oracle": oracle[["classifier", "raw_accuracy", "anonymized_accuracy",
                           "accuracy_delta", "raw_f1", "anonymized_f1", "f1_delta"]]
        .to_dict("records"),
        "outputs": {
            "paired": str(OUT_CSV.relative_to(ROOT)),
            "condition": str(OUT_COND.relative_to(ROOT)),
            "model": str(OUT_MODEL.relative_to(ROOT)),
        },
    }
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
