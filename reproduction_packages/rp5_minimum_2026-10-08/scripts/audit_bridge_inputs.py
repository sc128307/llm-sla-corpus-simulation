"""Audit the frozen inputs for the exploratory 24-cell feature-transfer bridge.

This script performs only a deterministic join audit. It does not regenerate
text, recompute features, train classifiers, or estimate the bridge relation.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EFFECTS = ROOT / "results" / "effect_sizes.csv"
CLASSIFIERS = ROOT / "results" / "l1_classification_summary_primary.csv"
OUTPUT = ROOT / "results" / "bridge_input_audit.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def classifier_key(classifier: str, models: list[str]) -> tuple[str, str] | None:
    if not classifier.startswith("sim_"):
        return None
    body = classifier[4:]
    for model in sorted(models, key=len, reverse=True):
        prefix = model + "_"
        if body.startswith(prefix):
            return model, body[len(prefix) :]
    return None


def main() -> None:
    effects = read_csv(EFFECTS)
    classifiers = read_csv(CLASSIFIERS)
    primary_effects = [row for row in effects if row.get("analysis_set") == "l2_primary"]
    models = sorted({row["model"] for row in primary_effects})
    conditions = sorted({row["condition"] for row in primary_effects})

    feature_counts = Counter((row["model"], row["condition"]) for row in primary_effects)
    effect_keys = sorted(feature_counts)
    duplicate_effect_rows = [key for key, count in feature_counts.items() if count != 22]

    classifier_rows: dict[tuple[str, str], dict[str, str]] = {}
    excluded = []
    duplicate_classifier_keys = []
    for row in classifiers:
        key = classifier_key(row.get("classifier", ""), models)
        if key is None or key[1] not in conditions:
            excluded.append(row.get("classifier", ""))
            continue
        if key in classifier_rows:
            duplicate_classifier_keys.append(key)
        classifier_rows[key] = row

    classifier_keys = sorted(classifier_rows)
    missing_classifier_keys = sorted(set(effect_keys) - set(classifier_rows))
    extra_classifier_keys = sorted(set(classifier_rows) - set(effect_keys))

    joins = []
    for key in effect_keys:
        model, condition = key
        rows = [r for r in primary_effects if (r["model"], r["condition"]) == key]
        dz = [float(r["d_z"]) for r in rows]
        outcome = classifier_rows.get(key)
        joins.append(
            {
                "model": model,
                "condition": condition,
                "n_features": len(rows),
                "mean_abs_d_z": sum(abs(x) for x in dz) / len(dz),
                "classifier": outcome["classifier"] if outcome else None,
                "accuracy": float(outcome["accuracy"]) if outcome else None,
                "macro_f1": float(outcome["f1"]) if outcome else None,
            }
        )

    passed = (
        len(primary_effects) == 24 * 22
        and len(models) == 6
        and len(conditions) == 4
        and not duplicate_effect_rows
        and len(effect_keys) == 24
        and len(classifier_keys) == 24
        and not duplicate_classifier_keys
        and not missing_classifier_keys
        and not extra_classifier_keys
        and all(row["macro_f1"] is not None for row in joins)
    )

    result = {
        "status": "passed" if passed else "blocked",
        "analysis": "bridge_input_audit",
        "scope": "l2_primary 24 model-by-condition cells; 22 frozen intrinsic features; Primary held-out outcomes",
        "inputs": {
            "effect_sizes": str(EFFECTS.relative_to(ROOT)).replace("\\", "/"),
            "effect_sizes_sha256": sha256(EFFECTS),
            "classifier_summary": str(CLASSIFIERS.relative_to(ROOT)).replace("\\", "/"),
            "classifier_summary_sha256": sha256(CLASSIFIERS),
        },
        "counts": {
            "effect_rows_l2_primary": len(primary_effects),
            "expected_effect_rows": 528,
            "models": models,
            "conditions": conditions,
            "effect_cells": len(effect_keys),
            "classifier_cells": len(classifier_keys),
        "excluded_classifier_rows": len(excluded),
        "expected_excluded_classifier_rows": 13,
        },
        "excluded_classifier_rows": excluded,
        "duplicate_effect_cells": [list(key) for key in duplicate_effect_rows],
        "duplicate_classifier_cells": [list(key) for key in duplicate_classifier_keys],
        "missing_classifier_cells": [list(key) for key in missing_classifier_keys],
        "extra_classifier_cells": [list(key) for key in extra_classifier_keys],
        "joined_cells": joins,
        "estimand_boundary": {
            "included": "l2_primary; zero_shot, one_shot, few_shot, cot; raw Primary classifier outcomes",
            "excluded": "eqmix, fullmix, human oracle, ENS, anonymised marker-sensitivity rows",
            "predictor": "mean absolute d_z across 22 features; lower means closer profile agreement",
            "outcome_primary": "macro_f1",
            "outcome_secondary": "accuracy",
        },
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(OUTPUT), "cells": len(joins)}, ensure_ascii=False))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
