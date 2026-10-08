"""Check integrity; optionally validate aggregate joins or excluded inputs."""
import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def integrity():
    inventory = read(ROOT / "inventory.json")
    expected = {item["path"] for item in inventory["files"]} | {"inventory.json"}
    actual = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*") if p.is_file()}
    # Generated outputs and privately supplied inputs are outside release content.
    actual = {p for p in actual if not p.startswith(("results/", "features/", "corpus/",
                                                    "models/", "notebook/", "libs/"))
              and "__pycache__" not in Path(p).parts}
    if actual != expected:
        raise ValueError(f"Package membership differs: {sorted(actual ^ expected)}")
    for item in inventory["files"]:
        path = ROOT / item["path"]
        if path.is_symlink() or sha(path) != item["sha256"] or path.stat().st_size != item["bytes"]:
            raise ValueError(f"Integrity mismatch: {item['path']}")
        if path.suffix == ".json":
            read(path)
        if path.suffix == ".py":
            ast.parse(path.read_text(encoding="utf-8-sig"), filename=item["path"])
        if "rows" in item:
            with path.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                rows = list(reader)
            if len(rows) != item["rows"] or reader.fieldnames != item["columns"]:
                raise ValueError(f"CSV schema/count mismatch: {item['path']}")
    return {"status": "passed", "files": len(expected), "hashes": "passed",
            "json_schema_counts_and_syntax": "passed"}


def aggregate_check():
    import pandas as pd

    with tempfile.TemporaryDirectory(prefix="minimum_repro_check_") as name:
        work = Path(name)
        (work / "scripts").mkdir()
        (work / "results").mkdir()
        for script in ("compare_primary_anonymized.py", "audit_bridge_inputs.py"):
            shutil.copyfile(ROOT / "scripts" / script, work / "scripts" / script)
        inputs = {"intrinsic_effect_sizes.csv": "effect_sizes.csv",
                  "regional_raw_summary.csv": "l1_classification_summary_primary.csv",
                  "regional_marker_replaced_summary.csv": "l1_classification_summary_primary_anonymized.csv"}
        for source, dest in inputs.items():
            shutil.copyfile(ROOT / "reference_results" / source, work / "results" / dest)
        for script in ("compare_primary_anonymized.py", "audit_bridge_inputs.py"):
            subprocess.run([sys.executable, str(work / "scripts" / script)], cwd=work,
                           check=True, capture_output=True, text=True, encoding="utf-8")
        for name, generated in (("marker_paired_comparison.csv", "l1_primary_raw_vs_anonymized.csv"),
                                ("marker_condition_summary.csv", "l1_primary_raw_vs_anonymized_condition_summary.csv")):
            pd.testing.assert_frame_equal(pd.read_csv(ROOT / "reference_results" / name),
                                          pd.read_csv(work / "results" / generated),
                                          check_exact=False, rtol=1e-10, atol=1e-12)
        audit = read(work / "results" / "bridge_input_audit.json")
        if audit["status"] != "passed" or len(audit["joined_cells"]) != 24:
            raise ValueError("Bridge join audit failed")
        expected = pd.read_csv(ROOT / "reference_results" / "bridge_cells.csv")
        actual = pd.DataFrame(audit["joined_cells"])[expected.columns]
        pd.testing.assert_frame_equal(expected, actual, check_exact=False,
                                      rtol=1e-10, atol=1e-12)
    return {"marker_pairs": 37, "marker_conditions": 6, "bridge_cells": 24,
            "numerical_parity": "passed", "tolerance": "rtol=1e-10; atol=1e-12",
            "full_experiment_rerun": False}


def inputs_check(stage):
    contract = read(ROOT / "inputs_contract.json")
    required = [item for item in contract["external_inputs"] if stage in item["stages"]]
    failures = []
    for item in required:
        path = ROOT / item["path"]
        if not path.is_file():
            failures.append({"path": item["path"], "reason": "missing"})
        elif path.is_symlink() or sha(path) != item["sha256"]:
            failures.append({"path": item["path"], "reason": "hash mismatch"})
    if failures:
        print(json.dumps({"status": "blocked", "stage": stage, "failures": failures}, indent=2))
        raise SystemExit(2)
    return {"stage": stage, "matched_inputs": len(required),
            "note": "Input hashes only; separately verify parser/model environments and output schemas."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate-check", action="store_true")
    parser.add_argument("--inputs-stage", choices=("features", "intrinsic", "classifiers", "writer"))
    args = parser.parse_args()
    result = integrity()
    if args.aggregate_check:
        result["aggregate_check"] = aggregate_check()
    if args.inputs_stage:
        result["inputs"] = inputs_check(args.inputs_stage)
    print(json.dumps(result, indent=2))
