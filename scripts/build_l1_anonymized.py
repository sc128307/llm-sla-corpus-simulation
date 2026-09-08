"""Build anonymised L1-dataset copies for explicit region-marker sensitivity."""
import re
import shutil
import json
from pathlib import Path

import pandas as pd

from audit_l1_region_markers import MARKERS

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
SOURCES = [
    (RESULTS / "l1_dataset", RESULTS / "l1_dataset_anonymized"),
]


def compile_masker():
    words = sorted(
        {word for values in MARKERS.values() for word in values},
        key=len,
        reverse=True,
    )
    pattern = re.compile(
        r"(?<!\w)(?:" + "|".join(re.escape(word) for word in words) + r")(?!\w)",
        re.IGNORECASE,
    )
    return pattern


def main():
    masker = compile_masker()
    for source_dir, dest_dir in SOURCES:
        if not source_dir.exists():
            raise FileNotFoundError(source_dir)
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        dest_dir.mkdir(parents=True)
        for path in sorted(source_dir.glob("*.parquet")):
            df = pd.read_parquet(path)
            df["text"] = df["text"].fillna("").astype(str).map(
                lambda text: masker.sub("[REGION]", text)
            )
            df.to_parquet(dest_dir / path.name, index=False)
        manifest = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))
        manifest["task"] = str(manifest["task"]) + "_explicit_region_masked_sensitivity"
        manifest["anonymisation"] = {
            "replacement": "[REGION]",
            "scope": "human and synthetic train/dev/test text columns",
            "marker_dictionary": MARKERS,
            "canonical_source_unchanged": True,
        }
        (dest_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"built {dest_dir}")


if __name__ == "__main__":
    main()
