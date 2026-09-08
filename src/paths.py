"""Canonical project paths.

The workspace intentionally keeps data, derived features, results, figures,
and model checkpoints at the repository root.  All scripts should import
these constants instead of reconstructing the old ``data/...`` layout.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONFIG_DIR = PROJECT_ROOT / "configs"
CORPUS_DIR = PROJECT_ROOT / "corpus"
HUMAN_PATH = CORPUS_DIR / "human" / "human_data_full.csv"
GENERATED_DIR = CORPUS_DIR / "generated"
MANIFEST_DIR = CORPUS_DIR / "manifest"
SPLIT_MANIFEST_PATH = MANIFEST_DIR / "split_manifest.csv"
EXEMPLAR_POOL_PATH = MANIFEST_DIR / "exemplar_pool.csv"

FEATURE_DIR = PROJECT_ROOT / "features"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURE_DIR = PROJECT_ROOT / "figures"
PAPER_DIR = PROJECT_ROOT / "paper"
MODEL_DIR = PROJECT_ROOT / "models"
ARCHIVE_DIR = PROJECT_ROOT / "archive"

FEATURES_PATH = FEATURE_DIR / "features_all_with_neosca.parquet"

def generated_path(model: str, condition: str) -> Path:
    """Return the canonical generated-corpus path for a matrix cell."""
    return GENERATED_DIR / f"Corpus_{model}_{condition}.csv"

