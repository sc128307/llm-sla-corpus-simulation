# The Alignment Paradox: LLM-Simulated Learner English

This repository provides the code and configuration needed to reproduce the
experiment of the paper "Prompting against Alignment: A Corpus-Based Study of
LLM-Simulated Learner English across Four Intervention Conditions"
(submitted to *Corpus-based Studies across Humanities*, De Gruyter).

## What this repo contains

- **Configuration** — the four prompt-intervention conditions and the six
  model configurations.
  - `configs/prompts.yaml` — zero-shot, one-shot, few-shot (k=3),
    chain-of-thought prompt templates.
  - `configs/models.yaml` — the six models, their API routing, and the
    reasoning-disable settings.
- **Generation** — the script that builds the paired synthetic corpus
  (6 models × 4 conditions × 5,600 essays) and the exemplar bank.
  - `src/generate.py`, `scripts/build_exemplar_bank.py`,
    `scripts/gpt_batch_orchestrate.py`.
- **Feature extraction** — Biber multi-dimensional probes and the
  grammar/Discourse indices.
  - `src/metrics.py`, `scripts/extract_features_v2.py`,
    `scripts/extract_neosca_pandas.py`, `scripts/merge_features.py`,
    `scripts/merge_neosca.py`.
- **Statistics** — paired bootstrap, JS divergence, effect sizes, and the
  downstream L1 classifier.
  - `scripts/phase4_statistics.py`, `scripts/robustness_checks.py`,
    `scripts/build_l1_dataset.py`, `scripts/train_l1_classifiers.py`,
    `scripts/plot_dose_response.py`, `scripts/plot_sim2real.py`.

## What is NOT in this repo

The data needed to run the pipeline are **not** committed (see `.gitignore`):
- **ICNALE** learner corpus (copyrighted) — obtain it separately.
- **Generated** synthetic corpora — produced by the generation script.
- **Model weights** (DeBERTa, RoBERTa-CoLA) and **neosca / stanza** models —
  downloaded separately.
- **API keys** — set them in a local `.env` (never committed).

## Reproducing

1. Install dependencies: `pip install -r requirements.txt` (plus the spaCy
   model `python -m spacy download en_core_web_sm` and the stanza/neosca
   models, see the comments in `requirements.txt`).
2. Place the ICNALE data path and API keys in a local `.env`.
3. Run the pipeline stages in order (generation → feature extraction →
   statistics / classifier). The scripts document their CLI arguments.

## Outputs

Run results (effect sizes, JS divergence, L1 classification accuracy) are
written to `data/results/` and can be regenerated with the scripts above.

## License / data

The research code is provided for reproducibility. The ICNALE corpus remains
copyrighted by its authors; the pre-trained model weights remain under their
respective licenses.
