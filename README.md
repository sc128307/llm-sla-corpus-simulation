# The Alignment Paradox: LLM-Simulated Learner English

This repository provides the code and configuration needed to reproduce the
experiment of the paper *Prompting against Alignment: A Corpus-Based Study of
LLM-Simulated Learner English across Four Intervention Conditions*.

## Contents

- **Configuration** — `configs/`: the four prompt-intervention conditions
  (`prompts.yaml`) and the six model configurations with reasoning-disable
  settings (`models.yaml`).
- **Generation** — `src/generate.py`, `scripts/build_exemplar_bank.py`,
  `scripts/gpt_batch_orchestrate.py`: build the paired synthetic corpus
  (6 models × 4 conditions × 5,600 essays) from the ICNALE baseline.
- **Feature extraction** — `src/metrics.py`, `scripts/extract_features_v2.py`,
  `scripts/extract_neosca_pandas.py`, `scripts/merge_features.py`,
  `scripts/merge_neosca.py`: Biber multi-dimensional probes (lexical,
  syntactic-stylistic, grammatical, discourse).
- **Statistics and downstream probe** — `scripts/phase4_statistics.py`,
  `scripts/robustness_checks.py`, `scripts/build_l1_dataset.py`,
  `scripts/train_l1_classifiers.py`, `scripts/train_grammar_cola.py`,
  `scripts/plot_dose_response.py`, `scripts/plot_sim2real.py`.

## Reproducing

1. Install dependencies: `pip install -r requirements.txt` (plus the spaCy
   model `python -m spacy download en_core_web_sm`; see the comments in
   `requirements.txt` for the neosca/stanza models).
2. Set API keys in a local `.env` and point `src/data_loader.py` at the
   ICNALE data.
3. Run the pipeline stages in order (generation → feature extraction →
   statistics / L1 classifier). Each script documents its CLI arguments.

## Note on data

The ICNALE corpus (copyrighted) and the generated synthetic corpora are not
committed; they are produced or obtained by running the pipeline. Model
weights (DeBERTa, RoBERTa-CoLA, stanza) are downloaded separately.
