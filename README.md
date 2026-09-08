# LLM–SLA Corpus Simulation: Code Release

This repository contains the code and configuration for the study
“Can Prompting Reproduce Learner-Language Distributions? A Corpus-Based Study
of Instruction-Tuned LLM Outputs.” It is a workflow release for users who have
lawful access to the required corpus and local model resources.

## Contents

- `configs/` — the four prompt blocks (C0–C3) and provider/model settings.
- `src/` — corpus loading, generation clients, canonical paths, and metrics.
- `scripts/` — exemplar construction, feature extraction and NeoSCA merging,
  paired bootstrap/sign-flip statistics, robustness checks, Primary ten-class
  L1 dataset construction and training, and marker-masking sensitivity.
- `requirements.txt` and `LICENSE`.

The release does not contain essays, generated corpora, feature tables,
classifier datasets or checkpoints, manuscript files, result tables, figure
scripts, or internal project documents.

## Required local inputs

Place lawful local inputs under the layout defined in `src/paths.py`:

```text
corpus/human/human_data_full.csv
corpus/generated/Corpus_<model>_<condition>.csv
corpus/manifest/split_manifest.csv
corpus/manifest/exemplar_pool.csv
features/features_all_with_neosca.parquet
models/                         # local grammar/L1 resources
results/                        # writable outputs
```

Feature extraction additionally requires `cefr_full_vocab.csv` (configure
`CEFR_VOCAB_DIR`) and a local NeoSCA/Stanza installation (`NEOSCA_HOME` when
needed). The ICNALE essays must be accessed and cited under the official
terms; cite Ishikawa (2023).

## Workflow

From the repository root, run the non-generation stages in this order:

```text
python scripts/build_exemplar_bank.py
python scripts/extract_features_v2.py A
python scripts/extract_features_v2.py B
python scripts/extract_neosca_pandas.py A
python scripts/extract_neosca_pandas.py B
python scripts/merge_neosca.py
python scripts/merge_features.py
python scripts/phase4_statistics.py
python scripts/phase4_signflip_inference.py
python scripts/robustness_checks.py
python scripts/build_l1_dataset.py
python scripts/build_l1_anonymized.py
python scripts/train_l1_classifiers.py --max-length <audited-token-limit>
```

Generation is optional, provider-billed, and never starts as a side effect of
installation or of the workflow commands above. If regeneration is explicitly
authorised, use `python -m src.generate --help`, supply credentials only
through local environment variables, and verify provider-specific
reasoning/thinking-disable settings first.
