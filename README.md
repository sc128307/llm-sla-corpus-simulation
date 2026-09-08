# v2 analysis code release

This is the minimal code-and-configuration release for the study
“Can Prompting Reproduce Learner-Language Distributions? A Corpus-Based Study
of Instruction-Tuned LLM Outputs.” It is intended for code inspection and for
re-running the workflow when the user has lawful access to the required input
data and model resources.

## What is included

- `configs/` — prompt blocks (C0--C3) and model/provider settings. Provider
  reasoning/thinking is disabled for the frozen configuration; C3 is a planning
  instruction, not an exposed chain-of-thought condition.
- `src/` — data loading, generation clients, canonical paths, and linguistic
  metrics.
- `scripts/` — the selected workflow entry points for exemplar construction,
  feature extraction, NeoSCA merging, statistics, sign-flip/bootstrap checks,
  robustness checks, L1 dataset construction, classifier training, and marker
  sensitivity analysis.
- `requirements.txt` and `LICENSE`.

Figure-generation scripts, manuscript source, internal project documents,
result tables, audit outputs, corpora, feature tables, and model checkpoints
are intentionally excluded.

## Data and model availability

The licensed ICNALE essays and the generated essay corpus are not redistributed.
Essay-level feature tables, classifier datasets, and local checkpoints are also
not included. The scripts therefore require user-supplied inputs under the
canonical layout defined in `src/paths.py`:

```text
corpus/human/human_data_full.csv
corpus/generated/Corpus_<model>_<condition>.csv
corpus/manifest/split_manifest.csv
corpus/manifest/exemplar_pool.csv
features/features_all_with_neosca.parquet
models/                         # local grammar/L1 model resources as needed
results/                        # writable output directory
```

Feature extraction additionally requires a user-supplied
`cefr_full_vocab.csv` (set `CEFR_VOCAB_DIR`) and, for NeoSCA features, a local
NeoSCA/Stanza installation (set `NEOSCA_HOME` when it is not on `PYTHONPATH`).

ICNALE access and citation must follow the official terms and the study's formal
citation (Ishikawa, 2023). Generation scripts require the user's own provider
credentials and may incur charges; credentials must be supplied through local
environment variables and must never be committed.

Because data and frozen result tables are not distributed, this repository is a
workflow/code release rather than a self-contained numerical replication
archive. Published numerical claims should be checked against the manuscript
and its privately retained frozen outputs.

## Non-paid workflow entry points

From the repository root, after placing lawful local inputs in the canonical
paths, the main stages are:

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
python scripts/train_l1_classifiers.py --max-length <audited-token-limit>
```

The generation client in `src/generate.py` is not invoked by these review commands.
Use them only when regeneration is explicitly intended and the provider terms,
cost, and reasoning-disable settings have been verified.

When regeneration is explicitly authorized, the entry point is
`python -m src.generate --help`; provider credentials are read only from local
environment variables. No generation job is submitted by installing this
package or by running its non-paid checks.

## Release boundary

This directory is a local staging copy. No corpus, result, manuscript, or
credential is included, and no GitHub write is performed by preparing it.
