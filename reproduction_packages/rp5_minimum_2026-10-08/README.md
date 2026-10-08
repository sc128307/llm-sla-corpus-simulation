# Minimum experiment reproduction materials

Minimum materials prepared for repository delivery, 2026-10-08. No manuscript, supplement, figures, raw
essays, essay-level exports, paid-generation clients, or model weights are
included. This package supersedes the earlier proposed reader/PDF package.
This package does not certify exact end-to-end reproduction.

## Contents and boundaries

`src/` and `scripts/` contain the original feature, statistical, regional
classification, marker-replacement and exploratory bridge code. `configs/`
records provider IDs and unexpanded prompt templates, without exemplars.
`reference_results/` contains numerical reference outputs, not experiment
inputs or a substitute for held-out predictions. [inventory.json](inventory.json)
records package hashes; [inputs_contract.json](inputs_contract.json) describes
excluded inputs, expected hashes, parameters and environment evidence.
[RIGHTS_AND_ATTRIBUTION.md](RIGHTS_AND_ATTRIBUTION.md) sets out resource access
and attribution. The MIT notice covers original software; it does not grant
rights to ICNALE or other separately acquired resources.

## Package and aggregate checks

Work in an extracted copy of this package, never the author's frozen project.
Python 3.10 was used for local QA. From the package root:

```text
python verify_package.py
python -m pip install -r requirements-analysis.txt
python verify_package.py --aggregate-check
```

The first command checks exact membership, hashes, JSON, CSV schema/counts
and Python syntax using only the standard library. The second installs a
functional analysis dependency set, not a historical environment lock.
The third runs the original marker-comparison and 24-cell bridge-input join
scripts in a temporary copy, comparing their numerical outputs to references.
It does not rerun bootstrap/permutation inference, feature extraction,
generation or training. Configure `PYTHONPATH` to the package root for the
pipeline commands below (PowerShell: `$env:PYTHONPATH=(Get-Location).Path`;
POSIX shell: `export PYTHONPATH="$PWD"`).

## Conditional experiment reproduction

Obtain the licensed canonical inputs separately. Matching the public raw
ICNALE archive alone is insufficient: the canonical baseline includes prior
cleaning, and the fixed writer split and existing synthetic texts must match
the input contract. Regenerating texts would create a different experiment.
No credentials or provider calls are necessary for the supplied workflow.
Use a fresh working copy and create `results/` and `features/` before running
the commands. Some original scripts replace their derived dataset directories
or fixed output names. Keep `reference_results/` immutable for comparison.

1. To rebuild features, first obtain the matching CEFR lexicon, fine-tuned
   RoBERTa-CoLA resource, spaCy model, NeoSCA/Stanza parser resources, human
   corpus and all 24 generated corpora. Install `requirements-features.txt`;
   prepare a separate NeoSCA environment using `requirements-neosca.txt`.
   Run `python verify_package.py --inputs-stage features` in each working
   copy before extraction. Confirm `engine: roberta-cola` in the log. Stop
   if another grammar engine is selected; it is not the recorded measurement.

   ```text
   python scripts/extract_features_v2.py A
   python scripts/extract_features_v2.py B
   python scripts/merge_features.py
   python scripts/extract_neosca_pandas.py A
   python scripts/extract_neosca_pandas.py B
   python scripts/extract_neosca_pandas.py C
   python scripts/merge_neosca.py
   ```

   Run the three NeoSCA commands in the separately configured environment.
   The two initial feature shards contain 72,800/67,200 rows; the merged
   result contains 140,000 rows. NeoSCA coverage must be 140,000 for both
   MLT and Clause_per_Sentence, with no duplicate joins or missing values.
   Full engine/environment identity is not established by syntax checks.

2. Alternatively start intrinsic analyses from the separately obtained frozen
   `features/features_all_with_neosca.parquet`. Validate it with
   `python verify_package.py --inputs-stage intrinsic`, then run
   `python scripts/phase4_statistics.py`. The Primary references comprise
   528 d_z rows and 528 pooled/5,280 regional JSD rows. ENS output has a
   separate sensitivity filename and must not enter Primary estimates.
   Pooled JSD uses 20 bins from the full scoped feature range. Historical
   bootstrap p/q fields are retained for provenance and are not certified
   writer-aware null inference.

3. For regional classifiers obtain the exact split manifest, human baseline,
   generated corpora and local DeBERTa-v3-base resource. Run
   `python verify_package.py --inputs-stage classifiers`, followed by:

   ```text
   python scripts/build_l1_dataset.py
   python scripts/build_l1_anonymized.py
   python scripts/train_l1_classifiers.py --dataset-dir results/l1_dataset --output-dir results/l1_models_primary --summary-path results/l1_classification_summary_primary.csv --max-length 512 --seed 42
   python scripts/train_l1_classifiers.py --dataset-dir results/l1_dataset_anonymized --output-dir results/l1_models_primary_anonymized --summary-path results/l1_classification_summary_primary_anonymized.csv --max-length 512 --seed 42
   python scripts/compare_primary_anonymized.py
   ```

   Build scripts also create separate ENS sensitivity datasets; do not train
   or combine them with the 37 Primary classifiers. Primary uses 10 L2 classes,
   85/15 writer-group train/dev and a fixed 1,040-essay/520-writer human test.
   Training uses essay-level overflow-chunk aggregation, not truncation.
   GPU/dependency differences can change trained results. There remain seven
   accuracy and 17 macro-F1 discrepancies between detailed inference and
   frozen summaries, and no reusable keyed test predictions. These references
   do not establish exact classifier reproduction or writer-cluster CIs.

4. For intrinsic writer sensitivity, obtain the original oracle dataset
   metadata and historical bootstrap reference specified in the input
   contract. Validate with `python verify_package.py --inputs-stage writer`,
   then run `python scripts/run_rp2_writer_cluster_bootstrap.py`. It writes
   528 rows using 10,000 replicates, 20 strata and 1,534 writer clusters.
   These sensitivity intervals do not validate the historical null p/q fields.

5. For the exploratory bridge, put matching `effect_sizes.csv` and
   `l1_classification_summary_primary.csv` in the working `results/` directory,
   then run `python scripts/audit_bridge_inputs.py` and, only after that audit
   passes, `python scripts/run_bridge_analysis.py`. The 24-cell model uses
   mean absolute d_z across 22 features, model/condition fixed effects,
   10,000 within-model permutations and 5,000 model-block bootstraps.
   Its interpretation is exploratory and non-causal. The default aggregate
   check validates the join only, not this full inferential rerun.

## Validation limits

Local QA checks package integrity and two deterministic aggregate routes.
Excluded inputs, missing original feature-run logs, incomplete historical
environment records and unreconciled classifier predictions prevent an
unconditional exact-reproduction claim. This release does not resolve those
issues or certify submission readiness. Use the fixed repository commit to
identify this delivery version and run the package verifier on its contents.
