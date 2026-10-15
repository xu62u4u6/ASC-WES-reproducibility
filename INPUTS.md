# Inputs

No participant data or model binaries are included. This repository does not
provide a public study-data download. Data-access instructions must be supplied
by the authors before claiming third-party reproduction of the study results.

`input_spec.json` describes file paths relative to the supplied data root,
SQLite table names and mandatory columns. Required core inputs include:

- Cohort/sample metadata and matched A/T/S case membership.
- Processed mutation-presence, gene-CNV, arm-CNV and signature tables.
- The frozen 114-cluster gene mapping.

Figure-dependent inputs include MAF files, PyClone output, clustering metrics,
gene annotations and protein domains. These are not all produced by the core
preparation command; inspect the selected figure step's `--help`. External
annotation licenses and upstream provenance must be checked separately.

Prepared products are `data/prepared/cohort.h5` plus tumor and paired
PivotTables for SNV, thresholded CNV, gene CNV, arm CNV, cluster CNV and
signatures. Metadata must align by sample ID; complete paired cases must have
one A, one T and one S sample. Do not substitute independent column sorting.

Model-derived figures require trusted frozen model artifacts, matching input
identifiers and saved predictions/CV/RFECV/importance outputs. Only load model
files from a trusted source. They are not distributed in this code package.

Existing source specifications contain checks against the study dimensions.
Synthetic tests demonstrate individual functions; arbitrary cohorts are not
guaranteed to satisfy manuscript-specific assertions.
