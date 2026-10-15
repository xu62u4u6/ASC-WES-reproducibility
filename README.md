# ASC WES reproducibility

Analysis and figure code for the ASC WES manuscript, starting from processed
mutation, CNV, and documented derived inputs. Python 3.12 and pymaftools 0.5.0
are pinned in `pyproject.toml` and `uv.lock`. This code-only repository
does not include participant data, fitted models, manuscript files, or results.

## Install and check

Run from this checkout on Linux. An editable installation is required because
figure routes also invoke scripts stored in the checkout.

```bash
uv sync --frozen --extra analysis --extra dev
uv run --frozen pytest -q
uv run --frozen python scripts/reproduce_figures.py --list
```

The tests use synthetic inputs. Passing them does not reproduce the study's
numbers. Research input availability and third-party resource permissions must
be resolved separately; see [INPUTS.md](INPUTS.md).

## Prepare inputs

```bash
uv run --frozen python check_inputs.py --data-root /path/to/processed-inputs
uv run --frozen python -m asc_reproduction.cohort_inputs \
  --source-root /path/to/processed-inputs --repo-root .
```

Preparation creates a shared Cohort and analysis-specific PivotTables under
`data/prepared/`. Frequency filtering and model feature selection occur later
in their respective analyses.

## Figures

`scripts/reproduce_figures.py --list` lists the registered analysis and assembly
steps for each figure. Inspect each step's arguments before running it:

```bash
uv run --frozen python scripts/reproduce_figures.py \
  --figure Supplementary_Figure_19 --step render -- --help
```

Provide a new `--output` directory for each step. Rendering and assembly use
saved results where required; they do not launch classifier training.
Figure 1, S2, S3, and S4 use externally supplied artwork via `retain`.
S19 requires saved CV summaries and RFECV curves; S20 requires saved fold
importance summaries. S16 group summaries require the additional
`scripts/compare_s16_permutations.py` step and its `050` outputs; run it from
this checkout after preparation. Its legacy comparison is retained for audit.
The registered steps are not a bundled one-command study rerun: a complete
input package and per-step arguments are still required.

## Models and tables

```bash
uv run --frozen python -m asc_reproduction.model_snapshot --help
uv run --frozen python -m asc_reproduction.model_runner --help
uv run --frozen python -m asc_reproduction.model_finalize --help
uv run --frozen python -m asc_reproduction.model_tables --help
```

The model uses a fixed stratified 80/20 reference-cohort split (seed 42), with
100 seeds × five-fold CV in development patients. TOP/RFE selection is fitted
inside outer and stacking internal training splits. RFECV uses internal
five-fold accuracy; configurations are ranked by mean AUROC before rounding.
The final selected model is fitted on development patients and evaluated on
holdout patients. Frozen CNV clusters were defined using the full reference
cohort, including later holdout participants.

Freeze source before a long run. This release has a different source identity
from historical snapshots: use original verified snapshots for existing runs.
Do not use this checkout to resume historical model checkpoints.

## Contents and limits

- `src/asc_reproduction/`: existing analysis and shared implementation.
- `scripts/`: figure routes, required renderers and assembly.
- `tests/`: synthetic checks of alignment, selection, summaries and routing.
- `input_spec.json`: input contract; `analysis_spec.json`: analysis settings.

FASTQ processing, variant calling, upstream CNV inference, signature extraction
and CNV cluster derivation are outside this code's scope. Legacy helper modules
remain where the existing runner requires them; use the figure-number entry
above for manuscript output. No explicit code license has been selected; public visibility does not grant
additional reuse rights. Third-party dependencies retain their own licenses.
