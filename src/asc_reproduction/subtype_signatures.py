"""S14 MSI, Ti/Tv and supplied SBS exposures across T-sample subtypes."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .dataio import read_pivot_sqlite
from .figure_style import plt
from .subtype_metrics import (
    calculate_subtype_metric_statistics,
    make_subtype_metric_figure,
    save_fixed_canvas_figure,
)

SIGNATURES = ("SBS1", "SBS4", "SBS5", "SBS6")
METRICS = ("MSI", "ti/tv", *SIGNATURES)


def prepare_signature_metadata(snv_metadata, values, signature_metadata):
    """Align supplied exposures to SNV sample IDs without refitting."""
    if (
        snv_metadata.index.has_duplicates
        or values.index.has_duplicates
        or values.columns.has_duplicates
        or signature_metadata.index.has_duplicates
    ):
        raise ValueError("Duplicate signature or sample keys")
    if set(values.columns) != set(snv_metadata.index) or set(values.columns) != set(
        signature_metadata.index
    ):
        raise ValueError("Signature and metadata sample keys differ")
    for column in ("sample_type", "subtype"):
        # Supplied signature DB has sample keys but no metadata columns. Legacy
        # Cohort.add_table propagates SNV annotations; verify any supplied overlap.
        if column in signature_metadata and not snv_metadata[column].equals(
            signature_metadata.loc[snv_metadata.index, column]
        ):
            raise ValueError("Signature cohort annotations differ")
    if not set(SIGNATURES).issubset(values.index):
        raise ValueError("Required SBS exposure missing")
    x = values.to_numpy(dtype=float)
    if not np.isfinite(x).all() or (x < 0).any() or (x > 1).any():
        raise ValueError("Invalid supplied fractional exposures")
    if not np.allclose(x.sum(axis=0), 1, rtol=0, atol=1e-12):
        raise ValueError("Supplied signature fractions do not sum to one")
    if set(SIGNATURES) & set(snv_metadata.columns):
        raise ValueError("Ambiguous duplicate exposure columns")
    # Do not refit or normalize the supplied exposures; align explicitly by sample.
    return pd.concat(
        [snv_metadata, values.loc[list(SIGNATURES), snv_metadata.index].T], axis=1
    )


def main():
    """Export six package-rendered panels and all 18 raw pairwise tests."""
    from .figure_style import setup_plotting

    setup_plotting()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Output must be new or empty")
    snv = read_pivot_sqlite(args.data_root / "data/WES/all_case_table.db")
    signatures = read_pivot_sqlite(args.data_root / "data/signature/signature_table.db")
    metadata = prepare_signature_metadata(
        snv.sample_metadata, signatures.values, signatures.sample_metadata
    )
    results = calculate_subtype_metric_statistics(metadata, metrics=METRICS)
    args.output.mkdir(parents=True, exist_ok=True)
    results.group_summary.to_csv(args.output / "S14_group_summary.csv", index=False)
    results.pairwise_statistics.to_csv(
        args.output / "S14_pairwise_raw.csv", index=False
    )
    figures = []
    for i, metric in enumerate(METRICS):
        fig = make_subtype_metric_figure(
            metadata, metric=metric, alpha=0.6 if metric in SIGNATURES else 0.8
        )
        # The source notebook uses whitegrid; retain horizontal reading guides.
        fig.axes[0].set_axisbelow(True)
        fig.axes[0].yaxis.grid(True, color="0.85", linewidth=0.8)
        figures.append(save_fixed_canvas_figure(fig, args.output / f"S14{chr(65 + i)}"))
        plt.close(fig)
    summary = {
        "panel": "S14",
        "renderer": "pymaftools 0.5.0 plot_boxplot_with_annot",
        "metrics": list(METRICS),
        "figures": figures,
        "counts": {"LUAD": 89, "ASC": 29, "LUSC": 82},
        "selection": "all T samples",
        "signature_input": "Supplied 10-signature fractions; no refitting or renormalization",
        "test": "Pairwise two-sided Mann-Whitney U; uncorrected legacy annotations",
        "limitations": "Signature fitting tool/version and upstream metric provenance pending. Manuscript BH wording unresolved.",
        "patient_or_sample_identifiers_written": False,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
