"""Reproduce the matched-case C47 A-minus-S bar panel (Figure 3C)."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .c47_profile import _as_sample_mask
from .dataio import read_pivot_sqlite
from .figure_style import plt, save_tiff_and_preview
from .paired_screen import cluster_mean_matrix, matched_part_matrices

EXPECTED_CASE_COUNT = 11


@dataclass(frozen=True)
class PairedC47DifferenceResult:
    """Per-case values in deterministic case order, with identifiers removed."""

    values: pd.DataFrame
    c47_gene_count: int


def calculate_paired_c47_difference(
    gene_cnv: pd.DataFrame,
    sample_metadata: pd.DataFrame,
    cluster_map: pd.DataFrame,
    *,
    expected_case_count: int = EXPECTED_CASE_COUNT,
    expected_cluster_count: int = 114,
) -> PairedC47DifferenceResult:
    """Calculate paired A−S from the per-sample frozen-cluster gene mean.

    A/T/S samples are restricted to ``AS_sample=true`` and matched by
    ``case_ID``. Cases are ordered by ID to match the prior figure's order, but
    only anonymous 1..n row labels are returned.
    """
    required = {"case_ID", "sample_type", "AS_sample"}
    absent = required - set(sample_metadata.columns)
    if absent:
        raise ValueError(f"Sample metadata is missing: {', '.join(sorted(absent))}")
    if not set(gene_cnv.columns).issubset(sample_metadata.index):
        raise ValueError("Sample metadata does not cover the gene-CNV matrix")

    cluster_map = cluster_map.loc[:, ["feature", "cluster"]].copy()
    if cluster_map["feature"].duplicated().any():
        raise ValueError("Cluster map contains duplicate gene features")
    cluster_means = cluster_mean_matrix(
        gene_cnv,
        cluster_map,
        expected_cluster_count=expected_cluster_count,
    )
    if "47" not in cluster_means.index.astype(str):
        raise ValueError("Frozen cluster map does not contain cluster 47")
    cluster_gene_count = int(
        cluster_map["cluster"].astype(str).str.removeprefix("C").eq("47").sum()
    )

    metadata = sample_metadata.loc[gene_cnv.columns]
    selected = metadata.loc[
        _as_sample_mask(metadata["AS_sample"])
        & metadata["sample_type"].isin(("A", "T", "S"))
    ]
    if selected["case_ID"].isna().any():
        raise ValueError("Selected matched-component metadata contains missing case_ID")
    cluster_samples = cluster_means.loc[["47"], selected.index]
    arrays = matched_part_matrices(
        cluster_samples,
        selected,
        expected_case_count=expected_case_count,
    )
    a_values = arrays["A"][0]
    s_values = arrays["S"][0]
    differences = a_values - s_values
    if not np.isfinite(differences).all():
        raise ValueError("Paired C47 A−S differences contain non-finite values")
    values = pd.DataFrame(
        {
            "A_mean": a_values,
            "S_mean": s_values,
            "A_minus_S": differences,
        },
        index=pd.RangeIndex(1, len(differences) + 1, name="anonymous_case_order"),
    )
    return PairedC47DifferenceResult(values=values, c47_gene_count=cluster_gene_count)


def make_paired_c47_difference_figure(
    result: PairedC47DifferenceResult,
    *,
    ax=None,
) -> plt.Figure:
    differences = result.values["A_minus_S"].to_numpy(dtype=float)
    x = np.arange(1, len(differences) + 1)
    colors = np.where(differences < 0, "steelblue", "indianred")
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.figure
    ax.bar(x, differences, color=colors)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xticks(x, [str(i) for i in x], rotation=90)
    ax.set_xlabel("Matched case (anonymized; sorted by case ID)")
    ax.set_ylabel("A − S CNV log2 ratio")
    ax.set_title("C47: case-wise A − S")
    return fig


def run(data_root: Path, output_dir: Path) -> dict[str, object]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty output directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    genes = read_pivot_sqlite(data_root / "data/CNV_0407/gene_cnv.db")
    cohort = read_pivot_sqlite(data_root / "data/WES/all_case_table.db")
    if set(genes.values.columns) != set(cohort.sample_metadata.index):
        raise ValueError("WES sample metadata does not cover the gene-CNV matrix")
    cluster_map = pd.read_csv(data_root / "data/clustering/best_k_cluster_labels.csv")
    result = calculate_paired_c47_difference(
        genes.values, cohort.sample_metadata, cluster_map
    )
    figure = make_paired_c47_difference_figure(result)
    figure_manifest = save_tiff_and_preview(
        figure, output_dir / "Figure_3C_C47_paired_A_minus_S"
    )
    plt.close(figure)

    differences = result.values["A_minus_S"]
    summary = {
        "status": "complete",
        "analysis": "matched-case C47 cluster-mean A-minus-S differences",
        "figure_panel": "Figure 3C",
        "sample_selection": "AS_sample=true; complete matched A/T/S cases",
        "matched_case_count": len(result.values),
        "c47_gene_count": result.c47_gene_count,
        "cluster_value": "arithmetic mean of gene-level CNV log2 ratios within C47 per sample",
        "contrast": "A component cluster mean minus S component cluster mean within case",
        "case_order": "sorted case_ID order; displayed using anonymous 1..n labels",
        "difference_sign_counts": {
            "negative": int((differences < 0).sum()),
            "zero": int((differences == 0).sum()),
            "nonnegative": int((differences >= 0).sum()),
        },
        "figure_manifest": figure_manifest,
        "patient_or_sample_identifiers_written": False,
        "patient_level_values_written_as_table": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    from .figure_style import setup_plotting

    setup_plotting()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="external data root containing the data/ directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new or empty local output directory (never overwritten)",
    )
    args = parser.parse_args()
    print(json.dumps(run(args.data_root.resolve(), args.output.resolve()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
