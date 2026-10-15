"""Reproduce the T-sample subtype C47 CNV profile (Figure 4D)."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .c47_profile import _cytoband_major, make_c47_profile_figure
from .dataio import read_pivot_sqlite
from .figure_style import plt, save_tiff_and_preview

SUBTYPES = ("ASC", "LUAD", "LUSC")
from .figure_style import default_color_manager

SUBTYPE_COLORS = {
    name: default_color_manager().get_cmap("subtype")[name]
    for name in ("ASC", "LUAD", "LUSC")
}
EXPECTED_SUBTYPE_COUNTS = {"LUAD": 89, "ASC": 29, "LUSC": 82}
DRIVER_GENES = ("PIK3CA", "SOX2", "TP63")


@dataclass(frozen=True)
class SubtypeC47ProfileResult:
    means: pd.DataFrame
    feature_metadata: pd.DataFrame
    sample_counts: dict[str, int]


def calculate_subtype_c47_profile(
    gene_cnv: pd.DataFrame,
    gene_metadata: pd.DataFrame,
    sample_metadata: pd.DataFrame,
    cluster_map: pd.DataFrame,
    *,
    expected_subtype_counts: dict[str, int] | None = EXPECTED_SUBTYPE_COUNTS,
) -> SubtypeC47ProfileResult:
    """Compute C47 gene means across reference and ASC tumor (T) samples.

    Unlike the paired-component panel, this subtype comparison uses all T
    samples (LUAD, ASC, LUSC), not only the matched ASC A/T/S subset.
    """
    if gene_cnv.index.has_duplicates or gene_cnv.columns.has_duplicates:
        raise ValueError("Gene-CNV matrix contains duplicate feature or sample keys")
    if "Cytoband" not in gene_metadata.columns:
        raise ValueError("Gene feature metadata must contain Cytoband")
    required_metadata = {"sample_type", "subtype"}
    if not required_metadata.issubset(sample_metadata.columns):
        raise ValueError("Sample metadata must contain sample_type and subtype")
    if not {"feature", "cluster"}.issubset(cluster_map.columns):
        raise ValueError("Cluster map must contain feature and cluster")
    if not set(gene_cnv.columns).issubset(sample_metadata.index):
        raise ValueError("Sample metadata does not cover the gene-CNV matrix")
    if not set(gene_cnv.index.astype(str)).issubset(gene_metadata.index.astype(str)):
        raise ValueError("Gene feature metadata does not cover the gene-CNV matrix")

    mapping = cluster_map.loc[:, ["feature", "cluster"]].copy()
    mapping["feature"] = mapping["feature"].astype(str)
    mapping["cluster"] = mapping["cluster"].astype(str).str.removeprefix("C")
    if mapping["feature"].duplicated().any():
        raise ValueError("Cluster map contains duplicate gene features")
    labels = mapping.set_index("feature")["cluster"].reindex(gene_cnv.index.astype(str))
    if labels.isna().any():
        raise ValueError("Cluster map does not cover the gene-CNV matrix")
    features = gene_cnv.index[labels.eq("47").to_numpy()]
    if len(features) == 0:
        raise ValueError("Frozen cluster map does not contain C47 genes")

    metadata = sample_metadata.loc[gene_cnv.columns]
    metadata = metadata.loc[
        metadata["sample_type"].eq("T") & metadata["subtype"].isin(SUBTYPES)
    ]
    sample_groups = {
        subtype: metadata.index[metadata["subtype"].eq(subtype)].tolist()
        for subtype in SUBTYPES
    }
    counts = {subtype: len(sample_groups[subtype]) for subtype in SUBTYPES}
    if expected_subtype_counts is not None and counts != expected_subtype_counts:
        raise ValueError(
            f"Expected T-sample counts {expected_subtype_counts}; observed {counts}"
        )
    if any(not sample_groups[subtype] for subtype in SUBTYPES):
        raise ValueError("Each subtype must contain at least one T sample")

    values = gene_cnv.loc[features, :].astype(float)
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("C47 gene-CNV matrix contains non-finite values")
    means = pd.DataFrame(
        {
            subtype: values.loc[:, sample_groups[subtype]].mean(axis=1)
            for subtype in SUBTYPES
        },
        index=pd.Index(features.astype(str), name="feature"),
    )
    aligned_metadata = gene_metadata.copy()
    aligned_metadata.index = aligned_metadata.index.astype(str)
    aligned_metadata = aligned_metadata.loc[means.index, ["Cytoband"]]
    absent_drivers = sorted(set(DRIVER_GENES) - set(means.index))
    if absent_drivers:
        raise ValueError(
            f"Expected C47 driver genes are missing: {', '.join(absent_drivers)}"
        )
    return SubtypeC47ProfileResult(means, aligned_metadata, counts)


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
    result = calculate_subtype_c47_profile(
        genes.values,
        genes.feature_metadata,
        cohort.sample_metadata,
        cluster_map,
    )

    result.means.join(result.feature_metadata).to_csv(
        output_dir / "C47_gene_level_subtype_means.csv", index_label="gene"
    )
    figure = make_c47_profile_figure(
        result.means,
        result.feature_metadata,
        group_colors=SUBTYPE_COLORS,
        title="CNV Profiles on C47 Region across Subtypes",
        driver_genes=DRIVER_GENES,
    )
    # Figure 4D spans the full composite width; balance its type with A–C.
    axis = figure.axes[0]
    axis.title.set_fontsize(16)
    axis.xaxis.label.set_fontsize(16)
    axis.yaxis.label.set_fontsize(16)
    axis.tick_params(labelsize=12)
    # Rebuild the legend: shrinking text alone retains the old tall layout
    # and obscures the LUSC curve. Two compact columns fit above the data.
    axis.legend(
        loc="upper left",
        ncols=2,
        fontsize=12,
        labelspacing=0.25,
        borderpad=0.4,
        columnspacing=1.0,
        handlelength=2.0,
    )
    figure_manifest = save_tiff_and_preview(
        figure, output_dir / "Figure_4D_C47_subtype_profile_TP63"
    )
    plt.close(figure)
    region_count = sum(
        major is not None and 26 <= major <= 28
        for major in (
            _cytoband_major(value) for value in result.feature_metadata["Cytoband"]
        )
    )
    summary = {
        "status": "complete",
        "analysis": "T-sample subtype C47 gene-level CNV means",
        "figure_panel": "Figure 4D",
        "sample_selection": "all sample_type=T samples, grouped by subtype; not limited to matched AS_sample cases",
        "subtype_counts": result.sample_counts,
        "cluster": "47",
        "gene_count": len(result.means),
        "highlighted_3q26_28_gene_count": int(region_count),
        "driver_genes": list(DRIVER_GENES),
        "mean_calculation": "arithmetic mean of gene-level CNV log2 ratios within each subtype",
        "figure_manifest": figure_manifest,
        "patient_or_sample_identifiers_written": False,
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
