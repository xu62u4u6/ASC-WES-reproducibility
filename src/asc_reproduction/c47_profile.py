"""Reproduce the C47 gene-level A/T/S mean CNV profile (Figure 3B)."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .dataio import read_pivot_sqlite
from .figure_style import plt, save_tiff_and_preview
from .paired_cnv import COLORS

COMPONENTS = ("A", "T", "S")
COMPONENT_COLORS = dict(zip(COMPONENTS, COLORS))
EXPECTED_COMPONENT_COUNTS = {"A": 11, "T": 11, "S": 11}
DRIVER_GENES = ("PIK3CA", "SOX2", "TP63")
GAIN_THRESHOLD = 0.10


@dataclass(frozen=True)
class C47ProfileResult:
    """Group means and feature metadata, with sample identifiers omitted."""

    means: pd.DataFrame
    feature_metadata: pd.DataFrame
    sample_counts: dict[str, int]


from .cohort_inputs import as_sample_mask as _as_sample_mask


def calculate_c47_component_profile(
    gene_cnv: pd.DataFrame,
    gene_metadata: pd.DataFrame,
    sample_metadata: pd.DataFrame,
    cluster_map: pd.DataFrame,
    *,
    cluster: str = "47",
    expected_component_counts: dict[str, int] | None = EXPECTED_COMPONENT_COUNTS,
) -> C47ProfileResult:
    """Calculate gene-wise arithmetic means over the matched ASC A/T/S samples.

    Features retain the order in the processed gene-CNV matrix, matching the
    legacy profile plot. Only genes mapped to the frozen requested cluster are
    retained. Samples are selected by ``AS_sample`` and ``sample_type``; sample
    keys are used only internally for alignment and never returned.
    """
    if gene_cnv.index.has_duplicates or gene_cnv.columns.has_duplicates:
        raise ValueError("Gene-CNV matrix contains duplicate feature or sample keys")
    if not {"Cytoband"}.issubset(gene_metadata.columns):
        raise ValueError("Gene feature metadata must contain Cytoband")
    if not {"sample_type", "AS_sample"}.issubset(sample_metadata.columns):
        raise ValueError("Sample metadata must contain sample_type and AS_sample")
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

    cluster_id = str(cluster).removeprefix("C")
    features = gene_cnv.index[labels.eq(cluster_id).to_numpy()]
    if len(features) == 0:
        raise ValueError(f"Cluster {cluster_id} has no mapped genes")

    metadata = sample_metadata.loc[gene_cnv.columns]
    selected = metadata.loc[
        _as_sample_mask(metadata["AS_sample"])
        & metadata["sample_type"].isin(COMPONENTS)
    ]
    sample_groups = {
        component: selected.index[selected["sample_type"].eq(component)].tolist()
        for component in COMPONENTS
    }
    counts = {component: len(sample_groups[component]) for component in COMPONENTS}
    if expected_component_counts is not None and counts != expected_component_counts:
        raise ValueError(
            f"Expected A/T/S sample counts {expected_component_counts}; observed {counts}"
        )
    if any(not sample_groups[component] for component in COMPONENTS):
        raise ValueError("All A/T/S component groups must contain samples")

    values = gene_cnv.loc[features, :].astype(float)
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("C47 gene-CNV matrix contains non-finite values")
    means = pd.DataFrame(
        {
            component: values.loc[:, sample_groups[component]].mean(axis=1)
            for component in COMPONENTS
        },
        index=pd.Index(features.astype(str), name="feature"),
    )
    aligned_metadata = gene_metadata.copy()
    aligned_metadata.index = aligned_metadata.index.astype(str)
    aligned_metadata = aligned_metadata.loc[means.index, ["Cytoband"]].copy()
    if means.index.name != "feature":
        means.index.name = "feature"
    if not set(DRIVER_GENES).issubset(means.index):
        absent = sorted(set(DRIVER_GENES) - set(means.index))
        raise ValueError(f"Expected C47 driver genes are missing: {', '.join(absent)}")
    return C47ProfileResult(
        means=means,
        feature_metadata=aligned_metadata,
        sample_counts=counts,
    )


def _cytoband_major(value: object) -> int | None:
    text = str(value)
    if not text.startswith("3q"):
        return None
    major = text[2:].split(".", 1)[0]
    return int(major) if major.isdigit() else None


def make_c47_component_figure(result: C47ProfileResult, *, ax=None) -> plt.Figure:
    """Draw the gene-level C47 A/T/S means with the three 3q driver markers."""
    return make_c47_profile_figure(
        result.means,
        result.feature_metadata,
        group_colors=COMPONENT_COLORS,
        title="CNV Profiles on C47 Region across Components",
        ax=ax,
    )


def make_c47_profile_figure(
    means: pd.DataFrame,
    feature_metadata: pd.DataFrame,
    *,
    group_colors: dict[str, str],
    title: str,
    driver_genes: tuple[str, ...] = DRIVER_GENES,
    ax=None,
) -> plt.Figure:
    """Draw a C47 per-gene mean profile with shared 3q/threshold annotations."""
    features = means.index.tolist()
    x_positions = np.arange(len(features))
    if ax is None:
        fig, ax = plt.subplots(figsize=(18, 4.5))
    else:
        fig = ax.figure

    major_bands = [
        _cytoband_major(value) for value in feature_metadata.loc[features, "Cytoband"]
    ]
    region_positions = [
        index
        for index, major in enumerate(major_bands)
        if major is not None and 26 <= major <= 28
    ]
    if region_positions:
        ax.axvspan(
            min(region_positions) - 0.5,
            max(region_positions) + 0.5,
            facecolor="#FADADD",
            edgecolor="#D55E5E",
            alpha=0.28,
            linewidth=1.2,
            zorder=0,
            label="3q26–28 region",
        )

    for component in means.columns:
        ax.plot(
            x_positions,
            means[component].to_numpy(dtype=float),
            color=group_colors[component],
            linewidth=2.2,
            label=f"{component} (mean)",
        )
    ax.axhline(
        0, color="gray", linestyle="--", linewidth=1.2, label="Neutral (Log2 Ratio = 0)"
    )
    ax.axhline(
        GAIN_THRESHOLD,
        color="red",
        linestyle="--",
        linewidth=1.0,
        label=f"Gain ({GAIN_THRESHOLD:.2f})",
    )
    ax.axhline(
        -GAIN_THRESHOLD,
        color="blue",
        linestyle="--",
        linewidth=1.0,
        label=f"Loss ({-GAIN_THRESHOLD:.2f})",
    )

    for gene in driver_genes:
        position = features.index(gene)
        ax.axvline(
            position, color="gray", linestyle="--", linewidth=1.0, alpha=0.7, zorder=1
        )
        ax.text(
            position,
            0.98,
            gene,
            transform=ax.get_xaxis_transform(),
            rotation=90,
            va="top",
            ha="right",
            fontsize=12,
        )

    all_values = means.to_numpy(dtype=float)
    ax.set_title(title, fontsize=22)
    ax.set_xlabel("Genes (sorted by chromosomal position)", fontsize=22)
    ax.set_ylabel("Log2 Ratio", fontsize=22)
    ax.set_xticks([])
    ax.tick_params(axis="both", labelsize=18)
    ax.tick_params(axis="x", bottom=False)
    ax.set_ylim(float(np.min(all_values)) - 0.5, float(np.max(all_values)) + 0.5)
    ax.legend(fontsize=14)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
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
    result = calculate_c47_component_profile(
        genes.values,
        genes.feature_metadata,
        cohort.sample_metadata,
        cluster_map,
    )

    profile = result.means.join(result.feature_metadata)
    profile.to_csv(
        output_dir / "C47_gene_level_component_means.csv", index_label="gene"
    )
    figure = make_c47_component_figure(result)
    figure_manifest = save_tiff_and_preview(
        figure, output_dir / "Figure_3B_C47_gene_profile_TP63"
    )
    plt.close(figure)

    cytobands = result.feature_metadata["Cytoband"].astype(str)
    region_count = sum(
        (major is not None and 26 <= major <= 28)
        for major in (_cytoband_major(value) for value in cytobands)
    )
    marker_positions = {
        gene: int(result.means.index.get_loc(gene)) for gene in DRIVER_GENES
    }
    summary = {
        "status": "complete",
        "analysis": "C47 gene-level CNV component means",
        "figure_panel": "Figure 3B",
        "sample_selection": "AS_sample=true, grouped by sample_type A/T/S",
        "sample_counts": result.sample_counts,
        "cluster": "47",
        "gene_count": len(result.means),
        "highlighted_3q26_28_gene_count": region_count,
        "driver_gene_positions_zero_based": marker_positions,
        "gain_loss_threshold_log2_ratio": GAIN_THRESHOLD,
        "mean_calculation": "arithmetic mean of gene-level log2 ratios within each component",
        "feature_order": "processed gene-CNV matrix order (chromosomal order)",
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
