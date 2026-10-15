"""C47/C6 distributions and paired differences: S10A and S15D–G."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon

from .c47_profile import _as_sample_mask
from .dataio import read_pivot_sqlite
from .figure_style import plt, save_tiff_and_preview
from .paired_screen import cluster_mean_matrix, matched_part_matrices
from .subtype_metrics import _significance_label

PARTS = ("A", "T", "S")
SUBTYPES = ("LUAD", "ASC", "LUSC")
PANEL_NAMES = {
    "C47_component": "S10A_C47_component",
    "C47_subtype": "S15D_C47_subtype",
    "C6_subtype": "S15E_C6_subtype",
    "C6_component": "S15F_C6_component",
}


@dataclass(frozen=True)
class ClusterPanels:
    """Validated cluster values, statistics and keyed plotting metadata."""
    groups: dict[str, dict[str, np.ndarray]]
    statistics: pd.DataFrame
    group_summary: pd.DataFrame
    gene_counts: dict[str, int]
    plot_metadata: dict[str, pd.DataFrame]


def calculate_cluster_panels(
    genes: pd.DataFrame,
    metadata: pd.DataFrame,
    mapping: pd.DataFrame,
    *,
    expected_cases: int = 11,
    expected_clusters: int = 114,
    expected_subtypes: dict | None = None,
) -> ClusterPanels:
    """Calculate original cluster means and tests without changing selection."""
    means = cluster_mean_matrix(
        genes, mapping, expected_cluster_count=expected_clusters
    )
    if not {"47", "6"}.issubset(means.index):
        raise ValueError("Both C47 and C6 are required")
    if set(genes.columns) != set(metadata.index):
        raise ValueError("CNV sample keys and metadata differ")
    selected = metadata.loc[_as_sample_mask(metadata["AS_sample"])]
    arrays = matched_part_matrices(
        means.loc[["47", "6"]], selected, expected_case_count=expected_cases
    )
    tumor = metadata.loc[metadata["sample_type"].eq("T")]
    counts = tumor["subtype"].value_counts().to_dict()
    if expected_subtypes is not None and counts != expected_subtypes:
        raise ValueError(f"Unexpected subtype counts: {counts}")
    groups, rows, summaries = {}, [], []
    plot_metadata = {}
    for i, key in enumerate(("47", "6")):
        for kind in ("component", "subtype"):
            panel = f"C{key}_{kind}"
            plot_data = (selected if kind == "component" else tumor).copy()
            plot_data = plot_data.sort_values(["case_ID", "sample_type"])
            plot_data["cluster_ratio"] = means.loc[key].reindex(plot_data.index)
            plot_metadata[panel] = plot_data
            if kind == "component":
                groups[panel] = {part: arrays[part][i] for part in PARTS}
            else:
                groups[panel] = {
                    group: means.loc[
                        key, tumor.index[tumor["subtype"].eq(group)]
                    ].to_numpy(dtype=float)
                    for group in SUBTYPES
                }
            labels = list(groups[panel])
            for label, values in groups[panel].items():
                if len(values) == 0 or not np.isfinite(values).all():
                    raise ValueError("Empty group or non-finite cluster values")
                summaries.append(
                    {
                        "panel": panel,
                        "group": label,
                        "n": len(values),
                        "median": float(np.median(values)),
                        "minimum": float(values.min()),
                        "maximum": float(values.max()),
                    }
                )
            for a, b in ((0, 1), (1, 2), (0, 2)):
                first, second = labels[a], labels[b]
                x, y = groups[panel][first], groups[panel][second]
                if kind == "component":
                    p = (
                        1.0
                        if np.array_equal(x, y)
                        else float(wilcoxon(x, y, alternative="two-sided").pvalue)
                    )
                else:
                    p = float(mannwhitneyu(x, y, alternative="two-sided").pvalue)
                rows.append(
                    {
                        "panel": panel,
                        "comparison": f"{first} vs {second}",
                        "p_raw_two_sided": p,
                        "test": "paired Wilcoxon"
                        if kind == "component"
                        else "Mann-Whitney U",
                        "adjustment": "none; legacy candidate, manuscript BH wording pending",
                        "annotation": _significance_label(p),
                    }
                )
    gene_counts = {
        f"C{key}": int(mapping["cluster"].astype(str).eq(key).sum())
        for key in ("47", "6")
    }
    return ClusterPanels(
        groups, pd.DataFrame(rows), pd.DataFrame(summaries), gene_counts, plot_metadata
    )


def make_cluster_boxplot(result: ClusterPanels, panel: str, *, ax=None) -> plt.Figure:
    """Use the original package API with real sample and paired case metadata."""
    from pymaftools import PivotTable

    from .figure_style import PART_COLORS, SUBTYPE_COLORS
    from .oncoplot_rendering import require_pymaftools_version

    require_pymaftools_version()
    data = result.plot_metadata[panel]
    component = panel.endswith("component")
    labels = list(PARTS if component else SUBTYPES)
    group_col = "sample_type" if component else "subtype"
    assert not data.index.has_duplicates
    assert data["cluster_ratio"].notna().all()
    if component:
        assert not data.duplicated(["case_ID", "sample_type"]).any()
        orders = [data.loc[data.sample_type.eq(p), "case_ID"].tolist() for p in PARTS]
        assert orders[0] == orders[1] == orders[2]
        for part in PARTS:
            np.testing.assert_allclose(
                data.loc[data.sample_type.eq(part), "cluster_ratio"],
                result.groups[panel][part],
            )
    table = PivotTable(pd.DataFrame(index=["cluster_ratio"], columns=data.index))
    table.sample_metadata = data
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure
    table.plot.plot_boxplot_with_annot(
        group_col=group_col,
        test_col="cluster_ratio",
        test="Wilcoxon" if component else "Mann-Whitney",
        is_paired=component,
        pair_col="case_ID",
        comparisons_correction=None,
        order=labels,
        palette=PART_COLORS if component else SUBTYPE_COLORS,
        ax=ax,
        fontsize=12,
        alpha=0.6,
        ylabel="CNV log2 ratio",
    )
    cluster, kind = panel.split("_")
    ax.set_title(f"{cluster} ({kind.title()})")
    return fig


def make_c6_difference(result: ClusterPanels) -> plt.Figure:
    """Render C6 paired differences with checked original case identifiers."""
    group = result.groups["C6_component"]
    differences = group["A"] - group["S"]
    x = np.arange(1, len(differences) + 1)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(x, differences, color=np.where(differences < 0, "steelblue", "indianred"))
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    metadata = result.plot_metadata["C6_component"]
    cases = metadata.loc[metadata.sample_type.eq("A"), "case_ID"].tolist()
    assert len(cases) == len(differences) == len(set(cases))
    ax.set_xticks(x, cases, rotation=90)
    ax.set(
        title="C6: case-wise A − S",
        ylabel="A − S CNV log2 ratio",
        xlabel="case ID",
    )
    return fig


def main() -> None:
    """Produce cluster panels without assembling or replacing manuscript figures."""
    from .figure_style import setup_plotting

    setup_plotting()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Output directory must be new or empty")
    genes = read_pivot_sqlite(args.data_root / "data/CNV_0407/gene_cnv.db")
    meta = read_pivot_sqlite(
        args.data_root / "data/WES/all_case_table.db"
    ).sample_metadata
    mapping = pd.read_csv(args.data_root / "data/clustering/best_k_cluster_labels.csv")
    result = calculate_cluster_panels(
        genes.values,
        meta,
        mapping,
        expected_subtypes={"LUAD": 89, "ASC": 29, "LUSC": 82},
    )
    args.output.mkdir(parents=True, exist_ok=True)
    manifests = []
    for panel, filename in PANEL_NAMES.items():
        fig = make_cluster_boxplot(result, panel)
        manifests.append(save_tiff_and_preview(fig, args.output / filename))
        plt.close(fig)
    fig = make_c6_difference(result)
    manifests.append(save_tiff_and_preview(fig, args.output / "S15G_C6_A_minus_S"))
    plt.close(fig)
    result.statistics.to_csv(args.output / "pairwise_tests.csv", index=False)
    result.group_summary.to_csv(args.output / "group_summary.csv", index=False)
    diff = result.groups["C6_component"]["A"] - result.groups["C6_component"]["S"]
    summary = {
        "status": "legacy annotations reproduced; correction rule pending",
        "gene_counts": result.gene_counts,
        "paired_cases": len(diff),
        "C6_A_minus_S_sign_counts": {
            "negative": int((diff < 0).sum()),
            "zero": int((diff == 0).sum()),
            "positive": int((diff > 0).sum()),
        },
        "figures": manifests,
        "per_sample_table_or_identifiers_written": False,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
