"""Regenerate S15A gene CNV and S6A/B paired arm/cluster heatmaps."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, Normalize, to_rgba
from matplotlib.patches import Patch
from pymaftools import OncoPlot, PivotTable

from .c47_profile import _as_sample_mask
from .cgc_oncoplot import group_mutation_order, mutation_presence
from .cohort_inputs import read_prepared_table, select_paired_metadata
from .dataio import read_pivot_sqlite
from .figure_style import (
    PART_COLORS,
    TRACK_COLORS,
    patient_palette,
    plt,
    save_tiff_and_preview,
)
from .oncoplot_rendering import (
    align_sample_label_gutter,
    require_pymaftools_version,
)
from .paired_screen import cluster_mean_matrix

PARTS = ("A", "T", "S")


@dataclass(frozen=True)
class HeatmapData:
    values: pd.DataFrame
    metadata: pd.DataFrame
    column_labels: list[str]
    label: str
    limit: float
    paired: bool


def prepare_heatmaps_from_prepared(data_root: Path) -> dict[str, HeatmapData]:
    """Use the shared Cohort-derived pivots for S15A and paired S6A/B."""
    gene = read_prepared_table(data_root, "tumor", "CNV_gene")
    arm = read_prepared_table(data_root, "paired", "CNV_arm")
    cluster = read_prepared_table(data_root, "paired", "CNV_cluster")
    if len(gene) != 25988 or len(arm) != 44 or len(cluster) != 114:
        raise ValueError("Prepared CNV feature counts differ from manuscript")
    if gene.sample_metadata["subtype"].value_counts().to_dict() != {
        "LUAD": 89,
        "ASC": 29,
        "LUSC": 82,
    }:
        raise ValueError("Prepared T-only subtype counts differ")
    paired_ids = arm.columns.tolist()
    if paired_ids != cluster.columns.tolist():
        raise ValueError("Paired arm and cluster sample orders differ")
    if any(not sample.endswith(("_A", "_T", "_S")) for sample in paired_ids):
        raise ValueError("Paired sample names must be original A/T/S IDs")
    return {
        "S15A_gene_CNV": HeatmapData(
            pd.DataFrame(gene),
            gene.sample_metadata,
            [],
            "Genes",
            2.0,
            False,
        ),
        "S6A_arm_CNV": HeatmapData(
            pd.DataFrame(arm),
            arm.sample_metadata,
            paired_ids,
            "Chromosome arms",
            1.0,
            True,
        ),
        "S6B_cluster_CNV": HeatmapData(
            pd.DataFrame(cluster),
            cluster.sample_metadata,
            paired_ids,
            "CNV clusters",
            1.0,
            True,
        ),
    }


def prepare_heatmaps(
    gene_values: pd.DataFrame,
    arm_values: pd.DataFrame,
    snv_values: pd.DataFrame,
    metadata: pd.DataFrame,
    cluster_map: pd.DataFrame,
    case_metadata: pd.DataFrame,
    *,
    expected_clusters: int = 114,
    expected_cases: int = 11,
    expected_subtypes: dict | None = None,
) -> dict[str, HeatmapData]:
    if metadata.index.has_duplicates or case_metadata["case_ID"].duplicated().any():
        raise ValueError("Duplicate sample or case metadata keys")
    for matrix in (gene_values, arm_values, snv_values):
        if matrix.index.has_duplicates or matrix.columns.has_duplicates:
            raise ValueError("Duplicate matrix feature or sample keys")
        if set(matrix.columns) != set(metadata.index):
            raise ValueError("CNV/SNV matrices and metadata sample keys must match")
    for matrix in (gene_values, arm_values):
        if not np.isfinite(matrix.to_numpy(dtype=float)).all():
            raise ValueError("Non-finite CNV values")
    # Legacy ASCDataLoader orders the entire cohort using the first ten SNV rows.
    ordered = group_mutation_order(
        mutation_presence(snv_values), metadata.loc[snv_values.columns]
    )
    tumors = metadata.loc[ordered]
    tumors = tumors.loc[tumors.sample_type.eq("T")]
    counts = tumors.subtype.value_counts().to_dict()
    if expected_subtypes is not None and counts != expected_subtypes:
        raise ValueError(f"Unexpected T-only subtype counts: {counts}")
    cases = case_metadata.loc[
        _as_sample_mask(case_metadata.AS_sample), "case_ID"
    ].tolist()
    if len(cases) != expected_cases or pd.isna(cases).any():
        raise ValueError("Unexpected complete-case cohort")
    paired = select_paired_metadata(
        metadata, case_metadata, expected_cases=expected_cases
    )
    # The manuscript heatmaps display each original case and component ID.
    labels = [f"{case}_{part}" for case in cases for part in PARTS]
    clusters = cluster_mean_matrix(
        gene_values, cluster_map, expected_cluster_count=expected_clusters
    )
    outputs = {
        "S15A_gene_CNV": HeatmapData(
            gene_values.loc[:, tumors.index], tumors, [], "Genes", 2.0, False
        ),
        "S6A_arm_CNV": HeatmapData(
            arm_values.loc[:, paired.index],
            paired,
            labels,
            "Chromosome arms",
            1.0,
            True,
        ),
        "S6B_cluster_CNV": HeatmapData(
            clusters.loc[:, paired.index], paired, labels, "CNV clusters", 1.0, True
        ),
    }
    for data in outputs.values():
        tracks = (
            {"sample_type": PART_COLORS, "smoke": TRACK_COLORS["smoke"]}
            if data.paired
            else TRACK_COLORS
        )
        for name, colors in tracks.items():
            if not data.metadata[name].isin(colors).all():
                raise ValueError(f"Unrecognized/missing {name} annotation")
    return outputs


def make_cnv_heatmap(data: HeatmapData) -> plt.Figure:
    tracks = (
        {"sample_type": PART_COLORS, "smoke": TRACK_COLORS["smoke"]}
        if data.paired
        else TRACK_COLORS
    )
    fig = plt.figure(figsize=(26, 12) if data.paired else (30, 10))
    grid = fig.add_gridspec(
        1 + len(tracks),
        3,
        width_ratios=[25, 0.6, 3],
        height_ratios=[20] + [1] * len(tracks),
        wspace=0.08,
        hspace=0.03,
    )
    ax = fig.add_subplot(grid[0, 0])
    # Numeric input remains unchanged. Dense gene rows are rasterized for export.
    heat = ax.imshow(
        data.values.to_numpy(dtype=float),
        cmap="coolwarm",
        aspect="auto",
        interpolation="nearest",
        vmin=-data.limit,
        vmax=data.limit,
    )
    ax.set(xticks=[], ylabel=data.label)
    ax.yaxis.label.set_size(17)
    if data.label == "Chromosome arms":
        ax.set_yticks(np.arange(len(data.values)), data.values.index, fontsize=12)
    else:
        ax.set_yticks([])
    colorbar_ax = fig.add_subplot(grid[0, 1])
    colorbar = fig.colorbar(
        heat, cax=colorbar_ax, ticks=np.linspace(-data.limit, data.limit, 5)
    )
    colorbar.set_label("CNV log2 ratio", fontsize=13)
    legend_ax = fig.add_subplot(grid[0, 2])
    legend_ax.axis("off")
    legends = []
    for i, (name, colors) in enumerate(tracks.items()):
        track_ax = fig.add_subplot(grid[i + 1, 0])
        keys = list(colors)
        encoded = np.array([keys.index(v) for v in data.metadata[name]])[None, :]
        track_ax.imshow(
            encoded,
            cmap=ListedColormap([to_rgba(v, 0.7) for v in colors.values()]),
            vmin=-0.5,
            vmax=len(keys) - 0.5,
            interpolation="nearest",
            aspect="auto",
        )
        track_ax.set(yticks=[], xticks=[])
        track_ax.set_ylabel(name, rotation=0, va="center", ha="right", fontsize=14)
        for spine in track_ax.spines.values():
            spine.set_visible(False)
        if data.paired and i == len(tracks) - 1:
            track_ax.set_xticks(
                np.arange(len(data.column_labels)),
                data.column_labels,
                rotation=90,
                fontsize=12,
            )
            track_ax.set_xlabel(
                "Matched case (anonymized) and sample type", fontsize=14
            )
        legends.append(
            legend_ax.legend(
                handles=[
                    Patch(facecolor=v, alpha=0.7, label=str(k))
                    for k, v in colors.items()
                ],
                title=name,
                loc="upper left",
                bbox_to_anchor=(0.2, 1 - i * 0.28),
                fontsize=13,
                title_fontsize=15,
                frameon=False,
            )
        )
        if len(legends) > 1:
            legend_ax.add_artist(legends[-2])
    return fig


def make_package_cohort_cnv_heatmap(data: HeatmapData) -> plt.Figure:
    """Render S15 cohort CNV through the original numeric-heatmap API."""
    require_pymaftools_version()
    if data.paired or not data.values.columns.equals(data.metadata.index):
        raise ValueError("Cohort CNV values and metadata must align")
    if not np.isfinite(data.values.to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite cohort CNV values")
    table = PivotTable(data.values.copy())
    table.sample_metadata = data.metadata.copy()
    table.sample_metadata["smoke"] = table.sample_metadata.smoke.astype(str)
    palette = {name: {str(k): v for k, v in colors.items()}
               for name, colors in TRACK_COLORS.items()}
    for name, colors in palette.items():
        if not table.sample_metadata[name].isin(colors).all():
            raise ValueError(f"Unregistered {name} annotation")
    plot = OncoPlot(table).set_config(figsize=(18, 7), ytick_fontsize=9)
    plot.numeric_heatmap(cmap="coolwarm", vmin=-2, vmax=2,
                         yticklabels=data.label != "Genes", linewidths=0)
    plot.add_sample_annotation(list(palette), cmap_dict=palette,
                               alpha=0.7, size=0.55, linewidths=0)
    plot.render(main_width=28, main_height=11, legend_width=2.5,
                legend_pad=1.5, show_sample_labels=False)
    np.testing.assert_allclose(
        np.asarray(plot.ax_heatmap.collections[0].get_array()).reshape(
            data.values.shape), data.values.to_numpy(dtype=float), rtol=0, atol=0
    )
    for name, colors in palette.items():
        plot.legend_manager.legend_dict[name] = {
            key: to_rgba(color, 0.7) for key, color in colors.items()
        }
    plot.plot_all_legends(fontsize=12, title_fontsize=13)
    for text in plot.ax_legend.texts:
        if text.get_text() in palette:
            x, y = text.get_position()
            text.set_position((x, y + 0.015))
    # Same reviewed numeric-legend-only patch as S6: vertical, red at top.
    for child in list(plot.ax_legend.child_axes):
        child.remove()
    for text in list(plot.ax_legend.texts):
        if text.get_text() == "value":
            text.remove()
    axis = plot.ax_heatmap.inset_axes([1.025, 0, 0.013, 1])
    colorbar = plot.fig.colorbar(
        ScalarMappable(norm=Normalize(-2, 2), cmap="coolwarm"),
        cax=axis, ticks=[-2, -1, 0, 1, 2]
    )
    colorbar.ax.tick_params(labelsize=10, length=0)
    colorbar.outline.set_visible(False)
    plot.ax_heatmap.set_ylabel(data.label, fontsize=13)
    plot.ax_heatmap.set_label("cnv_data")
    return plot.fig


def make_package_paired_cnv_heatmap(
    data: HeatmapData, *, show_legend: bool = True
) -> plt.Figure:
    """Draw Supplementary Figure 6A/B with pymaftools 0.5.0 heatmap tracks."""
    require_pymaftools_version()
    if not data.paired or data.label not in {"Chromosome arms", "CNV clusters"}:
        raise ValueError("This renderer is only for paired S6A/B CNV matrices")
    if not data.values.columns.equals(data.metadata.index):
        raise ValueError("CNV columns and sample metadata are not aligned")
    if len(data.column_labels) != len(data.values.columns):
        raise ValueError("Anonymous sample label count does not match CNV columns")
    expected_labels = [
        f"{case}_{part}"
        for case, part in zip(data.metadata.case_ID, data.metadata.sample_type)
    ]
    if data.column_labels != expected_labels:
        raise ValueError("S6 labels do not match their case ID and sample type")
    if data.limit != 1.0 or not np.isfinite(data.values.to_numpy(dtype=float)).all():
        raise ValueError("S6 CNV requires finite values and a fixed ±1 scale")

    table = PivotTable(data.values.copy())
    table.sample_metadata = data.metadata.copy()
    table.columns = data.column_labels
    table.sample_metadata.index = data.column_labels
    table.sample_metadata["patient"] = table.sample_metadata["case_ID"]
    table.sample_metadata["smoke"] = table.sample_metadata.smoke.astype(str)
    palette = {
        "patient": patient_palette(table.sample_metadata["patient"]),
        "sample_type": PART_COLORS,
        "smoke": {str(key): color for key, color in TRACK_COLORS["smoke"].items()},
    }
    for name, colors in palette.items():
        if not table.sample_metadata[name].isin(colors).all():
            raise ValueError(f"Missing {name} color for a paired sample")

    plot = OncoPlot(table).set_config(figsize=(15, 10), ytick_fontsize=10)
    plot.numeric_heatmap(
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        yticklabels=data.label == "Chromosome arms",
        linewidths=0,
    )
    plot.add_sample_annotation(
        ["patient", "sample_type", "smoke"],
        cmap_dict=palette,
        alpha=0.7,
        size=0.65,
        linewidths=0,
    )
    plot.render(
        main_width=24,
        main_height=16,
        legend_width=3.5,
        legend_pad=1.6,
        show_sample_labels=True,
    )
    plotted = np.asarray(plot.ax_heatmap.collections[0].get_array())
    np.testing.assert_allclose(
        plotted.reshape(data.values.shape),
        data.values.to_numpy(dtype=float),
        rtol=0,
        atol=0,
    )
    align_sample_label_gutter(plot)
    plot.add_xticklabel(rotation=90, fontsize=9)
    # Set legend type at its final composite scale, not the package's 8pt default.
    plot.plot_all_legends(fontsize=12, title_fontsize=14)
    # The 0.5.0 legend manager forces numeric legends to be horizontal.
    # Replace only that legend artist; the heatmap and categorical tracks
    # remain package-rendered. Match the manuscript's red-top/blue-bottom bar.
    for child in list(plot.ax_legend.child_axes):
        child.remove()
    for label in list(plot.ax_legend.texts):
        if label.get_text() == "value":
            label.remove()
    colorbar_axis = plot.ax_heatmap.inset_axes([1.045, 0, 0.015, 1])
    colorbar_axis.set_label("cnv_log2_colorbar")
    colorbar = plot.fig.colorbar(
        ScalarMappable(norm=Normalize(-1, 1), cmap="coolwarm"),
        cax=colorbar_axis,
        orientation="vertical",
        ticks=[-1, -0.5, 0, 0.5, 1],
    )
    colorbar.ax.tick_params(labelsize=8, length=2, width=0.5)
    colorbar.ax.yaxis.set_ticks_position("left")
    colorbar.outline.set_linewidth(0)
    if data.label == "Chromosome arms":
        gutters = [
            axis
            for axis in plot.fig.axes
            if axis is not plot.ax_heatmap and len(axis.texts) == len(data.values)
        ]
        if len(gutters) != 1:
            raise ValueError("Expected one chromosome-arm label gutter")
        gutters[0].set_ylabel(data.label, fontsize=13, labelpad=10)
    else:
        plot.ax_heatmap.set_ylabel(data.label, fontsize=13)
    plot.ax_heatmap.set_label("cnv_data")
    if not show_legend:
        plot.fig.delaxes(plot.ax_legend)
    return plot.fig


def run(data_root: Path, output: Path, *, panel=None, prepared=False) -> None:
    """Generate selected S15A or S6A/B heatmap candidates."""
    from .figure_style import setup_plotting

    setup_plotting()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be new or empty")
    root = data_root / "data"
    if prepared:
        data = prepare_heatmaps_from_prepared(data_root)
    else:
        genes = read_pivot_sqlite(root / "CNV_0407/gene_cnv.db")
        arms = read_pivot_sqlite(root / "CNV_0407/arm_cnv.db")
        snv = read_pivot_sqlite(root / "WES/all_case_table.db")
        data = prepare_heatmaps(
            genes.values,
            arms.values,
            snv.values,
            snv.sample_metadata,
            pd.read_csv(root / "clustering/best_k_cluster_labels.csv"),
            pd.read_csv(root / "ASC_all_case_metadata_0312.csv"),
            expected_subtypes={"LUAD": 89, "ASC": 29, "LUSC": 82},
        )
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "panels": {},
        "input_layer": "prepared Cohort pivots" if prepared else "source SQLite",
        "annotation_alpha": 0.7,
        "identifiers_or_sample_tables_written": False,
        "ordering": "S15A legacy full-cohort SNV-pattern order; S6 manuscript case IDs in CSV order, each A/T/S",
    }
    selected = panel or list(data)
    for panel in selected:
        values = data[panel]
        figure = (
            make_package_paired_cnv_heatmap(values, show_legend=panel != "S6A_arm_CNV")
            if values.paired
            else make_package_cohort_cnv_heatmap(values)
        )
        plot_axis = next(
            (axis for axis in figure.axes if axis.get_label() == "cnv_data"),
            None,
        )
        manifest = save_tiff_and_preview(figure, output / panel, plot_axis=plot_axis)
        plt.close(figure)
        matrix = values.values.to_numpy(dtype=float)
        summary["panels"][panel] = {
            "shape": list(matrix.shape),
            "color_limits": [-values.limit, values.limit],
            "minimum": float(matrix.min()),
            "maximum": float(matrix.max()),
            "values_outside_color_limits": int((np.abs(matrix) > values.limit).sum()),
            "renderer": "pymaftools 0.5.0 OncoPlot.numeric_heatmap",
            "figure": manifest,
        }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main() -> None:
    """Read CLI options and call the reusable analysis function."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--prepared",
        action="store_true",
        help="Read Cohort-derived HDF5 pivots under data/prepared",
    )
    parser.add_argument(
        "--panel",
        choices=("S15A_gene_CNV", "S6A_arm_CNV", "S6B_cluster_CNV"),
        action="append",
        help="Render only these panels (repeatable)",
    )
    args = parser.parse_args()
    run(args.data_root, args.output, panel=args.panel, prepared=args.prepared)


if __name__ == "__main__":
    main()
