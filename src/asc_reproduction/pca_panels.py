"""Figure 5A/B and S6C/D PCA using the legacy unscaled inputs."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA

from .cgc_oncoplot import mutation_presence
from .cnv_heatmaps import (
    HeatmapData,
    prepare_heatmaps,
    prepare_heatmaps_from_prepared,
)
from .cohort_inputs import read_prepared_table
from .dataio import read_pivot_sqlite
from .figure_style import (
    CASE_COLORS,
    TRACK_COLORS,
    patient_palette,
    plt,
    save_tiff_and_preview,
)
from .paired_screen import cluster_mean_matrix

EXPECTED_S6_CASE_IDS = (
    "AS_001",
    "AS_005",
    "AS_006",
    "AS_007",
    "AS_008",
    "AS_009",
    "AS_010",
    "AS_016",
    "AS_020",
    "AS_022",
    "AS_027",
)


@dataclass(frozen=True)
class PCAResult:
    inputs: pd.DataFrame
    scores: np.ndarray
    variance: np.ndarray
    components: np.ndarray
    metadata: pd.DataFrame
    paired: bool
    cutoff: float | None


def calculate_pca_panels(
    snv: pd.DataFrame,
    heatmaps: dict[str, HeatmapData],
    mapping: pd.DataFrame,
    *,
    expected_clusters: int = 114,
    expected_snv_counts: tuple[int, int] | None = (441, 90),
) -> dict[str, PCAResult]:
    binary = mutation_presence(snv)
    tumor = heatmaps["S15A_gene_CNV"]
    paired = heatmaps["S6B_cluster_CNV"]
    cnv_t = cluster_mean_matrix(
        tumor.values, mapping, expected_cluster_count=expected_clusters
    )
    results = {}
    configurations = [
        ("Figure_5A_SNV_PCA", tumor, binary.loc[:, tumor.metadata.index], 0.05),
        ("Figure_5B_CNV_PCA", tumor, cnv_t, None),
        ("S6C_SNV_PCA", paired, binary.loc[:, paired.metadata.index], 0.10),
        ("S6D_CNV_PCA", paired, paired.values, None),
    ]
    for panel, data, matrix, cutoff in configurations:
        if cutoff is not None:
            matrix = matrix.loc[matrix.mean(axis=1).ge(cutoff)]
        matrix = matrix.astype(float)
        if min(matrix.shape) < 2 or not np.isfinite(matrix.to_numpy()).all():
            raise ValueError(
                "PCA requires finite values and at least two samples/features"
            )
        # Legacy auto chooses full SVD for all four current input dimensions.
        # Explicit full fixes that effective algorithm, with centering but no scaling.
        fit = PCA(n_components=2, svd_solver="full")
        scores = fit.fit_transform(matrix.T.to_numpy())
        results[panel] = PCAResult(
            matrix,
            scores,
            fit.explained_variance_ratio_,
            fit.components_,
            data.metadata,
            data.paired,
            cutoff,
        )
    if expected_snv_counts is not None:
        counts = (
            len(results["Figure_5A_SNV_PCA"].inputs),
            len(results["S6C_SNV_PCA"].inputs),
        )
        if counts != expected_snv_counts:
            raise ValueError(
                f"Unexpected subtype/component mutation feature counts: {counts}"
            )
    return results


def make_pca_figure(result: PCAResult) -> plt.Figure:
    """Render subtype and matched-case panels with pymaftools 0.5.0."""
    if result.paired:
        return make_package_paired_pca_figure(result)
    return make_package_pca_figure(result)


def make_package_pca_figure(
    result: PCAResult, *, show_legend: bool = True
) -> plt.Figure:
    """Call the original package PCA API and verify unchanged coordinates.

    Already-binary SNV and raw cluster means both use to_binary=False here;
    a second event conversion would incorrectly change continuous CNVs.
    Paired-case styling is migrated separately, not silently altered.
    """
    from pymaftools import PivotTable

    from .oncoplot_rendering import require_pymaftools_version

    require_pymaftools_version()
    if not result.inputs.columns.equals(result.metadata.index):
        raise ValueError("PCA input and metadata sample order differ")
    table = PivotTable(result.inputs.copy())
    table.sample_metadata = result.metadata.copy()
    plotting = table.plot
    coordinates, variance, fitted = plotting.plot_pca_samples(
        color_col="subtype",
        to_binary=False,
        palette=TRACK_COLORS["subtype"],
        figsize=(10, 6),
        alpha=0.8,
        s=20,
        fontsize=12,
    )
    np.testing.assert_allclose(
        coordinates[["PC1", "PC2"]], result.scores, rtol=0, atol=1e-10
    )
    np.testing.assert_allclose(variance, result.variance, rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        fitted.components_, result.components, rtol=0, atol=1e-10
    )
    plotting.fig.axes[0].grid(True, color="lightgray", linewidth=0.7)
    plotting.fig.axes[0].set_axisbelow(True)
    if not show_legend:
        plotting.fig.axes[1].clear()
        plotting.fig.axes[1].axis("off")
    return plotting.fig


def make_package_paired_pca_figure(
    result: PCAResult, *, show_legend: bool = True
) -> plt.Figure:
    """Plot S6C/D with the original package API and verify unchanged PCA.

    Case IDs, the original tab20 palette, and A/T/S markers match the source.
    Package PCA receives the already prepared binary SNV or raw CNV matrix;
    it must not convert continuous CNV values to events or rescale features.
    """
    from pymaftools import PivotTable

    from .oncoplot_rendering import require_pymaftools_version

    require_pymaftools_version()
    if not result.paired or not result.inputs.columns.equals(result.metadata.index):
        raise ValueError("Paired PCA input and metadata must be aligned")
    if result.metadata[["case_ID", "sample_type"]].isna().to_numpy().any():
        raise ValueError("Paired PCA has missing case or component labels")
    cases = result.metadata.case_ID.drop_duplicates().tolist()
    expected_cases = len(CASE_COLORS)
    expected_samples = expected_cases * 3
    if len(cases) != expected_cases or len(result.metadata) != expected_samples:
        raise ValueError(f"S6 PCA requires {expected_cases} matched A/T/S triplets")
    for _, group in result.metadata.groupby("case_ID", sort=False):
        if len(group) != 3 or set(group.sample_type) != {"A", "T", "S"}:
            raise ValueError("S6 PCA has an incomplete A/T/S triplet")
    if result.metadata.sample_type.drop_duplicates().tolist() != ["A", "T", "S"]:
        raise ValueError("Package markers require first-occurrence order A/T/S")

    palette = patient_palette(result.metadata.case_ID)
    table = PivotTable(result.inputs.copy())
    table.sample_metadata = result.metadata.copy()
    plotting = table.plot
    coordinates, variance, fitted = plotting.plot_pca_samples(
        color_col="case_ID",
        shape_col="sample_type",
        to_binary=False,
        palette=palette,
        figsize=(10, 6),
        fontsize=12,
        s=20,
    )
    np.testing.assert_allclose(
        coordinates[["PC1", "PC2"]], result.scores, rtol=0, atol=1e-10
    )
    np.testing.assert_allclose(variance, result.variance, rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        fitted.components_, result.components, rtol=0, atol=1e-10
    )
    plotting.fig.axes[0].grid(True, color="lightgray", linewidth=0.7)
    plotting.fig.axes[0].set_axisbelow(True)
    if not show_legend:
        plot_axis, legend_axis = plotting.fig.axes[:2]
        plotting.fig.delaxes(legend_axis)
        # Keep the data rectangle identical to D; assembly allocates legend space.
    return plotting.fig


def make_reference_pca_figure(result: PCAResult) -> plt.Figure:
    """Retain the old renderer for paired panels and historical comparison."""
    fig, (ax, legend_ax) = plt.subplots(
        1,
        2,
        figsize=(10, 6) if not result.paired else (8, 6),
        gridspec_kw={"width_ratios": [4, 1]},
    )
    legend_ax.axis("off")
    if result.paired:
        cases = sorted(result.metadata.case_ID.unique())
        markers = {"A": "o", "T": "s", "S": "^"}
        case_colors = patient_palette(cases)
        for case in cases:
            for part, marker in markers.items():
                selected = (
                    result.metadata.case_ID.eq(case)
                    & result.metadata.sample_type.eq(part)
                ).to_numpy()
                ax.scatter(
                    result.scores[selected, 0],
                    result.scores[selected, 1],
                    color=case_colors[case],
                    marker=marker,
                    s=60,
                    alpha=0.8,
                )
        handles = [
            Line2D(
                [],
                [],
                color=case_colors[case],
                marker="o",
                linestyle="",
                label=str(i + 1),
            )
            for i, case in enumerate(cases)
        ]
        cases_legend = legend_ax.legend(
            handles=handles, title="Matched case", loc="upper left", frameon=False
        )
        legend_ax.add_artist(cases_legend)
        legend_ax.legend(
            handles=[
                Line2D([], [], color="gray", marker=marker, linestyle="", label=part)
                for part, marker in markers.items()
            ],
            title="Sample type",
            loc="lower left",
            frameon=False,
        )
    else:
        for group, color in TRACK_COLORS["subtype"].items():
            selected = result.metadata.subtype.eq(group).to_numpy()
            ax.scatter(
                result.scores[selected, 0],
                result.scores[selected, 1],
                color=color,
                label=group,
                s=20,
                alpha=0.8,
            )
        handles, labels = ax.get_legend_handles_labels()
        legend_ax.legend(
            handles, labels, title="subtype", loc="upper left", frameon=False
        )
    ax.set_xlabel(f"PC1 ({result.variance[0] * 100:.2f}%)")
    ax.set_ylabel(f"PC2 ({result.variance[1] * 100:.2f}%)")
    return fig


def run(data_root: Path, output: Path, *, panel=None, prepared=False) -> None:
    """Generate Figure 5A/B and S6C/D PCA without parsing CLI arguments."""
    from .figure_style import setup_plotting

    setup_plotting()
    from .provenance import file_hash, runtime_identity

    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be new or empty")
    root = data_root / "data"
    relative_inputs = (
        "CNV_0407/gene_cnv.db",
        "CNV_0407/arm_cnv.db",
        "WES/all_case_table.db",
        "clustering/best_k_cluster_labels.csv",
        "ASC_all_case_metadata_0312.csv",
    )
    input_hashes = {name: file_hash(root / name) for name in relative_inputs}
    runtime = runtime_identity()
    mapping = pd.read_csv(root / "clustering/best_k_cluster_labels.csv")
    if prepared:
        snv_t = read_prepared_table(data_root, "tumor", "SNV")
        snv_ats = read_prepared_table(data_root, "paired", "SNV")
        non_t_paired = snv_ats.columns[~snv_ats.columns.isin(snv_t.columns)]
        snv_values = pd.concat(
            [pd.DataFrame(snv_t), pd.DataFrame(snv_ats.loc[:, non_t_paired])],
            axis=1,
        )
        if snv_values.columns.has_duplicates or len(snv_values.columns) != 222:
            raise ValueError("Combined prepared SNV cohort must have 222 samples")
        heatmaps = prepare_heatmaps_from_prepared(data_root)
    else:
        genes = read_pivot_sqlite(root / "CNV_0407/gene_cnv.db")
        arms = read_pivot_sqlite(root / "CNV_0407/arm_cnv.db")
        snv = read_pivot_sqlite(root / "WES/all_case_table.db")
        snv_values = snv.values
        heatmaps = prepare_heatmaps(
            genes.values,
            arms.values,
            snv.values,
            snv.sample_metadata,
            mapping,
            pd.read_csv(root / "ASC_all_case_metadata_0312.csv"),
            expected_subtypes={"LUAD": 89, "ASC": 29, "LUSC": 82},
        )
    results = calculate_pca_panels(snv_values, heatmaps, mapping)
    for paired_panel in ("S6C_SNV_PCA", "S6D_CNV_PCA"):
        case_ids = tuple(results[paired_panel].metadata.case_ID.drop_duplicates())
        if case_ids != EXPECTED_S6_CASE_IDS:
            raise ValueError(f"{paired_panel} case IDs/order differ from manuscript")
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "input_layer": "prepared Cohort pivots" if prepared else "source SQLite",
        "preprocessing": "binary SNV, raw cluster-mean CNV; PCA centers columns; no variance scaling",
        "PCA_solver": "full (same effective solver as legacy auto for these inputs)",
        "sample_coordinates_or_identifiers_written": False,
        "panels": {},
    }
    selected = set(panel) if panel else set(results)
    if not selected or not selected.issubset(results):
        raise ValueError("Requested PCA panels must exist in calculated results")
    for panel, result in results.items():
        if panel not in selected:
            continue
        figure = (
            make_package_paired_pca_figure(result, show_legend=panel != "S6C_SNV_PCA")
            if result.paired
            else make_package_pca_figure(
                result, show_legend=panel != "Figure_5A_SNV_PCA"
            )
        )
        manifest = save_tiff_and_preview(
            figure,
            output / panel,
            plot_axis=figure.axes[0] if result.paired else None,
        )
        plt.close(figure)
        pd.DataFrame(
            {
                "feature": result.inputs.index,
                "PC1_loading": result.components[0],
                "PC2_loading": result.components[1],
            }
        ).to_csv(output / f"{panel}_loadings.csv", index=False)
        summary["panels"][panel] = {
            "n_features": len(result.inputs),
            "n_samples": len(result.metadata),
            "mutation_frequency_cutoff": result.cutoff,
            "mutation_frequency_operator": ">=" if result.cutoff is not None else None,
            "explained_variance_ratio": result.variance.tolist(),
            "figure": manifest,
        }
        summary["panels"][panel]["renderer"] = "pymaftools 0.5.0 plot_pca_samples"
    if runtime != runtime_identity() or input_hashes != {
        name: file_hash(root / name) for name in relative_inputs
    }:
        raise RuntimeError("PCA inputs or source changed during rendering")
    summary["input_sha256"] = input_hashes
    summary["runtime"] = runtime
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
        action="append",
        choices=(
            "Figure_5A_SNV_PCA",
            "Figure_5B_CNV_PCA",
            "S6C_SNV_PCA",
            "S6D_CNV_PCA",
        ),
        help="Render only selected panels; default is all four.",
    )
    args = parser.parse_args()
    run(args.data_root, args.output, panel=args.panel, prepared=args.prepared)


if __name__ == "__main__":
    main()
