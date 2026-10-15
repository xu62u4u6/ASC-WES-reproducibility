"""S7A–D and S16A–C similarity matrices with metric-specific color scales."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap, to_rgba
from pymaftools import PivotTable
from pymaftools.core.PairwiseMatrix import SimilarityMatrix

from .cgc_oncoplot import mutation_presence
from .cnv_heatmaps import HeatmapData, prepare_heatmaps
from .cnv_heatmaps import prepare_heatmaps_from_prepared
from .cohort_inputs import read_prepared_table
from .dataio import read_pivot_sqlite
from .figure_style import (
    PART_COLORS,
    TRACK_COLORS,
    panel_font_properties,
    patient_palette,
    plt,
    save_tiff_and_preview,
)
from .oncoplot_rendering import require_pymaftools_version
from .paired_screen import cluster_mean_matrix


@dataclass(frozen=True)
class SimilarityPanel:
    matrix: np.ndarray
    metadata: pd.DataFrame
    metric: str
    feature_count: int
    paired: bool
    title: str


def sample_similarity(values: pd.DataFrame, metric: str) -> np.ndarray:
    """Compute S7/S16 sample matrices with pymaftools 0.5.0, without inference."""
    require_pymaftools_version()
    x = values.T.to_numpy(dtype=float)
    if len(values) == 0 or not np.isfinite(x).all():
        raise ValueError("Similarity requires nonempty, finite feature values")
    if metric in ("jaccard", "hamming"):
        if not np.isin(x, [0, 1]).all():
            raise ValueError("Jaccard and SMC require binary input")
        table = PivotTable(values.astype(bool))
    elif metric == "cosine":
        table = PivotTable(values.astype(float))
    else:
        raise ValueError("Unsupported similarity metric")
    return table.compute_similarity(method=metric).to_numpy(dtype=float)


def calculate_similarity_panels(
    snv: pd.DataFrame,
    heatmaps: dict[str, HeatmapData],
    mapping: pd.DataFrame,
    *,
    expected_clusters: int = 114,
    paired_only: bool = False,
) -> dict[str, SimilarityPanel]:
    """Calculate the original metric panels with cohort-specific SNV filtering."""
    binary = mutation_presence(snv)
    paired = heatmaps["S6B_cluster_CNV"]
    tumor = heatmaps["S15A_gene_CNV"]
    contexts = [("S7", paired, paired.values)]
    if not paired_only:
        cnv_t = cluster_mean_matrix(
            tumor.values, mapping, expected_cluster_count=expected_clusters
        )
        contexts.append(("S16", tumor, cnv_t))
    results = {}
    for prefix, context, cnv in contexts:
        mutations = binary.loc[:, context.metadata.index]
        mutations = mutations.loc[mutations.mean(axis=1).ge(0.05)]
        specs = [
            (
                "A_Jaccard",
                mutations,
                "jaccard",
                "SNV&INDEL freq 005 Jaccard Similarity",
            ),
            ("B_SMC", mutations, "hamming", "SNV&INDEL freq 005 SMC Similarity"),
            ("C_cluster_cosine", cnv, "cosine", "CNV Cluster-level cosine Similarity"),
        ]
        if context.paired:
            specs.append(
                (
                    "D_arm_cosine",
                    heatmaps["S6A_arm_CNV"].values,
                    "cosine",
                    "CNV Arm-level cosine Similarity",
                )
            )
        for suffix, values, metric, title in specs:
            if list(values.columns) != list(context.metadata.index):
                raise ValueError("Matrix sample order differs from annotation order")
            matrix = sample_similarity(values, metric)
            results[f"{prefix}{suffix}"] = SimilarityPanel(
                matrix,
                context.metadata,
                metric,
                len(values),
                context.paired,
                ("ATS " if context.paired else "") + title,
            )
    return results


def draw_similarity_panel(fig, spec, panel: SimilarityPanel, letter: str | None = None):
    """Use the original package heatmap API and adapt its fixed scale/alpha.

    Version 0.5.0 hardcodes [-1,1] and annotation alpha=.4. Adjust those
    artists after rendering to retain the reviewed metric-specific scales
    and patient/subtype opacity. The second ATS track is a local annotation.
    """
    require_pymaftools_version()
    rows = 3 if panel.paired else 2
    grid = spec.subgridspec(
        rows,
        2,
        width_ratios=[20, 1],
        height_ratios=[20] + [0.45] * (rows - 1),
        hspace=0.015,
        wspace=0.04,
    )
    ax = fig.add_subplot(grid[0, 0])
    bar = fig.add_subplot(grid[0, 1])
    lower = 0 if panel.metric in ("jaccard", "hamming") else -1
    cmap = "Blues" if lower == 0 else "coolwarm"
    groupbar = fig.add_subplot(grid[1, 0])
    groups = panel.metadata.case_ID if panel.paired else panel.metadata.subtype
    colors = patient_palette(groups) if panel.paired else TRACK_COLORS["subtype"]
    matrix = SimilarityMatrix(
        panel.matrix, index=panel.metadata.index, columns=panel.metadata.index
    )
    matrix.plot_similarity(
        groups=groups,
        group_cmap=colors,
        cmap=cmap,
        ax=(ax, bar, groupbar),
        title=panel.title,
        title_fontsize=11,
    )
    heat = ax.collections[0]
    np.testing.assert_allclose(
        np.asarray(heat.get_array()).reshape(panel.matrix.shape),
        panel.matrix,
        rtol=0,
        atol=0,
    )
    heat.set_clim(lower, 1)
    heat.set_alpha(1)
    heat.colorbar.solids.set_alpha(1)
    heat.colorbar.set_ticks(np.linspace(lower, 1, 5))
    for patch in groupbar.patches:
        patch.set_alpha(1 if panel.paired else 0.8)
        patch.set_linewidth(0)
    groupbar.set(xlabel="", ylabel="")
    groupbar.text(
        -0.02,
        0.5,
        "patient" if panel.paired else "subtype",
        transform=groupbar.transAxes,
        ha="right",
        va="center",
        fontsize=8,
    )
    ax.set(xticks=[], yticks=[], ylabel="sample")
    ax.set_title(panel.title, fontsize=11)
    if letter:
        ax.text(
            -0.12,
            1.08,
            letter,
            transform=ax.transAxes,
            fontproperties=panel_font_properties(22),
            va="top",
        )
    if panel.paired:
        tracks = [
            ("sample type", panel.metadata.sample_type, PART_COLORS, 0.55),
        ]
    else:
        tracks = []
    for i, (name, values, colors, alpha) in enumerate(tracks, 2):
        strip = fig.add_subplot(grid[i, 0])
        keys = list(colors)
        codes = np.array([keys.index(value) for value in values])[None, :]
        strip.imshow(
            codes,
            aspect="auto",
            interpolation="nearest",
            cmap=ListedColormap([to_rgba(v, alpha) for v in colors.values()]),
            vmin=-0.5,
            vmax=len(keys) - 0.5,
        )
        strip.set(xticks=[], yticks=[])
        strip.text(
            -0.02,
            0.5,
            name,
            transform=strip.transAxes,
            ha="right",
            va="center",
            fontsize=8,
        )
        for spine in strip.spines.values():
            spine.set_visible(False)
    if not panel.paired:
        # Interpret subtype colors without writing any sample identifiers.
        ax.text(
            0.5,
            -0.10,
            "LUAD (orange)  ·  ASC (green)  ·  LUSC (blue)",
            transform=ax.transAxes,
            ha="center",
            fontsize=8,
        )


def run(data_root: Path, output: Path) -> None:
    """Generate S7A–D and S16A–C candidates with metric-specific scales."""
    from .figure_style import setup_plotting

    setup_plotting()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be new or empty")
    root = data_root / "data"
    genes = read_pivot_sqlite(root / "CNV_0407/gene_cnv.db")
    arms = read_pivot_sqlite(root / "CNV_0407/arm_cnv.db")
    snv = read_pivot_sqlite(root / "WES/all_case_table.db")
    mapping = pd.read_csv(root / "clustering/best_k_cluster_labels.csv")
    heatmaps = prepare_heatmaps(
        genes.values,
        arms.values,
        snv.values,
        snv.sample_metadata,
        mapping,
        pd.read_csv(root / "ASC_all_case_metadata_0312.csv"),
        expected_subtypes={"LUAD": 89, "ASC": 29, "LUSC": 82},
    )
    panels = calculate_similarity_panels(snv.values, heatmaps, mapping)
    output.mkdir(parents=True, exist_ok=True)
    records, images = [], []
    for name, panel in panels.items():
        fig = plt.figure(figsize=(8, 8.1))
        draw_similarity_panel(fig, fig.add_gridspec(1, 1)[0], panel)
        images.append(save_tiff_and_preview(fig, output / name))
        plt.close(fig)
        upper = panel.matrix[np.triu_indices(len(panel.matrix), k=1)]
        records.append(
            {
                "panel": name,
                "metric": panel.metric,
                "n_samples": len(panel.matrix),
                "n_features": panel.feature_count,
                "minimum": float(panel.matrix.min()),
                "maximum": float(panel.matrix.max()),
                "mean_unique_off_diagonal": float(upper.mean()),
                "diagonal_minimum": float(panel.matrix.diagonal().min()),
                "diagonal_maximum": float(panel.matrix.diagonal().max()),
                "color_min": 0 if panel.metric != "cosine" else -1,
                "color_max": 1,
            }
        )
    fig = plt.figure(figsize=(16, 16.2))
    grid = fig.add_gridspec(2, 2, wspace=0.30, hspace=0.22)
    for i, (_, panel) in enumerate(
        (x for x in panels.items() if x[0].startswith("S7"))
    ):
        draw_similarity_panel(fig, grid[i // 2, i % 2], panel, "ABCD"[i])
    images.append(save_tiff_and_preview(fig, output / "S7_ABCD_matrix_panels"))
    plt.close(fig)
    pd.DataFrame(records).to_csv(output / "similarity_summary.csv", index=False)
    summary = {
        "renderer": "pymaftools 0.5.0 SimilarityMatrix.plot_similarity",
        "calculator": "pymaftools 0.5.0 PivotTable.compute_similarity",
        "panels": records,
        "figures": images,
        "mutation_filter": ">=0.05 within each displayed cohort",
        "binary_definitions": {
            "Jaccard": "shared presences / union of presences",
            "SMC": "matching presences and absences / all selected genes",
        },
        "zero_vector_convention": "two all-zero binary vectors have Jaccard=SMC=1; cosine with a zero vector=0 (sklearn)",
        "ATS_annotation_tracks": {
            "patient": "same color within case",
            "sample type": "A/T/S; alpha 0.55",
        },
        "matrix_values_or_identifiers_exported": False,
        "inferential_tests": "none; dependent within/between-patient and subtype comparisons remain pending",
        "composite_scope": "S7A-D matrices only; does not include S7E-G",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"panels": records, "tiffs": len(images)}, indent=2))


def within_across_records(panel: SimilarityPanel) -> pd.DataFrame:
    """Retain legacy directed across pairs for descriptive replication only.

    Mirrors similarity_analysis._within_across_records. Across pairs occur
    twice (both directions); observations share patients. No test is run.
    """
    metadata = panel.metadata
    if metadata.duplicated(["case_ID", "sample_type"]).any():
        raise ValueError("Duplicate case/component")
    lookup = {
        (row.case_ID, row.sample_type): index
        for index, row in enumerate(metadata.itertuples())
    }
    cases = metadata.case_ID.unique()
    if len(lookup) != len(cases) * 3:
        raise ValueError("Complete ATS triplets required")
    records = []
    for case in cases:
        pairs = [
            (case, a, case, b, "within")
            for a, b in [("A", "T"), ("A", "S"), ("T", "S")]
        ]
        pairs.extend(
            (case, part, other, part, f"{part}-across")
            for other in cases
            if other != case
            for part in ("A", "T", "S")
        )
        for first, a, second, b, group in pairs:
            value = panel.matrix[lookup[first, a], lookup[second, b]]
            records.append((first, a, second, b, group, value))
    return pd.DataFrame(
        records, columns=["case_1", "type_1", "case_2", "type_2", "group", "similarity"]
    )


def draw_within_across(ax, records: pd.DataFrame, title: str, metric: str) -> list:
    """Restore notebook two-sided U tests, retaining its dependent pair inputs."""
    import seaborn as sns
    from statannotations.Annotator import Annotator

    order = ["within", "A-across", "T-across", "S-across"]
    sns.boxplot(
        data=records,
        x="group",
        y="similarity",
        hue="group",
        order=order,
        hue_order=order,
        palette="Set2",
        legend=False,
        ax=ax,
    )
    ax.set_title(title, fontsize=11)
    ax.set(xlabel="group", ylabel="similarity")
    lower = min(0, float(records.similarity.min()) - 0.03)
    if metric == "hamming":
        lower = 0.5
        if records.similarity.min() < lower:
            raise ValueError("SMC values fall below the manuscript axis limit")
    ax.set_ylim(lower, 1)
    pairs = [("within", group) for group in order[1:]]
    annotator = Annotator(
        ax, pairs, data=records, x="group", y="similarity", order=order
    )
    annotator.configure(
        test="Mann-Whitney",
        text_format="star",
        loc="inside",
        verbose=0,
        comparisons_correction=None,
    )
    annotator.apply_and_annotate()
    return [
        {
            "group_1": annotation.structs[0]["label"],
            "group_2": annotation.structs[1]["label"],
            "U": float(annotation.data.stat_value),
            "p_value": float(annotation.data.pvalue),
            "alternative": "two-sided",
            "multiple_testing_correction": "none (notebook)",
        }
        for annotation in annotator.annotations
    ]


def run_s7_prepared(data_root: Path, output: Path) -> None:
    """Export S7 A–G and traceable legacy pair records from prepared pivots."""
    from .figure_style import setup_plotting

    setup_plotting()
    if output.exists():
        raise FileExistsError("Use a new S7 candidate directory")
    snv = read_prepared_table(data_root, "paired", "SNV")
    heatmaps = prepare_heatmaps_from_prepared(data_root)
    panels = calculate_similarity_panels(
        pd.DataFrame(snv), heatmaps, pd.DataFrame(), paired_only=True
    )
    output.mkdir(parents=True)
    summaries = {}
    tests = []
    for name, panel in panels.items():
        assert panel.matrix.shape == (33, 33)
        np.testing.assert_allclose(panel.matrix, panel.matrix.T, atol=1e-14)
        pd.DataFrame(
            panel.matrix, index=panel.metadata.index, columns=panel.metadata.index
        ).to_csv(output / f"{name}_matrix.csv")
        fig = plt.figure(figsize=(8, 8.1))
        draw_similarity_panel(fig, fig.add_gridspec(1, 1)[0], panel)
        summaries[name] = save_tiff_and_preview(fig, output / name)
        plt.close(fig)
    for letter, panel in zip("EFG", list(panels.values())[:3]):
        records = within_across_records(panel)
        assert records.groupby("group").size().to_dict() == {
            "within": 33,
            "A-across": 110,
            "T-across": 110,
            "S-across": 110,
        }
        records.to_csv(output / f"S7{letter}_pairs.csv", index=False)
        fig, ax = plt.subplots(figsize=(6, 5))
        results = draw_within_across(ax, records, panel.title, panel.metric)
        tests.extend({"panel": f"S7{letter}", **result} for result in results)
        summaries[f"S7{letter}"] = save_tiff_and_preview(fig, output / f"S7{letter}")
        plt.close(fig)
    pd.DataFrame(tests).to_csv(output / "S7_EFG_notebook_tests.csv", index=False)
    (output / "summary.json").write_text(
        json.dumps(
            {
                "input": "data/prepared Cohort pivots",
                "calculator": "pymaftools 0.5.0",
                "heatmap_renderer": "SimilarityMatrix.plot_similarity",
                "boxplot_renderer": "seaborn Set2 + statannotations Mann-Whitney",
                "across_pairs": "110 directed entries per group, 55 unique pairs",
                "status": "notebook replication; dependent-pair validity unresolved",
                "tests": "two-sided Mann-Whitney U; no multiplicity correction",
                "figures": summaries,
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    """Read CLI options and call the reusable analysis function."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--s7-prepared", action="store_true")
    args = parser.parse_args()
    if args.s7_prepared:
        run_s7_prepared(args.data_root, args.output)
    else:
        run(args.data_root, args.output)


if __name__ == "__main__":
    main()
