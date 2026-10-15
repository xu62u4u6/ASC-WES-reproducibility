"""Recompute Figure 3 A/B/C together, using the verified single-panel functions."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .c47_profile import (
    _as_sample_mask,
    calculate_c47_component_profile,
    make_c47_component_figure,
)
from .cohort_inputs import read_prepared_cohort
from .dataio import read_pivot_sqlite
from .figure_style import panel_font_properties, plt, save_tiff_and_preview
from .paired_c47_difference import (
    calculate_paired_c47_difference,
    make_paired_c47_difference_figure,
)
from .paired_cnv import (
    EXPECTED_GAIN_COUNTS,
    EXPECTED_PAIRED_P,
    calculate_paired_3q,
    make_paired_figure,
)


def make_figure(arm, profile, difference, *, case_labels=None):
    """Use the inspected manuscript layout: A/C above full-width B."""
    # Numeric artists, not rescaled PNG/TIFF panels. Preserve individual CLIs.
    fig = plt.figure(figsize=(14, 10), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[1.4, 1])
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[1, :]),
        fig.add_subplot(grid[0, 1]),
    ]
    make_paired_figure(arm.values, arm.paired_p_value, ax=axes[0])
    make_c47_component_figure(profile, ax=axes[1])
    make_paired_c47_difference_figure(difference, ax=axes[2])
    axes[2].set_xlabel("Sample ID")
    if case_labels is not None:
        if len(case_labels) != len(difference.values):
            raise ValueError("Case labels and paired differences must align")
        axes[2].set_xticks(np.arange(1, len(case_labels) + 1), case_labels, rotation=90)
    for letter, ax in zip("ABC", axes):
        # Physical point offsets remain equal for single- and double-width axes.
        ax.annotate(
            letter,
            xy=(0, 1),
            xycoords="axes fraction",
            xytext=(-28, 10),
            textcoords="offset points",
            fontproperties=panel_font_properties(22),
            ha="left",
            va="bottom",
            annotation_clip=False,
        )
        ax.title.set_fontsize(15)
        ax.xaxis.label.set_fontsize(13)
        ax.yaxis.label.set_fontsize(13)
        ax.tick_params(labelsize=11)
    # Rebuild at the target font size so spacing is not inherited from 14 pt.
    axes[1].legend(
        loc="upper left",
        ncols=2,
        fontsize=10,
        labelspacing=0.25,
        borderpad=0.4,
        handlelength=2,
        handletextpad=0.6,
        borderaxespad=0.5,
    )
    axes[1].title.set_fontsize(12)
    axes[1].xaxis.label.set_fontsize(10)
    axes[1].yaxis.label.set_fontsize(10)
    axes[1].tick_params(labelsize=9)
    for text in axes[1].texts:
        if text.get_text() not in ("A", "B", "C"):
            text.set_fontsize(9)
    return fig


def run(data_root, output, *, prepared=False):
    """Render Figure 3 from source SQLite or the verified prepared Cohort."""
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Refusing to overwrite Figure 3 output")
    data = Path(data_root) / "data"
    if prepared:
        cohort = read_prepared_cohort(Path(data_root))
        arm_values = pd.DataFrame(cohort.CNV_arm)
        gene_values = pd.DataFrame(cohort.CNV_gene)
        gene_metadata = cohort.CNV_gene.feature_metadata
        meta = cohort.sample_metadata
    else:
        cohort = read_pivot_sqlite(data / "WES/all_case_table.db")
        arms = read_pivot_sqlite(data / "CNV_0407/arm_cnv.db")
        genes = read_pivot_sqlite(data / "CNV_0407/gene_cnv.db")
        arm_values = arms.values
        gene_values = genes.values
        gene_metadata = genes.feature_metadata
        meta = cohort.sample_metadata
    if set(arm_values.columns) != set(meta.index) or set(gene_values.columns) != set(
        meta.index
    ):
        raise ValueError("Figure 3 omics sample sets differ")
    selected = meta.loc[
        _as_sample_mask(meta.AS_sample) & meta.sample_type.isin(["A", "T", "S"])
    ]
    # Check the historical A complete-triplet rule and B/C AS_sample rule select
    # the same cases; do not silently combine different 11-case populations.
    counts = (
        meta.loc[meta.sample_type.isin(["A", "T", "S"])]
        .groupby("case_ID")
        .sample_type.agg(set)
    )
    complete_cases = set(counts.index[counts.map(lambda v: v == {"A", "T", "S"})])
    selected_counts = selected.groupby("case_ID").sample_type.agg(set)
    if (
        set(selected_counts.index) != complete_cases
        or not selected_counts.map(lambda v: v == {"A", "T", "S"}).all()
    ):
        raise ValueError("Figure 3 panel case selections differ")
    mapping = pd.read_csv(data / "clustering/best_k_cluster_labels.csv")
    arm = calculate_paired_3q(arm_values, meta)
    profile = calculate_c47_component_profile(gene_values, gene_metadata, meta, mapping)
    difference = calculate_paired_c47_difference(gene_values, meta, mapping)
    if (
        not np.isclose(arm.paired_p_value, EXPECTED_PAIRED_P, atol=1e-12, rtol=0)
        or tuple(arm.gain_summary.n_gain) != EXPECTED_GAIN_COUNTS
        or len(profile.means) != 817
        or difference.c47_gene_count != 817
        or not difference.values.A_minus_S.lt(0).all()
    ):
        raise ValueError("Figure 3 differs from verified numeric anchors")
    # Average gene mean difference must equal mean of paired cluster differences.
    if not np.isclose(
        (profile.means.A - profile.means.S).mean(),
        difference.values.A_minus_S.mean(),
        atol=1e-12,
        rtol=0,
    ):
        raise ValueError("Figure 3 B/C aggregation is inconsistent")
    output.mkdir(parents=True, exist_ok=True)
    figure = make_figure(
        arm,
        profile,
        difference,
        case_labels=sorted(selected.case_ID.unique()),
    )
    image_info = save_tiff_and_preview(figure, output / "Figure_3_ABC")
    plt.close(figure)
    profile.means.to_csv(
        output / "C47_gene_level_component_means.csv", index_label="feature"
    )
    summary = {
        "scope": "Figure 3 numeric composite; caption reconciliation pending",
        "input_layer": "prepared Cohort" if prepared else "source SQLite",
        "panels": ["A", "B", "C"],
        "matched_cases": len(arm.values),
        "same_case_selection_verified": True,
        "C47_genes": len(profile.means),
        "paired_S_vs_A_Wilcoxon_raw_p": arm.paired_p_value,
        "gain_counts_A_T_S_ge_0_10": arm.gain_summary.n_gain.tolist(),
        "negative_C47_A_minus_S": int(difference.values.A_minus_S.lt(0).sum()),
        "B_C_aggregation_consistent": True,
        "figure": image_info,
        "manuscript_case_labels_in_figure": True,
        "per_case_tables_written": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    """Parse Figure 3 data source and output location."""
    from .figure_style import setup_plotting

    setup_plotting()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--prepared",
        action="store_true",
        help="Read the Cohort HDF5 under data/prepared",
    )
    args = parser.parse_args()
    run(args.data_root, args.output, prepared=args.prepared)


if __name__ == "__main__":
    main()
