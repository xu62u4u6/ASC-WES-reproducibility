"""Produce the two S10 panels with shared styling and existing calculations."""

import argparse
import json
from pathlib import Path

import pandas as pd

from .cluster_panels import calculate_cluster_panels, make_cluster_boxplot
from .dataio import read_pivot_sqlite
from .figure_style import PART_COLORS, plt, save_tiff_and_preview, setup_plotting
from .paired_cnv import calculate_paired_3q, make_gain_figure


def style_axis(axis):
    """Apply identical text, frame and category styling to both S10 axes."""
    axis.tick_params(labelsize=12)
    axis.title.set_fontsize(14)
    axis.xaxis.label.set_size(12)
    axis.yaxis.label.set_size(12)
    axis.grid(False)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
    # Seaborn may store hue patches in a different order than x categories.
    patches = sorted(axis.patches, key=lambda p: p.get_extents().x0)
    assert len(patches) == 3
    for patch, part in zip(patches, ("A", "T", "S")):
        patch.set_facecolor(PART_COLORS[part])
        patch.set_alpha(0.6)
    for text in axis.texts:
        text.set_fontsize(12)


def run(data_root: Path, output: Path):
    """Reuse validated C47 and inclusive 3q calculations; write small panels."""
    setup_plotting()
    output.mkdir(parents=True, exist_ok=False)
    genes = read_pivot_sqlite(data_root / "data/CNV_0407/gene_cnv.db")
    arm = read_pivot_sqlite(data_root / "data/CNV_0407/arm_cnv.db")
    metadata = read_pivot_sqlite(
        data_root / "data/WES/all_case_table.db"
    ).sample_metadata
    mapping = pd.read_csv(data_root / "data/clustering/best_k_cluster_labels.csv")
    clusters = calculate_cluster_panels(genes.values, metadata, mapping)
    gain = calculate_paired_3q(arm.values, metadata)
    assert gain.gain_summary.n_gain.tolist() == [2, 4, 8]
    assert clusters.gene_counts["C47"] == 817
    figures = []
    for letter in "AB":
        fig, axis = plt.subplots(figsize=(6, 4.5))
        fig.subplots_adjust(left=0.16, right=0.97, bottom=0.19, top=0.88)
        if letter == "A":
            make_cluster_boxplot(clusters, "C47_component", ax=axis)
        else:
            make_gain_figure(gain.gain_summary, ax=axis, alpha=0.6)
            axis.set_title("3q-gain proportion by component")
            axis.set_ylabel("Samples with 3q gain (%)\n(log2 ratio ≥ 0.10)")
            axis.set_xlabel("sample_type")
        style_axis(axis)
        figures.append(
            save_tiff_and_preview(fig, output / f"S10{letter}", plot_axis=axis)
        )
        plt.close(fig)
    clusters.statistics.loc[clusters.statistics.panel.eq("C47_component")].to_csv(
        output / "S10A_tests.csv", index=False
    )
    clusters.group_summary.loc[clusters.group_summary.panel.eq("C47_component")].to_csv(
        output / "S10A_summary.csv", index=False
    )
    gain.gain_summary.to_csv(output / "S10B_counts.csv", index=False)
    summary = {
        "figures": figures,
        "cases": 11,
        "gain_operator": ">=",
        "gain_threshold": 0.10,
        "gain_counts": [2, 4, 8],
        "renderers": {
            "A": "pymaftools 0.5.0 paired Wilcoxon boxplot",
            "B": "original matplotlib barplot",
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main():
    """Run S10 only, without regenerating S15 or changing model results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.data_root, args.output)


if __name__ == "__main__":
    main()
