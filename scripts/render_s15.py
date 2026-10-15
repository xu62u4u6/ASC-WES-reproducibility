"""Generate S15 panels from shared cohort pivots and existing analyses."""

import argparse
import json
from pathlib import Path

import pandas as pd
from matplotlib_venn import venn2

from asc_reproduction.cluster_panels import (
    calculate_cluster_panels, make_c6_difference, make_cluster_boxplot,
)
from asc_reproduction.cnv_heatmaps import HeatmapData, make_package_cohort_cnv_heatmap
from asc_reproduction.cohort_inputs import read_prepared_cohort, read_prepared_table
from asc_reproduction.figure_style import plt, save_tiff_and_preview, setup_plotting
from asc_reproduction.subtype_cnv import chr_sort_key


def render_overlap(summary):
    """Restore original arm-name Venn labels using verified saved sets."""
    arms = set(summary["arm_level_significant_arms"])
    clusters = set(summary["cluster_represented_arms"])
    shared = sorted(arms & clusters, key=chr_sort_key)
    assert len(shared) == 29
    figure, axis = plt.subplots(figsize=(6, 7))
    diagram = venn2(
        [clusters, arms],
        set_labels=("Cluster-level\nsignificant arms", "Arm-level\nsignificant arms"),
        ax=axis,
    )
    labels = {
        "10": "\n".join(sorted(clusters - arms, key=chr_sort_key)),
        "01": "\n".join(sorted(arms - clusters, key=chr_sort_key)),
        "11": "\n".join("  ".join(shared[i:i + 2])
                         for i in range(0, len(shared), 2)),
    }
    for key, label in labels.items():
        diagram.get_label_by_id(key).set_text(label)
        diagram.get_label_by_id(key).set_fontsize(12)
    for label in diagram.set_labels:
        label.set_fontsize(12)
        x, _ = label.get_position()
        label.set_position((x, -0.62))
        label.set_verticalalignment("top")
    return figure


def render(data_root: Path, output: Path, statistics: Path) -> None:
    """Keep sample-key alignment and save panels with data-axis geometry."""
    setup_plotting()
    output.mkdir(parents=True, exist_ok=False)
    gene = read_prepared_table(data_root, "tumor", "CNV_gene")
    arm = read_prepared_table(data_root, "tumor", "CNV_arm")
    assert gene.columns.equals(arm.columns)
    assert gene.shape == (25988, 200) and arm.shape == (44, 200)
    manifests = {}
    for letter, table, label in (("A", gene, "Genes"),
                                  ("B", arm, "Chromosome arms")):
        data = HeatmapData(pd.DataFrame(table), table.sample_metadata,
                           [], label, 2.0, False)
        figure = make_package_cohort_cnv_heatmap(data)
        axis = next(a for a in figure.axes if a.get_label() == "cnv_data")
        manifests[letter] = save_tiff_and_preview(
            figure, output / f"S15{letter}", plot_axis=axis
        )
        plt.close(figure)
    summary = json.loads(statistics.read_text())
    figure = render_overlap(summary)
    manifests["C"] = save_tiff_and_preview(
        figure, output / "S15C", plot_axis=figure.axes[0]
    )
    plt.close(figure)
    all_gene = read_prepared_cohort(data_root).CNV_gene
    mapping = pd.read_csv(data_root / "data/clustering/best_k_cluster_labels.csv")
    result = calculate_cluster_panels(
        pd.DataFrame(all_gene), all_gene.sample_metadata, mapping,
        expected_subtypes={"LUAD": 89, "ASC": 29, "LUSC": 82},
    )
    for letter, name in (("D", "C47_subtype"), ("E", "C6_subtype"),
                         ("F", "C6_component"), ("G", None)):
        figure = (make_cluster_boxplot(result, name) if name
                  else make_c6_difference(result))
        if letter in ("F", "G"):
            figure.axes[0].xaxis.set_label_coords(0.5, -0.25)
        manifests[letter] = save_tiff_and_preview(
            figure, output / f"S15{letter}", plot_axis=figure.axes[0]
        )
        plt.close(figure)
    result.statistics.to_csv(output / "pairwise_tests.csv", index=False)
    (output / "panels.json").write_text(json.dumps(manifests, indent=2) + "\n")


def main():
    """Run the S15 analysis-topic entry without composing panels."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--statistics", type=Path, required=True)
    args = parser.parse_args()
    render(args.data_root, args.output, args.statistics)


if __name__ == "__main__":
    main()
