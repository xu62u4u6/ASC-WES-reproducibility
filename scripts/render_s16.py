"""Audit S16 group means and render package-backed descriptive panels."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pymaftools.core.PairwiseMatrix import SimilarityMatrix

from asc_reproduction.cgc_oncoplot import mutation_presence
from asc_reproduction.cohort_inputs import read_prepared_table
from asc_reproduction.figure_style import plt, save_tiff_and_preview, setup_plotting
from asc_reproduction.similarity_panels import (
    SimilarityPanel,
    draw_similarity_panel,
    sample_similarity,
)

GROUPS = ["LUAD", "ASC", "LUSC"]


def render(data_root: Path, output: Path) -> None:
    """Export unchanged matrices and old/new means; do not infer new p-values."""
    setup_plotting()
    output.mkdir(parents=True, exist_ok=False)
    snv = read_prepared_table(data_root, "tumor", "SNV")
    cnv = read_prepared_table(data_root, "tumor", "CNV_cluster")
    assert snv.columns.equals(cnv.columns)
    metadata = snv.sample_metadata
    assert metadata.subtype.value_counts().to_dict() == {
        "LUAD": 89, "ASC": 29, "LUSC": 82
    }
    binary = mutation_presence(pd.DataFrame(snv))
    binary = binary.loc[binary.mean(axis=1).ge(0.05)]
    comparisons = []
    images = {}
    specifications = [
        ("A", binary, "jaccard", "SNV Jaccard"),
        ("B", binary, "hamming", "SNV SMC"),
        ("C", pd.DataFrame(cnv), "cosine", "CNV cluster cosine"),
    ]
    for letter, values, metric, title in specifications:
        matrix = sample_similarity(values, metric)
        lower = -1 if metric == "cosine" else 0
        assert matrix.shape == (200, 200)
        assert matrix.min() >= lower - 1e-12 and matrix.max() <= 1 + 1e-12
        np.testing.assert_allclose(matrix, matrix.T, atol=1e-12)
        table = SimilarityMatrix(matrix, index=values.columns, columns=values.columns)
        table.to_csv(output / f"S16{letter}_matrix.csv")
        means = table.get_mean_group_similarity(metadata.subtype, GROUPS)
        means.to_csv(output / f"S16{letter}_means_050.csv")
        for first in GROUPS:
            for second in GROUPS:
                block = table.loc[metadata.subtype.eq(first), metadata.subtype.eq(second)]
                old = float(block.to_numpy().mean())
                new = float(means.loc[first, second])
                if first != second:
                    np.testing.assert_allclose(old, new, atol=1e-14)
                comparisons.append({
                    "metric": metric, "first": first, "second": second,
                    "legacy_mean": old, "mean_050": new, "difference": new - old,
                })
        panel = SimilarityPanel(matrix, metadata, metric, len(values), False, title)
        figure = plt.figure(figsize=(8, 8.1))
        draw_similarity_panel(figure, figure.add_gridspec(1, 1)[0], panel)
        # Keep S7 defaults unchanged; this S16 candidate uses approved metadata alpha.
        for axis in figure.axes:
            for patch in axis.patches:
                patch.set_alpha(0.7)
        images[f"{letter}_matrix"] = save_tiff_and_preview(
            figure, output / f"S16{letter}_matrix", plot_axis=figure.axes[0]
        )
        plt.close(figure)
        figure, axis = plt.subplots(figsize=(6, 6))
        SimilarityMatrix.plot_group_heatmap(
            means, "Group mean (0.5.0; candidate)", ax=axis,
            tick_size=12, fontsize=12, annot_size=12,
        )
        heat = axis.collections[0]
        axis.set_aspect("equal", adjustable="box")
        heat.set_clim(lower, 1)
        heat.colorbar.set_ticks(np.linspace(lower, 1, 5))
        for text in axis.texts:
            value = float(text.get_text())
            text.set_color("white" if (value - lower) / (1 - lower) > 0.55 else "black")
        images[f"{letter}_mean"] = save_tiff_and_preview(
            figure, output / f"S16{letter}_mean", plot_axis=axis
        )
        plt.close(figure)
    pd.DataFrame(comparisons).to_csv(output / "mean_definition_comparison.csv", index=False)
    (output / "panels.json").write_text(json.dumps(images, indent=2) + "\n")


def main():
    """Run the descriptive audit while inferential method approval is pending."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.data_root, args.output)


if __name__ == "__main__":
    main()
