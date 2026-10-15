"""Compare notebook and 0.5.0 S16 inference using identical label shuffles."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter
from pymaftools.core.PairwiseMatrix import SimilarityMatrix

from asc_reproduction.cohort_inputs import read_prepared_table
from asc_reproduction.figure_style import plt, save_tiff_and_preview, setup_plotting

GROUPS = ["LUAD", "ASC", "LUSC"]


class LegacyMeans(SimilarityMatrix):
    """Preserve historical full-block averaging while sharing seeded shuffles."""

    def get_mean_group_similarity(self, groups, group_order=None):
        """Include diagonal self-pairs as the original notebook package did."""
        labels = np.asarray(groups)
        order = GROUPS if group_order is None else group_order
        result = pd.DataFrame(index=order, columns=order, dtype=float)
        for first in order:
            for second in order:
                block = self.iloc[labels == first, labels == second]
                result.loc[first, second] = block.mean().mean()
        return result


def draw_summary(values, title, destination, lower=0, pvalues=False):
    """Render a square summary through pymaftools, retaining numeric precision."""
    figure, axis = plt.subplots(figsize=(6, 6))
    SimilarityMatrix.plot_group_heatmap(
        values, title, ax=axis, tick_size=12, fontsize=12, annot_size=12,
        cmap="Reds_r" if pvalues else "Blues",
    )
    axis.set_aspect("equal", adjustable="box")
    heat = axis.collections[0]
    heat.set_clim(lower, 1)
    heat.colorbar.set_ticks(np.linspace(lower, 1, 5))
    heat.colorbar.ax.set_position([0.765, 0.2, 0.025, 0.6])
    heat.colorbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    for label in axis.texts:
        x, y = label.get_position()
        value = values.iloc[int(y), int(x)]
        if pvalues:
            label.set_text("<0.001" if 0 < value < 0.001 else f"{value:.3f}")
        else:
            label.set_text(f"{value:.2f}")
        red, green, blue, _ = heat.cmap(heat.norm(value))
        luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
        label.set_color("white" if luminance < 0.5 else "black")
    record = save_tiff_and_preview(figure, destination, plot_axis=axis)
    plt.close(figure)
    return record


def render_saved(source, output):
    """Restyle approved 0.5.0 summaries without rerunning permutation tests."""
    setup_plotting()
    output.mkdir(parents=True, exist_ok=False)
    records = {}
    for letter in "ABC":
        for kind, suffix, title in (
            ("mean", "means", "Group mean"),
            ("p", "pvalues", "Group Similarity P-values"),
        ):
            values = pd.read_csv(source / f"S16{letter}_{suffix}.csv", index_col=0)
            records[f"{letter}_{kind}"] = draw_summary(
                values, title, output / f"S16{letter}_{kind}",
                lower=-1 if letter == "C" and kind == "mean" else 0,
                pvalues=kind == "p",
            )
    (output / "panels.json").write_text(json.dumps(records, indent=2))


def compare(source, output):
    """Save both estimators, all null means and paired significance comparisons."""
    setup_plotting()
    output.mkdir(parents=True, exist_ok=False)
    metadata = read_prepared_table(Path("."), "tumor", "SNV").sample_metadata
    comparisons = []
    for version in ("legacy", "050"):
        (output / version).mkdir()
    records = {version: {} for version in ("legacy", "050")}
    for letter in "ABC":
        matrix = pd.read_csv(source / f"S16{letter}_matrix.csv", index_col=0)
        assert list(matrix.index) == list(metadata.index)
        labels = metadata.subtype
        results = {}
        for version, cls in (("legacy", LegacyMeans), ("050", SimilarityMatrix)):
            table = cls(matrix)
            means = table.get_mean_group_similarity(labels, GROUPS)
            null = table.generate_permutation_list(
                labels, GROUPS, n_permutations=1000, random_state=42,
            )
            p = table.calculate_group_similarity_pvalues(means, null, GROUPS)
            if version == "legacy":
                # Historical right tail is extreme/B, without pseudocount.
                p = ((p * 1001 - 1).round() / 1000).clip(0, 1)
            destination = output / version
            means.to_csv(destination / f"S16{letter}_means.csv")
            p.to_csv(destination / f"S16{letter}_pvalues.csv")
            np.save(destination / f"S16{letter}_null_means.npy", np.stack(null))
            for kind, values, title in (
                ("mean", means, "Group mean"),
                ("p", p, "Permutation P (right tail)"),
            ):
                records[version][f"{letter}_{kind}"] = draw_summary(
                    values, title, destination / f"S16{letter}_{kind}",
                    lower=-1 if letter == "C" and kind == "mean" else 0,
                    pvalues=kind == "p",
                )
            results[version] = (means, p)
        for i, first in enumerate(GROUPS):
            for second in GROUPS[i:]:
                old = results["legacy"][1].loc[first, second]
                new = results["050"][1].loc[first, second]
                comparisons.append({
                    "panel": letter, "first": first, "second": second,
                    "legacy_mean": results["legacy"][0].loc[first, second],
                    "mean_050": results["050"][0].loc[first, second],
                    "legacy_p": old, "p_050": new,
                    "changed_at_005": bool((old < 0.05) != (new < 0.05)),
                })
        print(f"Finished S16{letter}", flush=True)
    pd.DataFrame(comparisons).to_csv(output / "comparison.csv", index=False)
    for version, items in records.items():
        (output / version / "panels.json").write_text(json.dumps(items, indent=2))
    (output / "settings.json").write_text(json.dumps({
        "seed": 42, "permutations": 1000, "tail": "right",
        "identical_shuffles": True,
        "historical_random_draws_reproduced": False,
        "note": "Original unseeded draws unavailable; matched-seed method comparison.",
    }, indent=2))


def main():
    """Run the user-authorized two-method comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    if args.render_only:
        render_saved(args.source, args.output)
    else:
        compare(args.source, args.output)


if __name__ == "__main__":
    main()
