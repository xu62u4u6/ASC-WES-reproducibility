"""S13: ASC T-sample mutation classes grouped by smoking status.

Reproduces legacy input-first-50 selection, not an ASC frequency top-50.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap, to_rgba
from matplotlib.patches import Patch

from .cgc_oncoplot import GROUPS, CGCData, group_mutation_order, mutation_presence
from .dataio import read_pivot_sqlite
from .figure_style import MUTATION_COLORS, TRACK_COLORS, plt, save_tiff_and_preview


def prepare_smoking_oncoplot(values, metadata, *, expected_n=29):
    if (
        values.index.has_duplicates
        or values.columns.has_duplicates
        or metadata.index.has_duplicates
    ):
        raise ValueError("Duplicate mutation keys")
    if set(values.columns) != set(metadata.index):
        raise ValueError("Mutation metadata keys differ")
    metadata = metadata.loc[values.columns]
    if not metadata.subtype.isin(GROUPS).all():
        raise ValueError("Unknown subtype")
    binary = mutation_presence(values)
    initial = group_mutation_order(binary, metadata)
    selected = metadata.loc[initial]
    selected = selected.loc[selected.subtype.eq("ASC") & selected.sample_type.eq("T")]
    if expected_n is not None and len(selected) != expected_n:
        raise ValueError("Unexpected ASC T-sample count")
    if not selected.smoke.isin([0, 1]).all():
        raise ValueError("Invalid smoking annotation")
    if not np.isfinite(selected.TMB.to_numpy(dtype=float)).all():
        raise ValueError("Invalid TMB")
    if any(not selected.smoke.eq(group).any() for group in [1, 0]):
        raise ValueError("Empty smoking group")
    # The legacy function adds frequencies without sorting features, then head(50).
    genes = values.index[:50]
    if not len(genes):
        raise ValueError("Empty gene matrix")
    top = binary.loc[:, selected.index].head(10).astype(int)
    weights = np.power(2, np.arange(len(top) - 1, -1, -1)) @ top.to_numpy()
    scores = pd.Series(weights, index=top.columns)
    order = []
    for group in [1, 0]:
        members = selected.index[selected.smoke.eq(group)]
        order.extend(scores.loc[members].sort_values(ascending=False).index)
    selected = selected.loc[order]
    classes = values.loc[genes, order].replace({False: "WT"})
    if set(classes.to_numpy().ravel()) - set(MUTATION_COLORS):
        raise ValueError("Unsupported mutation class")
    summary = binary.loc[genes, order].mean(axis=1).to_frame("freq")
    for group, name in [(1, "smoker"), (0, "non-smoker")]:
        members = selected.index[selected.smoke.eq(group)]
        summary[name + "_freq"] = binary.loc[genes, members].mean(axis=1)
        summary[name + "_count"] = binary.loc[genes, members].sum(axis=1)
    summary.index.name = "gene"
    return CGCData(classes, selected, summary)


def plot_smoking_oncoplot(data):
    """Render S13 through the shared pymaftools 0.5.0 adapter."""
    from .oncoplot_rendering import render_frequency_oncoplot

    return render_frequency_oncoplot(
        data,
        smoking=True,
        preserve_frequency_names=True,
        show_sample_labels=True,
        legend_fontsize=14,
        legend_title_fontsize=16,
    )


def plot_smoking_oncoplot_legacy(data):
    fig = plt.figure(figsize=(15, 17))
    grid = fig.add_gridspec(
        3,
        3,
        width_ratios=[20, 5, 5],
        height_ratios=[2, 30, 1],
        wspace=0.05,
        hspace=0.025,
    )
    n = len(data.metadata)
    bar = fig.add_subplot(grid[0, 0])
    bar.bar(np.arange(n) + 0.5, data.metadata.TMB, width=0.95, color="gray")
    bar.set(xlim=(0, n), xticks=[], ylabel="TMB")
    bar.spines[["top", "right", "bottom"]].set_visible(False)
    ax = fig.add_subplot(grid[1, 0])
    keys = list(MUTATION_COLORS)
    codes = data.classes.apply(lambda c: c.map({v: i for i, v in enumerate(keys)}))
    sns.heatmap(
        codes,
        ax=ax,
        cmap=ListedColormap(list(MUTATION_COLORS.values())),
        vmin=-0.5,
        vmax=len(keys) - 0.5,
        cbar=False,
        linewidths=0.5,
        linecolor="white",
        xticklabels=False,
        yticklabels=data.classes.index,
    )
    ax.set(xlabel="", ylabel="")
    ax.tick_params(axis="y", rotation=0, labelsize=10, length=0)
    freq = fig.add_subplot(grid[1, 1])
    sns.heatmap(
        data.gene_summary[["freq", "smoker_freq", "non-smoker_freq"]],
        ax=freq,
        cmap="Blues",
        vmin=0,
        vmax=1,
        cbar=False,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 8},
        linewidths=0.5,
        yticklabels=False,
        xticklabels=["All ASC", "Smoker", "Non-smoker"],
    )
    freq.set(xlabel="", ylabel="", title="Frequency")
    freq.tick_params(axis="x", rotation=90, labelsize=10)
    track = fig.add_subplot(grid[2, 0])
    palette = TRACK_COLORS["smoke"]
    sns.heatmap(
        data.metadata.smoke.to_numpy(dtype=int)[None, :],
        ax=track,
        cmap=ListedColormap([to_rgba(palette[i], 0.7) for i in [0, 1]]),
        vmin=-0.5,
        vmax=1.5,
        cbar=False,
        linewidths=0.5,
        xticklabels=False,
        yticklabels=False,
    )
    track.set_ylabel("Smoking", rotation=0, ha="right", va="center")
    track.set_xlabel("ASC T samples (identifiers omitted)")
    legend = fig.add_subplot(grid[1:, 2])
    legend.axis("off")
    observed = set(data.classes.to_numpy().ravel()) - {"WT"}
    handles = [
        Patch(facecolor=color, label=name)
        for name, color in MUTATION_COLORS.items()
        if name in observed
    ]
    handles += [
        Patch(
            facecolor=palette[i],
            alpha=0.7,
            label=f"{name} (n={int(data.metadata.smoke.eq(i).sum())})",
        )
        for i, name in [(1, "Smoker"), (0, "Non-smoker")]
    ]
    legend.legend(handles=handles, loc="upper left", frameon=False, fontsize=9)
    return fig


def run(data_root: Path, output: Path) -> None:
    """Generate the existing S13 candidate grouped by smoking status."""
    from .figure_style import setup_plotting

    setup_plotting()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output must be new or empty")
    pivot = read_pivot_sqlite(data_root / "data/WES/all_case_table.db")
    data = prepare_smoking_oncoplot(pivot.values, pivot.sample_metadata)
    if len(data.classes) != 50:
        raise ValueError("Expected 50 input-order genes")
    output.mkdir(parents=True, exist_ok=True)
    fig = plot_smoking_oncoplot(data)
    info = save_tiff_and_preview(fig, output / "S13_ASC_smoking_oncoplot")
    plt.close(fig)
    data.gene_summary.to_csv(output / "S13_gene_summary.csv")
    summary = {
        "panel": "S13",
        "sample_type": "T",
        "subtype": "ASC",
        "genes": 50,
        "group_counts": {
            name: int(data.metadata.smoke.eq(i).sum())
            for i, name in [(1, "smoker"), (0, "non-smoker")]
        },
        "selection": "First 50 input-order genes; no ASC-specific frequency sorting/filter",
        "ordering": "Initial cohort subtype/top10 pattern, then smoke 1/0 and top10 pattern",
        "figure": info,
        "annotation_alpha": 0.7,
        "limitations": "Legacy descriptive panel; not ASC top-frequency selection. Processed TMB provenance pending.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main():
    """Read CLI options and call the reusable analysis function."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.data_root, args.output)


if __name__ == "__main__":
    main()
