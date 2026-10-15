"""S11: all genes mutated in at least 10% of the 200 T samples."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap, to_rgba
from matplotlib.patches import Patch

from .cgc_oncoplot import (
    EXPECTED_COUNTS,
    GROUPS,
    CGCData,
    group_mutation_order,
    mutation_presence,
)
from .dataio import read_pivot_sqlite
from .figure_style import MUTATION_COLORS, TRACK_COLORS, plt, save_tiff_and_preview


def prepare_subtype_oncoplot(values, metadata, *, expected_counts=EXPECTED_COUNTS):
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
    tumors = metadata.loc[initial]
    tumors = tumors.loc[tumors.sample_type.eq("T")]
    counts = tumors.subtype.value_counts().reindex(GROUPS, fill_value=0).to_dict()
    if expected_counts is not None and counts != expected_counts:
        raise ValueError("Unexpected subtype counts")
    if any(v == 0 for v in counts.values()):
        raise ValueError("Empty subtype")
    frequency = binary.loc[:, tumors.index].mean(axis=1)
    genes = frequency.loc[frequency.ge(0.1)].sort_values(ascending=False).index
    if not len(genes):
        raise ValueError("No genes at inclusive 10% threshold")
    order = group_mutation_order(binary.loc[genes, tumors.index], tumors)
    tumors = tumors.loc[order]
    if not np.isfinite(tumors.TMB.to_numpy(dtype=float)).all():
        raise ValueError("Invalid TMB")
    for name, colors in TRACK_COLORS.items():
        if not tumors[name].isin(colors).all():
            raise ValueError("Invalid sample annotation")
    classes = values.loc[genes, order].replace({False: "WT"})
    if set(classes.to_numpy().ravel()) - set(MUTATION_COLORS):
        raise ValueError("Unsupported mutation class")
    summary = frequency.loc[genes].to_frame("freq")
    for group in GROUPS:
        selected = tumors.index[tumors.subtype.eq(group)]
        summary[group + "_freq"] = binary.loc[genes, selected].mean(axis=1)
        summary[group + "_count"] = binary.loc[genes, selected].sum(axis=1)
    summary.index.name = "gene"
    return CGCData(classes, tumors, summary)


def plot_subtype_oncoplot(data):
    """Render S11 through the shared pymaftools 0.5.0 adapter."""
    from .oncoplot_rendering import render_frequency_oncoplot

    return render_frequency_oncoplot(
        data,
        cell_linewidth=0.02,
        annotation_linewidth=0.02,
        legend_width=6.0,
        legend_fontsize=14.0,
        legend_title_fontsize=16.0,
        figsize=(30, 18),
        main_width=40.0,
        preserve_frequency_names=True,
    )


def plot_subtype_oncoplot_legacy(data):
    fig = plt.figure(figsize=(30, 16))
    grid = fig.add_gridspec(
        5,
        3,
        width_ratios=[25, 3, 3],
        height_ratios=[1, 20, 1, 1, 1],
        wspace=0.03,
        hspace=0.025,
    )
    n = len(data.metadata)
    bar = fig.add_subplot(grid[0, 0])
    bar.bar(np.arange(n) + 0.5, data.metadata.TMB, width=0.95, color="gray")
    bar.set(xlim=(0, n), xticks=[], ylabel="TMB")
    bar.spines[["top", "right", "bottom"]].set_visible(False)
    ax = fig.add_subplot(grid[1, 0])
    colors = list(MUTATION_COLORS)
    codes = data.classes.apply(lambda c: c.map({v: i for i, v in enumerate(colors)}))
    sns.heatmap(
        codes,
        ax=ax,
        cmap=ListedColormap(list(MUTATION_COLORS.values())),
        vmin=-0.5,
        vmax=len(colors) - 0.5,
        cbar=False,
        linewidths=0.5,
        linecolor="white",
        xticklabels=False,
        yticklabels=data.classes.index,
    )
    ax.set(xlabel="", ylabel="")
    ax.tick_params(axis="y", rotation=0, labelsize=14, length=0)
    freq = fig.add_subplot(grid[1, 1])
    sns.heatmap(
        data.gene_summary[[g + "_freq" for g in GROUPS]],
        ax=freq,
        cmap="Blues",
        cbar=False,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 10},
        linewidths=0.5,
        yticklabels=False,
        xticklabels=GROUPS,
    )
    freq.set(xlabel="", ylabel="", title="Frequency")
    freq.tick_params(axis="x", rotation=90, labelsize=12)
    for row, (track, palette) in enumerate(TRACK_COLORS.items(), 2):
        axis = fig.add_subplot(grid[row, 0])
        keys = list(palette)
        encoded = np.array([keys.index(v) for v in data.metadata[track]])[None, :]
        sns.heatmap(
            encoded,
            ax=axis,
            cmap=ListedColormap([to_rgba(v, 0.7) for v in palette.values()]),
            vmin=-0.5,
            vmax=len(keys) - 0.5,
            cbar=False,
            linewidths=0.5,
            xticklabels=False,
            yticklabels=False,
        )
        axis.set_ylabel(track, rotation=0, ha="right", va="center", fontsize=13)
    legend = fig.add_subplot(grid[1:, 2])
    legend.axis("off")
    observed = set(data.classes.to_numpy().ravel()) - {"WT"}
    handles = [
        Patch(facecolor=color, label=name)
        for name, color in MUTATION_COLORS.items()
        if name in observed
    ]
    handles.extend(
        Patch(facecolor=color, alpha=0.7, label=f"{track}: {name}")
        for track, palette in TRACK_COLORS.items()
        for name, color in palette.items()
    )
    legend.legend(handles=handles, loc="upper left", frameon=False, fontsize=11)
    return fig


def run(data_root: Path, output: Path) -> None:
    """Generate S11 using the existing >=10% mutation-frequency definition."""
    from .figure_style import setup_plotting

    setup_plotting()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output must be new or empty")
    pivot = read_pivot_sqlite(data_root / "data/WES/all_case_table.db")
    data = prepare_subtype_oncoplot(pivot.values, pivot.sample_metadata)
    if len(data.classes) != 39:
        raise ValueError("Expected 39 genes at inclusive 10%")
    output.mkdir(parents=True, exist_ok=True)
    fig = plot_subtype_oncoplot(data)
    info = save_tiff_and_preview(fig, output / "S11_subtype_oncoplot_ge10")
    plt.close(fig)
    data.gene_summary.to_csv(output / "S11_gene_summary.csv")
    summary = {
        "panel": "S11",
        "sample_type": "T",
        "group_counts": EXPECTED_COUNTS,
        "genes": len(data.classes),
        "frequency_filter": ">=0.10 within all T samples",
        "figure": info,
        "annotation_alpha": 0.7,
        "cell_linewidth": 0.02,
        "annotation_linewidth": 0.02,
        "legend_width": 6.0,
        "legend_fontsize": 14.0,
        "legend_title_fontsize": 16.0,
        "figsize": [30, 18],
        "main_width": 40.0,
        "frequency_labels": ["LUAD_freq", "ASC_freq", "LUSC_freq"],
        "ordering": "legacy full-cohort top10 pattern; frequency-sorted selected genes; subtype top10 pattern",
        "limitations": "Processed TMB provenance remains pending; manuscript >10% wording requires correction.",
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
