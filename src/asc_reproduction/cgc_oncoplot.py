"""Figure 4A: CGC Tier 1 mutations in the 200 tumor samples."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap, to_rgba
from matplotlib.patches import Rectangle

from .dataio import read_pivot_sqlite
from .figure_style import MUTATION_COLORS, TRACK_COLORS, plt, save_tiff_and_preview

GROUPS = ("LUAD", "ASC", "LUSC")
EXPECTED_COUNTS = {"LUAD": 89, "ASC": 29, "LUSC": 82}
ROLE_COLORS = {
    "Oncogene": "firebrick",
    "TSG": "steelblue",
    "Oncogene/TSG": "mediumpurple",
    "Fusion": "darkorange",
    "Other": "lightgray",
}


@dataclass(frozen=True)
class CGCData:
    # Sample identifiers stay in memory; only aggregate tables are exported.
    classes: pd.DataFrame
    metadata: pd.DataFrame
    gene_summary: pd.DataFrame


def mutation_presence(values: pd.DataFrame) -> pd.DataFrame:
    if values.isna().to_numpy().any():
        raise ValueError("Mutation matrix contains missing values")
    return values.ne("WT") & values.ne(False)


def group_mutation_order(binary: pd.DataFrame, metadata: pd.DataFrame) -> list:
    """Legacy ordering: subtype, then descending first-ten-row binary code.

    Pandas' default quicksort preserves the legacy tie behavior. The original
    matrix order and pinned pandas version therefore form part of the input.
    """
    top = binary.head(10).astype(int)
    weights = np.power(2, np.arange(len(top) - 1, -1, -1)) @ top.to_numpy()
    scores = pd.Series(weights, index=top.columns)
    ordered = []
    for group in GROUPS:
        members = metadata.index[metadata["subtype"].eq(group)]
        ordered.extend(scores.loc[members].sort_values(ascending=False).index)
    return ordered


def role_category(value: object) -> str:
    role = str(value).lower()
    if "oncogene" in role and "tsg" in role:
        return "Oncogene/TSG"
    if "oncogene" in role:
        return "Oncogene"
    if "tsg" in role:
        return "TSG"
    return "Fusion" if "fusion" in role else "Other"


def prepare_cgc_data(
    values: pd.DataFrame,
    metadata: pd.DataFrame,
    census: pd.DataFrame,
    *,
    expected_counts: dict | None = EXPECTED_COUNTS,
) -> CGCData:
    required = {"sample_type", "subtype", "TMB", "sex", "smoke"}
    if not required.issubset(metadata.columns):
        raise ValueError("Required tumor sample metadata is missing")
    if (
        values.index.has_duplicates
        or values.columns.has_duplicates
        or metadata.index.has_duplicates
    ):
        raise ValueError("Mutation matrix or metadata contains duplicate keys")
    if set(values.columns) != set(metadata.index):
        raise ValueError("Mutation matrix and sample metadata keys differ")
    if census.index.has_duplicates or not {"Tier", "Role in Cancer"}.issubset(
        census.columns
    ):
        raise ValueError(
            "Census requires unique gene symbols, Tier, and Role in Cancer"
        )
    metadata = metadata.loc[values.columns]
    if not metadata["subtype"].isin(GROUPS).all():
        raise ValueError("Unexpected subtype in mutation input")
    binary = mutation_presence(values)
    # ASCDataLoader first orders the full cohort; the panel then orders T samples.
    initial = group_mutation_order(binary, metadata)
    selected = metadata.loc[initial]
    selected = selected.loc[selected["sample_type"].eq("T")].copy()
    counts = selected["subtype"].value_counts().reindex(GROUPS, fill_value=0).to_dict()
    if expected_counts is not None and counts != expected_counts:
        raise ValueError(f"Unexpected T-sample counts: {counts}")
    if any(count == 0 for count in counts.values()):
        raise ValueError("Each subtype must have tumor samples")
    if not np.isfinite(selected["TMB"].to_numpy(dtype=float)).all():
        raise ValueError("TMB contains non-finite values")
    for track, colors in TRACK_COLORS.items():
        if not selected[track].isin(colors).all():
            raise ValueError(f"Unexpected or missing values in {track} track")
    freq = binary.loc[:, selected.index].mean(axis=1)
    tier1 = census.index[pd.to_numeric(census["Tier"]).eq(1)]
    keep = freq.loc[freq.index.isin(tier1) & freq.gt(0.05)]
    genes = keep.to_frame("freq").sort_values("freq", ascending=False).index
    if len(genes) == 0:
        raise ValueError("No Tier 1 genes pass the strict >5% cutoff")
    order = group_mutation_order(binary.loc[genes, selected.index], selected)
    selected = selected.loc[order]
    classes = values.loc[genes, order].replace({False: "WT"})
    unknown = set(classes.to_numpy().ravel()) - set(MUTATION_COLORS)
    if unknown:
        raise ValueError(
            f"Selected genes contain unsupported mutation classes: {sorted(unknown)}"
        )
    summary = pd.DataFrame({"freq": freq.loc[genes]})
    for group in GROUPS:
        members = selected.index[selected["subtype"].eq(group)]
        summary[f"{group}_count"] = binary.loc[genes, members].sum(axis=1)
        summary[f"{group}_freq"] = binary.loc[genes, members].mean(axis=1)
    summary["CGC_role"] = census.loc[genes, "Role in Cancer"].map(role_category)
    summary.index.name = "gene"
    return CGCData(classes, selected, summary)


def make_cgc_figure(data: CGCData) -> plt.Figure:
    """Render Figure 4A with pinned pymaftools; keep validated input ordering."""
    from .oncoplot_rendering import render_cgc_oncoplot

    return render_cgc_oncoplot(data)


def make_cgc_reference_figure(data: CGCData) -> plt.Figure:
    """Render the pre-migration image for comparison, never the formal CLI."""
    fig = plt.figure(figsize=(30, 16))
    gs = fig.add_gridspec(
        5,
        4,
        width_ratios=[25, 3, 1.2, 3.5],
        height_ratios=[1, 20, 1, 1, 1],
        wspace=0.02,
        hspace=0.025,
    )
    n = len(data.metadata)
    bar = fig.add_subplot(gs[0, 0])
    bar.bar(
        np.arange(n) + 0.5,
        data.metadata["TMB"],
        width=0.95,
        color="gray",
        edgecolor="white",
        linewidth=0.25,
    )
    bar.set(xlim=(0, n), xticks=[], ylabel="TMB")
    bar.spines[["top", "right", "bottom"]].set_visible(False)
    ax = fig.add_subplot(gs[1, 0])
    codes = data.classes.apply(
        lambda col: col.map({key: i for i, key in enumerate(MUTATION_COLORS)})
    )
    sns.heatmap(
        codes,
        ax=ax,
        cmap=ListedColormap(list(MUTATION_COLORS.values())),
        vmin=-0.5,
        vmax=len(MUTATION_COLORS) - 0.5,
        cbar=False,
        linewidths=0.5,
        linecolor="white",
        xticklabels=False,
        yticklabels=data.classes.index,
    )
    ax.set(xlabel="", ylabel="")
    ax.tick_params(axis="y", labelsize=15, rotation=0, length=0)
    freq = fig.add_subplot(gs[1, 1])
    sns.heatmap(
        data.gene_summary[[f"{g}_freq" for g in GROUPS]],
        ax=freq,
        cmap="Blues",
        cbar=False,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 11},
        linewidths=0.5,
        yticklabels=False,
        xticklabels=GROUPS,
    )
    freq.set(ylabel="", xlabel="")
    freq.set_title("Frequency", fontsize=13)
    freq.tick_params(axis="x", labelsize=12, rotation=90, length=0)
    roles = list(ROLE_COLORS)
    role = fig.add_subplot(gs[1, 2])
    role.imshow(
        np.array([roles.index(v) for v in data.gene_summary["CGC_role"]])[:, None],
        cmap=ListedColormap(list(ROLE_COLORS.values())),
        aspect="auto",
        vmin=-0.5,
        vmax=len(roles) - 0.5,
        interpolation="nearest",
    )
    role.set(xticks=[], yticks=[], title="Role")
    for spine in role.spines.values():
        spine.set_visible(False)
    for row, (track, colors) in enumerate(TRACK_COLORS.items(), 2):
        track_ax = fig.add_subplot(gs[row, 0])
        keys = list(colors)
        encoded = np.array([keys.index(v) for v in data.metadata[track]])[None, :]
        sns.heatmap(
            encoded,
            ax=track_ax,
            cmap=ListedColormap([to_rgba(c, 0.7) for c in colors.values()]),
            vmin=-0.5,
            vmax=len(keys) - 0.5,
            cbar=False,
            linewidths=0.5,
            xticklabels=False,
            yticklabels=False,
        )
        track_ax.set_ylabel(track, rotation=0, ha="right", va="center", fontsize=13)
    legend = fig.add_subplot(gs[1:, 3])
    legend.axis("off")
    observed = set(data.classes.to_numpy().ravel()) - {"WT"}
    blocks = [
        ("Mutation", {k: v for k, v in MUTATION_COLORS.items() if k in observed}, 1)
    ]
    blocks += [(k, v, 0.7) for k, v in TRACK_COLORS.items()]
    blocks += [
        (
            "CGC Role",
            {
                k: v
                for k, v in ROLE_COLORS.items()
                if k in set(data.gene_summary["CGC_role"])
            },
            1,
        )
    ]
    y = 0.98
    for title, colors, alpha in blocks:
        legend.text(0, y, title, fontsize=14, weight="bold", va="top")
        y -= 0.037
        for label, color in colors.items():
            legend.add_patch(
                Rectangle(
                    (0, y - 0.019),
                    0.075,
                    0.019,
                    facecolor=color,
                    alpha=alpha,
                    edgecolor="none",
                )
            )
            legend.text(0.10, y, str(label), fontsize=11, va="top")
            y -= 0.027
        y -= 0.018
    return fig


def run(data_root: Path, output: Path) -> None:
    """Generate Figure 4A CGC Tier 1 oncoplot for 200 tumor samples."""
    from .figure_style import setup_plotting

    setup_plotting()
    import pymaftools

    from .provenance import file_hash, runtime_identity

    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be new or empty")
    input_paths = {
        "mutation_table": data_root / "data/WES/all_case_table.db",
        "census": data_root / "data/Census All Months Jan 27 2025.csv",
    }
    input_hashes = {name: file_hash(path) for name, path in input_paths.items()}
    runtime = runtime_identity()
    pivot = read_pivot_sqlite(input_paths["mutation_table"])
    census = pd.read_csv(input_paths["census"], index_col="Gene Symbol")
    data = prepare_cgc_data(pivot.values, pivot.sample_metadata, census)
    if len(data.classes) != 24:
        raise ValueError(f"Expected 24 selected CGC genes; found {len(data.classes)}")
    output.mkdir(parents=True, exist_ok=True)
    fig = make_cgc_figure(data)
    image_info = save_tiff_and_preview(fig, output / "Figure_4A_CGC_oncoplot")
    plt.close(fig)
    data.gene_summary.to_csv(output / "Figure_4A_gene_summary.csv")
    summary = {
        "panel": "Figure 4A",
        "sample_type": "T",
        "group_counts": EXPECTED_COUNTS,
        "genes": len(data.classes),
        "selection": "CGC Tier 1 and T-cohort mutation frequency > 0.05",
        "ordering": "Legacy full-cohort top-10 mutation-pattern sort, then T-only CGC top-10 sort within subtype",
        "tracks": ["TMB", "subtype", "sex", "smoke", "CGC role"],
        "annotation_alpha": 0.7,
        "figure": image_info,
        "renderer": {
            "api": "pymaftools.OncoPlot",
            "version": pymaftools.__version__,
            "module_path": pymaftools.__file__,
        },
        "input_sha256": input_hashes,
        "runtime": runtime,
        "limitations": "Processed TMB and supplied CGC annotation; input provenance and redistribution rights remain under review.",
    }
    if (
        input_hashes != {name: file_hash(path) for name, path in input_paths.items()}
        or runtime != runtime_identity()
    ):
        raise RuntimeError("Figure inputs or runtime changed during rendering")
    summary["artifacts"] = {
        path.name: file_hash(path)
        for path in sorted(output.iterdir())
        if path.is_file()
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main(argv: list[str] | None = None) -> None:
    """Read CLI options and call the reusable analysis function."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    run(args.data_root, args.output)


if __name__ == "__main__":
    main()
