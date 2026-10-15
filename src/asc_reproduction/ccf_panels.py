"""Figure 2B heatmaps from supplied PyClone-VI results (not a PyClone refit)."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns

from .c47_profile import _as_sample_mask
from .figure_style import plt, save_tiff_and_preview


def prepare_ccf(frame, sample_names):
    required = {"mutation_id", "sample_id", "cluster_id", "cellular_prevalence"}
    if not required.issubset(frame) or frame.empty:
        raise ValueError("Missing PyClone fields or rows")
    if frame[list(required)].isna().any().any():
        raise ValueError("Missing PyClone values")
    if frame.duplicated(["mutation_id", "sample_id"]).any():
        raise ValueError("Duplicate mutation/sample")
    if set(frame.sample_id) != set(sample_names):
        raise ValueError("Unexpected component sample keys")
    values = frame.cellular_prevalence.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("CCF outside 0–1")
    if frame.groupby("mutation_id").cluster_id.nunique().ne(1).any():
        raise ValueError("Inconsistent mutation cluster")
    pivot = frame.pivot(
        index="mutation_id", columns="sample_id", values="cellular_prevalence"
    )
    if pivot.isna().any().any():
        raise ValueError("Incomplete mutation/component matrix")
    meta = pd.DataFrame(
        {
            "mean_ccf": frame.groupby("mutation_id").cellular_prevalence.mean(),
            "cluster": frame.groupby("mutation_id").cluster_id.first(),
        }
    )
    order = meta.groupby("cluster").mean_ccf.mean().sort_values(ascending=False).index
    fm = (
        meta.set_index("cluster", append=True)
        .reorder_levels(["cluster", "mutation_id"])
        .sort_values(
            by=["cluster", "mean_ccf"],
            key=lambda c: (
                pd.Categorical(c, categories=order, ordered=True)
                if c.name == "cluster"
                else c
            ),
            ascending=[True, False],
        )
        .reset_index()
    )
    # Legacy pandas pivot sorts sample columns lexically (A, S, T), not A/T/S.
    matrix = pivot.loc[fm.mutation_id]
    return matrix, fm


def plot_ccf(panels):
    fig = plt.figure(figsize=(20, 10))
    outer = fig.add_gridspec(2, 6, wspace=0.16, hspace=0.72)
    for i, spec in enumerate(outer):
        if i >= len(panels):
            fig.add_subplot(spec).axis("off")
            continue
        matrix, meta, number = panels[i]
        inner = spec.subgridspec(1, 2, width_ratios=[0.15, 1], wspace=0.08)
        a = fig.add_subplot(inner[0])
        b = fig.add_subplot(inner[1])
        sns.heatmap(
            meta[["mean_ccf"]],
            ax=a,
            cmap="Reds",
            vmin=0,
            vmax=1,
            cbar=False,
            yticklabels=False,
            xticklabels=["mean_ccf"],
        )
        sns.heatmap(
            matrix,
            ax=b,
            cmap="Blues",
            vmin=0,
            vmax=1,
            cbar=False,
            yticklabels=False,
            xticklabels=matrix.columns.astype(str).tolist(),
        )
        a.set(xlabel="", ylabel="")
        a.tick_params(axis="x", rotation=45, pad=3.5)
        for label in a.get_xticklabels():
            label.set_horizontalalignment("right")
        b.set(title=str(number), xlabel="", ylabel="")
        b.tick_params(axis="x", rotation=45)
        for label in b.get_xticklabels():
            label.set_horizontalalignment("right")
    # The original manuscript provides both continuous scale keys.
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    fig.subplots_adjust(right=0.88)
    for x, cmap, label in (
        (0.90, "Reds", "Mean CCF"),
        (0.95, "Blues", "Cellular prevalence"),
    ):
        axis = fig.add_axes([x, 0.12, 0.012, 0.76])
        colorbar = fig.colorbar(
            ScalarMappable(norm=Normalize(0, 1), cmap=cmap), cax=axis
        )
        colorbar.set_label(label)
        colorbar.outline.set_visible(False)
    return fig


def main():
    from .figure_style import setup_plotting

    setup_plotting()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Output must be new or empty")
    cases = pd.read_csv(args.data_root / "data/ASC_all_case_metadata_0312.csv")
    if cases.case_ID.duplicated().any():
        raise ValueError("Duplicate case keys")
    selected = cases.loc[_as_sample_mask(cases.AS_sample), "case_ID"].tolist()
    if len(selected) != 11 or pd.isna(selected).any():
        raise ValueError("Expected 11 cases")
    numbers = {c: i + 1 for i, c in enumerate(sorted(selected))}
    panels = []
    counts = []
    for case in selected:
        if Path(str(case)).name != str(case):
            raise ValueError("Unsafe case filename")
        d = pd.read_csv(
            args.data_root / "data/pyclone_output" / f"{case}.pyclone.tsv", sep="\t"
        )
        matrix, meta = prepare_ccf(d, [case + "_" + part for part in ["A", "T", "S"]])
        panels.append((matrix, meta, case))
        counts.append(
            {"mutations": len(matrix), "clusters": int(meta.cluster.nunique())}
        )
    args.output.mkdir(parents=True, exist_ok=True)
    fig = plot_ccf(panels)
    info = save_tiff_and_preview(fig, args.output / "Figure2B_CCF")
    plt.close(fig)
    summary = {
        "panel": "Figure2B",
        "cases": 11,
        "case_mutations_total": sum(v["mutations"] for v in counts),
        "mutations_per_case_range": [
            min(v["mutations"] for v in counts),
            max(v["mutations"] for v in counts),
        ],
        "clusters_per_case_range": [
            min(v["clusters"] for v in counts),
            max(v["clusters"] for v in counts),
        ],
        "figure": info,
        "scale": [0, 1],
        "column_order": "Legacy lexical sample order A/S/T",
        "limitations": "Uses supplied PyClone results, not a refit. Provenance/version linkage still under audit. Original manuscript case/sample labels shown; no per-mutation values exported.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
