"""S19A/D: complete development-CV performance panels, never partial rankings."""

import argparse
import textwrap
from pathlib import Path

import numpy as np

from .figure_style import plt, save_tiff_and_preview
from .model_runner import model_lock
from .model_summary import collect_complete_cv, snapshot_identity
from .model_tables import BASELINE_STACK, build_tables
from .provenance import file_hash, runtime_identity, write_json

METRICS = [
    ("acc", "Accuracy", "#4C78A8"),
    ("f1", "Macro-F1", "#F58518"),
    ("auc", "AUROC", "#54A24B"),
]


def plot_performance(table, title, *, ax=None):
    if len(table) == 0 or table.model.duplicated().any():
        raise ValueError("Expected unique nonempty model rows")
    means = table[[m + "_mean" for m, _, _ in METRICS]].to_numpy(dtype=float)
    sd = table[[m + "_sd" for m, _, _ in METRICS]].to_numpy(dtype=float)
    if (
        not np.isfinite(means).all()
        or not np.isfinite(sd).all()
        or (means < 0).any()
        or (means > 1).any()
        or (sd < 0).any()
    ):
        raise ValueError("Invalid performance summaries")
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 6.5))
    else:
        fig = ax.figure
    x = np.arange(len(table))
    width = 0.24
    for j, (metric, label, color) in enumerate(METRICS):
        ax.bar(
            x + (j - 1) * width,
            means[:, j],
            width,
            yerr=sd[:, j],
            capsize=3,
            color=color,
            label=label,
            error_kw={"elinewidth": 1},
        )
    names = [
        str(n).replace("SNV_", "SNV-").replace("-CNV_", "\nCNV-") for n in table.model
    ]
    ax.set(
        xticks=x,
        xticklabels=names,
        ylabel="Development CV score",
        title=title,
        ylim=(
            min(0, float((means - sd).min())) - 0.025,
            max(1, float((means + sd).max())) + 0.12,
        ),
    )
    ax.legend(loc="upper left", ncols=3, frameon=False)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    note = "Mean ± sample SD across 100 × 5 correlated folds; not a confidence interval"
    if "STACK" in set(table.model):
        note += "\nSTACK: SNV-TOP + all CNV clusters (fixed baseline)"
    if standalone:
        fig.text(0.5, 0.025, note, ha="center", fontsize=9)
        fig.tight_layout(rect=(0, 0.09, 1, 1))
    else:
        note = "\n".join(textwrap.fill(line, 65) for line in note.split("\n"))
        ax.text(
            0.5, -0.21, note, transform=ax.transAxes, ha="center", va="top", fontsize=8
        )
    return fig


def render_performance(cv_run, snapshot, data_root, output):
    with model_lock(cv_run):
        expected = snapshot_identity(snapshot, data_root)
        execution = runtime_identity()
        metrics, _, _ = collect_complete_cv(cv_run, expected)
        tables, _, _ = build_tables(metrics)
        if (
            snapshot_identity(snapshot, data_root) != expected
            or runtime_identity() != execution
        ):
            raise ValueError("Inputs or source changed before rendering")
        output.mkdir(parents=True, exist_ok=False)
        try:
            figures = {}
            for panel, number, title in [
                ("S19A", 5, "Model benchmarking — development CV"),
                ("S19D", 6, "Stacking configurations — development CV"),
            ]:
                fig = plot_performance(tables[number], title)
                try:
                    figures[panel] = save_tiff_and_preview(fig, output / panel)
                finally:
                    plt.close(fig)
                tables[number].to_csv(
                    output / f"{panel}_numeric_summary.csv", index=False
                )
            if (
                snapshot_identity(snapshot, data_root) != expected
                or runtime_identity() != execution
            ):
                raise ValueError("Inputs or source changed during rendering")
            write_json(
                output / "COMPLETE.json",
                {
                    "status": "succeeded",
                    "figures": figures,
                    "scope": "S19A/D only; full-development outer CV; no holdout or ASC results",
                    "complete_folds": 500,
                    "candidate_rows": 4000,
                    "baseline_STACK_candidate": BASELINE_STACK,
                    "dispersion": "sample SD across correlated folds, not CI; no pairwise significance annotations",
                    "limitations": "Historical frozen CNV clusters include holdout-derived information",
                    "cv_identity": expected,
                    "render_runtime": execution,
                    "artifacts": {
                        p.name: {"sha256": file_hash(p), "bytes": p.stat().st_size}
                        for p in sorted(output.iterdir())
                        if p.is_file()
                    },
                },
            )
        except BaseException as exc:
            write_json(output / "FAILED.json", {"error_type": type(exc).__name__})
            raise


def main():
    from .figure_style import setup_plotting

    setup_plotting()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cv-run", "snapshot", "data-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    render_performance(args.cv_run, args.snapshot, args.data_root, args.output)


if __name__ == "__main__":
    main()
