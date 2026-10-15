"""S4 schematic of the confirmed validation design, not a claim of run completion."""

import argparse
import json
from pathlib import Path

from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from .figure_style import plt, save_tiff_and_preview
from .provenance import file_hash, write_json


def workflow_labels(spec):
    c = spec["classifier"]
    s = c["split"]
    if (s["development_n"], s["holdout_n"], s["seed"]) != (136, 35, 42):
        raise ValueError("Workflow layout requires review for changed split")
    if (
        c["development_evaluation"]["seeds"] != "0 through 99"
        or c["stacking"]["internal_oof_folds"] != 5
    ):
        raise ValueError("Workflow layout requires review for changed CV")
    if (
        c["development_evaluation"]["outer_cv"]
        != "StratifiedKFold, 5 folds, shuffle=true"
        or not c["stacking"]["feature_selection_within_each_internal_training_split"]
        or c["model_selection_order"]
        != ["mean AUROC", "mean macro-F1", "mean accuracy"]
        or c["exact_tie_rule"] != "lexicographic model name"
        or s["development_class_counts_LUAD_LUSC"] != [71, 65]
        or s["holdout_class_counts_LUAD_LUSC"] != [18, 17]
        or not s["patient_level_split"]
    ):
        raise ValueError(
            "Workflow layout requires review for changed evaluation design"
        )
    return {
        "reference": "171 LUAD/LUSC patients\nSort IDs; stratified patient-level 80/20 split (seed 42)",
        "development": "Development: 136 patients\n71 LUAD / 65 LUSC",
        "holdout": "Reserved internal holdout: 35 patients\n18 LUAD / 17 LUSC",
        "cv": "DEVELOPMENT ONLY: 100 seeds × 5 outer folds\n\nOuter training: fit TOP / RFECV and base models\nStacking: 5 inner folds; reselect features in each inner training split\nFit meta-model on out-of-fold base predictions\n\nOuter validation: evaluate the fitted candidates\n4 standalone models + 4 stacking configurations",
        "selection": "Choose the method using complete development CV\nMean AUROC → macro-F1 → accuracy → model name",
        "refit": "Refit the selected method on all 136 development patients\nRepeat feature selection; retain the resulting fixed feature list\nStacking, if selected, uses internal out-of-fold training again",
        "freeze": "Freeze the fitted model and feature list\nDo not refit using holdout or ASC",
        "test": "Evaluate the SAME frozen model on 35 holdout patients\nAccuracy / macro-F1 / AUROC\nDo not use these results to choose the method",
        "asc": "Apply the SAME frozen model to ASC samples\nDescriptive reference-model scores only\nASC is not used for fitting or method selection",
    }


def plot_workflow(labels):
    fig, ax = plt.subplots(figsize=(13, 13))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    boxes = {
        "reference": (0.5, 0.92, 0.9, 0.075),
        "development": (0.30, 0.805, 0.50, 0.075),
        "holdout": (0.79, 0.805, 0.35, 0.075),
        "cv": (0.30, 0.635, 0.50, 0.22),
        "selection": (0.30, 0.45, 0.50, 0.095),
        "refit": (0.30, 0.315, 0.50, 0.10),
        "freeze": (0.30, 0.18, 0.50, 0.075),
        "test": (0.79, 0.605, 0.35, 0.12),
        "asc": (0.79, 0.31, 0.35, 0.12),
    }
    for name, (x, y, w, h) in boxes.items():
        color = "#E8F1FA" if name not in ("holdout", "test", "asc") else "#EDF5EA"
        ax.add_patch(
            FancyBboxPatch(
                (x - w / 2, y - h / 2),
                w,
                h,
                boxstyle="round,pad=0.005",
                facecolor=color,
                edgecolor="#526575",
                linewidth=1,
            )
        )
        ax.text(
            x,
            y,
            labels[name],
            ha="center",
            va="center",
            fontsize=9 if name in ("cv", "test", "asc") else 10,
        )

    def arrow(points):
        for start, end in zip(points[:-2], points[1:-1]):
            ax.plot(
                [start[0], end[0]], [start[1], end[1]], color="#526575", linewidth=1.2
            )
        ax.add_patch(
            FancyArrowPatch(
                points[-2],
                points[-1],
                arrowstyle="-|>",
                mutation_scale=12,
                color="#526575",
                linewidth=1.2,
            )
        )

    arrow([(0.5, 0.877), (0.5, 0.86), (0.30, 0.86), (0.30, 0.848)])
    arrow([(0.5, 0.86), (0.79, 0.86), (0.79, 0.848)])
    for start, end in [(0.762, 0.752), (0.52, 0.503), (0.398, 0.372), (0.26, 0.222)]:
        arrow([(0.30, start), (0.30, end)])
    arrow([(0.79, 0.762), (0.79, 0.671)])
    arrow([(0.556, 0.18), (0.585, 0.18), (0.585, 0.605), (0.609, 0.605)])
    arrow([(0.585, 0.31), (0.609, 0.31)])
    ax.text(
        0.585,
        0.48,
        "Frozen model",
        rotation=90,
        ha="right",
        va="center",
        fontsize=9,
        color="#526575",
    )
    ax.text(
        0.5,
        0.073,
        "RFECV uses internal five-fold accuracy; TOP is fitted before RFECV within each supplied training set.\n"
        "Limitation: the frozen 114 CNV clusters were derived from the full reference cohort, including the later holdout.\n"
        "This is retrospective internal validation, not an external or fully independent validation cohort.",
        ha="center",
        va="center",
        fontsize=9,
        color="#6B431C",
    )
    ax.set_title(
        "Model development, internal validation and ASC scoring", fontsize=15, pad=12
    )
    fig.tight_layout()
    return fig


def main():
    from .figure_style import setup_plotting

    setup_plotting()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--data-root",
        type=Path,
        help="Accepted for runner compatibility; no patient data is read",
    )
    a = p.parse_args()
    spec_path = Path(__file__).resolve().parents[2] / "analysis_spec.json"
    digest = file_hash(spec_path)
    labels = workflow_labels(json.loads(spec_path.read_text()))
    if a.output.exists() and any(a.output.iterdir()):
        raise ValueError("Output must be new or empty")
    a.output.mkdir(parents=True, exist_ok=True)
    fig = plot_workflow(labels)
    try:
        info = save_tiff_and_preview(fig, a.output / "S4_model_workflow")
        fig.savefig(a.output / "S4_model_workflow.svg", format="svg")
    finally:
        plt.close(fig)
    if file_hash(spec_path) != digest:
        raise ValueError("Specification changed during rendering")
    write_json(
        a.output / "summary.json",
        {
            "scope": "confirmed method schematic; not numerical results or run completion",
            "analysis_spec_sha256": digest,
            "labels": labels,
            "figure": info,
            "editable_source": "model_workflow.py",
            "artifacts": {
                f.name: {"sha256": file_hash(f), "bytes": f.stat().st_size}
                for f in a.output.iterdir()
                if f.is_file()
            },
        },
    )


if __name__ == "__main__":
    main()
