"""Render original-style S19 rows from saved CV and RFECV results only."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pymaftools.plot.ModelPlot import ModelPlot

from .figure_style import plt, setup_plotting
from .oncoplot_rendering import require_pymaftools_version
from .provenance import file_hash, write_json


def save_row(figure, target):
    """Keep an identical raster width for all rows without stretching."""
    figure.savefig(
        target.with_suffix(".tiff"),
        dpi=600,
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    figure.savefig(target.with_suffix(".png"), dpi=150, facecolor="white")
    plt.close(figure)


def space_annotations(axis):
    """Reposition package-generated brackets and stars without changing tests."""
    brackets = []
    for line in axis.lines:
        x = np.asarray(line.get_xdata())
        y = np.asarray(line.get_ydata())
        if (
            len(x) == 4
            and x[0] == x[1]
            and x[2] == x[3]
            and x[0] != x[3]
            and y[0] == y[3]
            and y[1] == y[2]
        ):
            brackets.append(line)
    annotations = [text for text in axis.texts if hasattr(text, "xy")]
    assert len(brackets) == len(annotations) and brackets
    pairs = sorted(
        zip(brackets, annotations), key=lambda pair: max(pair[0].get_ydata())
    )
    span = 1 - axis.get_ylim()[0]
    for level, (line, text) in enumerate(pairs):
        bottom = 1 + span * (0.10 + level * 0.13)
        top = bottom + span * 0.016
        line.set_ydata([bottom, top, top, bottom])
        text.xy = (float(np.mean(line.get_xdata())), top)
        text.set_position((0, 3))
        text.set_fontsize(9)
    axis.set_ylim(axis.get_ylim()[0], top + span * 0.16)


def saved_input_paths(cv_run, summary_directory, rfecv_directory):
    """Resolve all saved inputs before rendering, without fitting any model."""
    paths = {}
    for seed in range(100):
        for fold in range(5):
            pointer = cv_run / f"seed_{seed:03d}_fold_{fold}.latest.json"
            name = json.loads(pointer.read_text())["attempt"]
            if Path(name).name != name or not name.startswith(f"s{seed:03}f{fold}-"):
                raise ValueError("Invalid saved CV attempt path")
            paths[f"cv/{seed}/{fold}"] = cv_run / name / "metrics.csv"
    for letter in "AD":
        paths[f"summary/{letter}"] = summary_directory / f"S19{letter}_numeric_summary.csv"
    for letter in "BC":
        paths[f"rfecv/{letter}"] = rfecv_directory / f"S19{letter}_rfecv_values.csv"
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    return paths


def render(cv_run, summary_directory, rfecv_directory, output):
    """Validate saved summaries and reproduce notebook Mann–Whitney labels."""
    require_pymaftools_version()
    paths = saved_input_paths(cv_run, summary_directory, rfecv_directory)
    hashes = {name: file_hash(path) for name, path in paths.items()}
    setup_plotting()
    output.mkdir(parents=True, exist_ok=False)
    frames = []
    for seed in range(100):
        for fold in range(5):
            path = paths[f"cv/{seed}/{fold}"]
            frame = pd.read_csv(path)
            assert frame.seed.eq(seed).all() and frame.fold.eq(fold).all()
            frames.append(frame)
    metrics = pd.concat(frames, ignore_index=True)
    assert len(metrics) == 4000
    assert not metrics.duplicated(["model", "seed", "fold"]).any()
    names = {"acc": "Accuracy", "f1": "F1 Score", "auc": "AUC"}
    for letter in "AD":
        path = paths[f"summary/{letter}"]
        summary = pd.read_csv(path)
        selected = metrics[metrics.model.isin(summary.candidate)].copy()
        for metric in names:
            grouped = selected.groupby("model")[metric]
            np.testing.assert_allclose(
                grouped.mean().loc[summary.candidate], summary[f"{metric}_mean"]
            )
            np.testing.assert_allclose(
                grouped.std().loc[summary.candidate], summary[f"{metric}_sd"]
            )
        assert selected.groupby("model").size().eq(500).all()
        labels = [
            str(x)
            .replace("SNV_", "SNV-")
            .replace("-CNV_", "\nCNV-")
            .replace("CNV-ALL", "CNV-Cluster")
            for x in summary.model
        ]
        labels = [x.replace("-", "\n", 1) for x in labels] if letter == "A" else labels
        selected["model"] = selected.model.map(dict(zip(summary.candidate, labels)))
        plot = ModelPlot().plot_metric_comparison_with_annotation(
            selected.rename(columns=names),
            metrics=list(names.values()),
            order=labels,
            test="Mann-Whitney",
            figsize=(12, 5),
            fontsize=9,
            rotation=0,
        )
        for axis in plot.fig.axes:
            space_annotations(axis)
            axis.set_yticks([t for t in axis.get_yticks() if 0 <= t <= 1])
        plot.fig.subplots_adjust(
            left=0.07, right=0.99, bottom=0.18, top=0.96, wspace=0.30
        )
        save_row(plot.fig, output / f"S19{letter}")
    for letter, count in [("B", 45), ("C", 10)]:
        path = paths[f"rfecv/{letter}"]
        values = pd.read_csv(path)
        x, mean, sd = [
            values[k].to_numpy()
            for k in ("n_features", "mean_test_score", "std_test_score")
        ]
        assert (
            count in x and np.isfinite(mean).all() and ((mean >= 0) & (mean <= 1)).all()
        )
        fig, ax = plt.subplots(figsize=(12, 2.8))
        ax.plot(
            x,
            mean,
            color="steelblue",
            label="Mean internal CV accuracy",
        )
        ax.fill_between(
            x,
            mean - sd,
            mean + sd,
            color="steelblue",
            alpha=0.2,
            label="±1 SD across 5 folds",
        )
        ax.axvline(
            count, color="tomato", linestyle="--", label=f"Selected ({count} features)"
        )
        ax.set(
            xlabel="Number of features selected",
            ylabel="Mean internal 5-fold CV accuracy",
            ylim=(max(0, float((mean - sd).min()) - 0.02), 1),
        )
        ax.legend(loc="lower left", fontsize=9)
        ax.grid(alpha=0.2)
        fig.subplots_adjust(left=0.07, right=0.99, bottom=0.22, top=0.94)
        save_row(fig, output / f"S19{letter}")
    if any(file_hash(paths[name]) != digest for name, digest in hashes.items()):
        raise ValueError("Saved S19 input changed while rendering")
    write_json(
        output / "manifest.json",
        {
            "inputs": hashes,
            "model_fitting": False,
            "pymaftools": "0.5.0",
            "test": "Notebook-default two-sided Mann–Whitney, unadjusted; correlated CV folds",
            "limitations": "Descriptive reproduction, not independent-fold inference. B/C SD outside 0–1 is clipped by display, not changed in data.",
            "status": "candidate; approval pending",
        },
    )


def main():
    """Read explicit, relocatable saved results and write candidate panels."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv-run", type=Path, required=True)
    parser.add_argument("--summary-directory", type=Path, required=True)
    parser.add_argument("--rfecv-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.cv_run, args.summary_directory, args.rfecv_directory, args.output)


if __name__ == "__main__":
    main()
