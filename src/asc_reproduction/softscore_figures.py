"""Figure 5C/D from the frozen 136-person model, without any fitting.

Source: paper/paper_scripts/result3.3.ipynb, cell 51 (65f9b993).
Preserve its subtype Mann–Whitney and paired-component Wilcoxon comparisons.
The full reference cohort is descriptive: 136 training plus 35 test patients.
Individual scores and patient identifiers are retained in memory only.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score

from .score_inputs import keyed_matrices, select_score_panels
from .dataio import read_pivot_sqlite
from .figure_style import plt, save_tiff_and_preview
from .model_inputs import load_model_inputs
from .provenance import file_hash, runtime_identity, write_json


def comparison_table(subtype, paired):
    """Report original raw two-sided comparisons without new correction."""
    if paired.duplicated(["case_ID", "sample_type"]).any():
        raise ValueError("Duplicate case/component in paired comparison")
    matrix = paired.pivot(
        index="case_ID", columns="sample_type", values="soft_score"
    ).reindex(columns=["A", "T", "S"])
    if len(matrix) != 11 or matrix.isna().any().any():
        raise ValueError("Paired comparison requires 11 complete triples")
    records = []
    for panel, frame, column, order in (
        ("5C", subtype, "subtype", ["LUAD", "ASC", "LUSC"]),
        ("5D", paired, "sample_type", ["A", "T", "S"]),
    ):
        for first, second in combinations(order, 2):
            x = frame.loc[frame[column].eq(first), "soft_score"].to_numpy()
            y = frame.loc[frame[column].eq(second), "soft_score"].to_numpy()
            if panel == "5D":
                x = matrix[first].to_numpy()
                y = matrix[second].to_numpy()
            result = (
                mannwhitneyu(x, y, alternative="two-sided", method="auto")
                if panel == "5C"
                else wilcoxon(x, y, alternative="two-sided", method="auto")
            )
            records.append(
                {
                    "panel": panel,
                    "first": first,
                    "second": second,
                    "test": "Mann-Whitney" if panel == "5C" else "Wilcoxon",
                    "statistic": float(result.statistic),
                    "raw_p": float(result.pvalue),
                    "correction": "none",
                }
            )
    return pd.DataFrame(records)


def draw_score_panel(frame, *, paired: bool):
    """Use the pymaftools 0.5.0 boxplot API on correctly ordered rows."""
    from pymaftools import PivotTable

    from .oncoplot_rendering import require_pymaftools_version

    require_pymaftools_version()
    order = ["A", "T", "S"] if paired else ["LUAD", "ASC", "LUSC"]
    colors = (
        ["salmon", "seagreen", "steelblue"]
        if paired
        else [
            "orange",
            "green",
            "blue",
        ]
    )
    table = PivotTable(pd.DataFrame(index=["score"], columns=frame.index))
    table.sample_metadata = frame.copy()
    figure, axis = plt.subplots(figsize=(7, 5))
    table.plot.plot_boxplot_with_annot(
        group_col="sample_type" if paired else "subtype",
        test_col="soft_score",
        order=order,
        palette=dict(zip(order, colors)),
        ax=axis,
        title="",
        xlabel="Component" if paired else "Subtype",
        ylabel="LUAD–LUSC classification score",
        fontsize=12,
        test="Wilcoxon" if paired else "Mann-Whitney",
        comparisons_correction=None,
        is_paired=paired,
        pair_col="case_ID" if paired else None,
    )
    axis.set_yticks(np.linspace(0, 1, 6))
    axis.set_ylim(0, max(1.02, axis.get_ylim()[1]))
    figure.tight_layout()
    return figure


def run(data_root: Path, model_directory: Path, output: Path) -> None:
    """Validate a trusted frozen model, score and save aggregate evidence.

    Never supply an untrusted joblib directory: deserialization executes code.
    This entry point does not export individual scores or fit any model.
    """
    if output.exists():
        raise FileExistsError("Use a new softscore output directory")
    decision = json.loads((model_directory / "selection_decision.json").read_text())
    input_hashes = decision["cv_identity"]["inputs"]
    for relative_path, expected_hash in input_hashes.items():
        if file_hash(data_root / relative_path) != expected_hash:
            raise ValueError("Processed inputs differ from the frozen run")
    runtime = runtime_identity()
    model_path = model_directory / "model.joblib"
    frozen = json.loads((model_directory / "FROZEN_MODEL.json").read_text())
    digest = file_hash(model_path)
    if digest != frozen["sha256"] or frozen["refit_on_holdout"]:
        raise ValueError("Frozen model identity or holdout policy differs")
    model = joblib.load(model_path)
    if (
        model.name != "SNV_RFE-CNV_RFE"
        or model.classes != ("LUAD", "LUSC")
        or {branch.name: len(branch.features) for branch in model.branches}
        != {"snv_rfe": 45, "cnv_rfe": 10}
    ):
        raise ValueError("Unexpected frozen model configuration")
    expected_features = json.loads(
        (model_directory / "selected_features.json").read_text()
    )
    if (
        model.training_count != 136
        or {branch.name: list(branch.features) for branch in model.branches}
        != expected_features
    ):
        raise ValueError("Final training cohort or selected features differ")
    data, split = load_model_inputs(data_root)
    reference, predicted = model.predict(keyed_matrices(data, asc=False))
    asc_scores, _ = model.predict(keyed_matrices(data, asc=True))
    holdout = split.holdout
    truth = data["y"][holdout]
    metrics = {
        "acc": accuracy_score(truth, predicted[holdout]),
        "f1": f1_score(truth, predicted[holdout], average="macro"),
        "auc": roc_auc_score(truth, reference[holdout]),
    }
    expected = json.loads((model_directory / "holdout_metrics.json").read_text())
    for name, value in metrics.items():
        np.testing.assert_allclose(value, expected[name], rtol=0, atol=1e-12)
    np.testing.assert_array_equal(
        confusion_matrix(truth, predicted[holdout]),
        expected["confusion_matrix_LUAD_LUSC"],
    )
    asc_expected = json.loads((model_directory / "ASC_score_summary.json").read_text())
    if len(asc_scores) != asc_expected["n"] or len(asc_scores) != 51:
        raise ValueError("ASC inference cohort differs")
    for name in ("mean", "min", "max"):
        value = getattr(asc_scores, name)()
        np.testing.assert_allclose(value, asc_expected[name], rtol=0, atol=1e-12)
    metadata = read_pivot_sqlite(
        data_root / "data/WES/all_case_table.db"
    ).sample_metadata
    subtype, paired = select_score_panels(
        pd.Series(reference, index=data["patients"]),
        pd.Series(asc_scores, index=data["asc_samples"]),
        metadata,
    )
    output.mkdir(parents=True)
    comparison_table(subtype, paired).to_csv(output / "tests.csv", index=False)
    for name, frame, is_paired in (
        ("Figure_5C_softscore", subtype, False),
        ("Figure_5D_softscore", paired, True),
    ):
        figure = draw_score_panel(frame, paired=is_paired)
        save_tiff_and_preview(figure, output / name)
        plt.close(figure)
    if file_hash(model_path) != digest:
        raise ValueError("Frozen model changed during inference")
    if runtime != runtime_identity() or any(
        file_hash(data_root / path) != digest for path, digest in input_hashes.items()
    ):
        raise ValueError("Inputs or source changed during softscore rendering")
    write_json(
        output / "COMPLETE.json",
        {
            "model_sha256": digest,
            "model_fitting": False,
            "runtime": runtime,
            "holdout_replay": metrics,
            "input_sha256": input_hashes,
            "panel_counts": {"5C": len(subtype), "5D": len(paired)},
            "interpretation": "5C includes training data; not validation",
            "artifacts": {p.name: file_hash(p) for p in output.iterdir()},
        },
    )


def main() -> None:
    """Parse explicit processed-input, trusted model and new-output paths."""
    from .figure_style import setup_plotting

    setup_plotting()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.data_root, args.model_directory, args.output)


if __name__ == "__main__":
    main()
