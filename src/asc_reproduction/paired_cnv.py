"""Reproduce the matched-component 3q CNV test and gain-proportion panel."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from .dataio import read_pivot_sqlite
from .figure_style import (
    PART_COLORS,
    plt,
    save_tiff_and_preview,
    setup_plotting,
)

PARTS = ("A", "T", "S")
COLORS = tuple(PART_COLORS.values())
EXPECTED_CASES = 11
GAIN_THRESHOLD = 0.10
EXPECTED_GAIN_COUNTS = (2, 4, 8)
EXPECTED_PAIRED_P = 0.001953125


@dataclass(frozen=True)
class Paired3QResult:
    values: pd.DataFrame
    gain_summary: pd.DataFrame
    paired_p_value: float


def calculate_paired_3q(
    arm_cnv: pd.DataFrame,
    sample_metadata: pd.DataFrame,
    *,
    expected_case_count: int = EXPECTED_CASES,
    gain_threshold: float = GAIN_THRESHOLD,
) -> Paired3QResult:
    """Align 3q A/T/S values by case and compute the prespecified summaries.

    The returned per-case frame uses integer row labels, not case/sample IDs.
    The paired inferential contrast is S versus A; gain proportions use the
    inclusive threshold ``log2 ratio >= gain_threshold`` for each component.
    """

    required = {"case_ID", "sample_type"}
    absent = required - set(sample_metadata.columns)
    if absent:
        raise ValueError(f"Sample metadata is missing: {', '.join(sorted(absent))}")
    if "3q" not in arm_cnv.index.astype(str):
        raise ValueError("Arm-level CNV matrix does not contain feature '3q'")

    metadata = sample_metadata.loc[
        sample_metadata["sample_type"].isin(PARTS), ["case_ID", "sample_type"]
    ].copy()
    metadata = metadata.loc[metadata.index.isin(arm_cnv.columns)]
    if metadata[["case_ID", "sample_type"]].isna().to_numpy().any():
        raise ValueError("Component sample metadata contains missing case/type")
    if metadata.duplicated(["case_ID", "sample_type"]).any():
        raise ValueError("A case has more than one sample for a component type")

    sample_by_case = metadata.reset_index(names="sample").pivot(
        index="case_ID", columns="sample_type", values="sample"
    )
    if not set(PARTS).issubset(sample_by_case.columns):
        raise ValueError("A/T/S complete matched cases could not be identified")
    complete = sample_by_case.dropna(subset=list(PARTS)).sort_index()
    if len(complete) != expected_case_count:
        raise ValueError(
            f"Expected {expected_case_count} complete A/T/S cases; found {len(complete)}"
        )

    feature = arm_cnv.index.astype(str).get_loc("3q")
    row = arm_cnv.iloc[feature]
    values = pd.DataFrame(
        {
            part: row.reindex(complete[part].tolist()).to_numpy(dtype=float)
            for part in PARTS
        },
        index=pd.RangeIndex(len(complete), name="matched_case_index"),
    )
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("Matched 3q CNV values contain non-finite values")

    gain_counts = (values.loc[:, PARTS] >= gain_threshold).sum(axis=0).astype(int)
    gain_summary = pd.DataFrame(
        {
            "component": PARTS,
            "n_gain": [int(gain_counts[part]) for part in PARTS],
            "n_total": len(values),
            "percent": [100.0 * int(gain_counts[part]) / len(values) for part in PARTS],
        }
    )
    p_value = float(wilcoxon(values["S"], values["A"], alternative="two-sided").pvalue)
    return Paired3QResult(
        values=values, gain_summary=gain_summary, paired_p_value=p_value
    )


def make_paired_figure(values: pd.DataFrame, p_value: float, *, ax=None) -> plt.Figure:
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure
    for row in values.loc[:, PARTS].itertuples(index=False, name=None):
        ax.plot(range(3), row, color="gray", alpha=0.45, linewidth=0.8, zorder=1)
    for index, part in enumerate(PARTS):
        ax.scatter(
            np.full(len(values), index),
            values[part],
            color=COLORS[index],
            s=30,
            zorder=2,
            label=part,
        )
    # Restore the original manuscript's box summaries over the matched lines.
    boxes = ax.boxplot(
        [values[part].to_numpy() for part in PARTS],
        positions=range(3),
        widths=0.42,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "orange"},
    )
    for patch, color in zip(boxes["boxes"], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax.set_xticks(range(3), ("Adeno", "Tumor", "Squamous"))
    ax.set_ylabel("3q CNV log2 ratio")
    ax.set_title("3q paired components (11 matched cases)")
    ax.text(
        0.02,
        0.98,
        f"S vs A: paired Wilcoxon p = {p_value:.6g}",
        transform=ax.transAxes,
        va="top",
        fontsize=10,
    )
    return fig


def make_gain_figure(gain_summary: pd.DataFrame, *, ax=None, alpha=1.0) -> plt.Figure:
    """Draw saved gain proportions, optionally on a shared-style panel axis."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure
    bars = ax.bar(
        gain_summary["component"],
        gain_summary["percent"],
        color=[PART_COLORS[p] for p in gain_summary["component"]],
        alpha=alpha,
    )
    for bar, row in zip(bars, gain_summary.itertuples(index=False)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            row.percent + 2,
            f"{row.percent:.0f}%\n({row.n_gain}/{row.n_total})",
            ha="center",
            va="bottom",
        )
    ax.set_ylim(0, 90)
    ax.set_ylabel("% of samples with 3q gain (log2 ratio ≥ 0.10)")
    ax.set_title("3q-gain proportion by component (threshold ≥ 0.10)")
    return fig


def run(data_root: Path, output_dir: Path) -> dict[str, object]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty output directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    snv = read_pivot_sqlite(data_root / "data/WES/all_case_table.db")
    arm = read_pivot_sqlite(data_root / "data/CNV_0407/arm_cnv.db")
    sample_metadata = snv.sample_metadata
    if set(arm.values.columns) != set(sample_metadata.index):
        raise ValueError("SNV sample metadata does not cover the arm-CNV matrix")

    result = calculate_paired_3q(arm.values, sample_metadata)
    gain_counts = tuple(result.gain_summary["n_gain"].astype(int))
    if gain_counts != EXPECTED_GAIN_COUNTS:
        raise AssertionError(
            f"Expected inclusive-threshold gain counts {EXPECTED_GAIN_COUNTS}; got {gain_counts}"
        )
    if not np.isclose(result.paired_p_value, EXPECTED_PAIRED_P):
        raise AssertionError(
            f"Expected paired S-vs-A p={EXPECTED_PAIRED_P}; got {result.paired_p_value}"
        )

    result.gain_summary.to_csv(output_dir / "3q_gain_summary_ge_0.10.csv", index=False)
    figure_manifest = []
    for name, figure in (
        (
            "Figure_3A_paired_3q",
            make_paired_figure(result.values, result.paired_p_value),
        ),
        (
            "Supplementary_Figure_10B_gain_ge_0.10",
            make_gain_figure(result.gain_summary),
        ),
    ):
        figure_manifest.append(save_tiff_and_preview(figure, output_dir / name))
        plt.close(figure)

    summary = {
        "status": "complete",
        "analysis": "matched-component 3q CNV",
        "n_matched_cases": len(result.values),
        "paired_contrast": "S versus A",
        "paired_test": "two-sided Wilcoxon signed-rank",
        "paired_p_value": result.paired_p_value,
        "gain_operator": ">=",
        "gain_threshold_log2_ratio": GAIN_THRESHOLD,
        "gain_counts_A_T_S": list(gain_counts),
        "figure_manifest": figure_manifest,
        "patient_or_sample_identifiers_written": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    """Initialize plotting and reproduce the matched 3q panels."""
    setup_plotting()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="external data root containing the data/ directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new or empty local output directory (never overwritten)",
    )
    args = parser.parse_args()
    print(json.dumps(run(args.data_root.resolve(), args.output.resolve()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
