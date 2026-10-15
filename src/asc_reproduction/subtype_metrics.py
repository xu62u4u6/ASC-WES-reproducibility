"""Reproduce the cohort-level TMB and HRD boxplots (Figure 4B/C)."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from scipy.stats import mannwhitneyu
from statannotations.Annotator import Annotator

from .dataio import read_pivot_sqlite
from .figure_style import SUBTYPE_COLORS, plt

GROUP_ORDER = ("LUAD", "ASC", "LUSC")
EXPECTED_GROUP_COUNTS = {"LUAD": 89, "ASC": 29, "LUSC": 82}
METRICS = ("TMB", "HRD-sum")
PALETTE = SUBTYPE_COLORS
PAIRS = tuple(combinations(GROUP_ORDER, 2))


@dataclass(frozen=True)
class SubtypeMetricResults:
    group_summary: pd.DataFrame
    pairwise_statistics: pd.DataFrame


def _significance_label(p_value: float) -> str:
    if p_value <= 1e-4:
        return "****"
    if p_value <= 1e-3:
        return "***"
    if p_value <= 1e-2:
        return "**"
    if p_value <= 0.05:
        return "*"
    return "ns"


def calculate_subtype_metric_statistics(
    sample_metadata: pd.DataFrame,
    *,
    metrics: tuple[str, ...] = METRICS,
    expected_group_counts: dict[str, int] | None = EXPECTED_GROUP_COUNTS,
) -> SubtypeMetricResults:
    """Summarize processed metrics and run the legacy raw pairwise tests.

    The legacy figure calls ``plot_boxplot_with_annot`` with its defaults:
    unpaired, two-sided Mann–Whitney U tests and no multiple-comparison
    correction. This function preserves those annotations and records raw
    p-values explicitly; the manuscript's BH wording is tracked separately.
    """
    required = {"sample_type", "subtype", *metrics}
    absent = required - set(sample_metadata.columns)
    if absent:
        raise ValueError(f"Sample metadata is missing: {', '.join(sorted(absent))}")
    selected = sample_metadata.loc[
        sample_metadata["sample_type"].eq("T")
        & sample_metadata["subtype"].isin(GROUP_ORDER),
        ["subtype", *metrics],
    ].copy()
    counts = (
        selected["subtype"].value_counts().reindex(GROUP_ORDER, fill_value=0).to_dict()
    )
    if expected_group_counts is not None and counts != expected_group_counts:
        raise ValueError(
            f"Expected T-sample counts {expected_group_counts}; observed {counts}"
        )

    group_rows = []
    pairwise_rows = []
    for metric in metrics:
        if not np.isfinite(selected[metric].to_numpy(dtype=float)).all():
            raise ValueError(f"Metric {metric!r} contains missing or non-finite values")
        for group in GROUP_ORDER:
            values = selected.loc[selected["subtype"].eq(group), metric].to_numpy(
                dtype=float
            )
            group_rows.append(
                {
                    "metric": metric,
                    "subtype": group,
                    "n": len(values),
                    "median": float(np.median(values)),
                    "minimum": float(np.min(values)),
                    "maximum": float(np.max(values)),
                }
            )
        for first, second in PAIRS:
            first_values = selected.loc[selected["subtype"].eq(first), metric].to_numpy(
                dtype=float
            )
            second_values = selected.loc[
                selected["subtype"].eq(second), metric
            ].to_numpy(dtype=float)
            test_result = mannwhitneyu(
                first_values, second_values, alternative="two-sided"
            )
            pairwise_rows.append(
                {
                    "metric": metric,
                    "comparison": f"{first} vs {second}",
                    "n_first": len(first_values),
                    "n_second": len(second_values),
                    "U_statistic": float(test_result.statistic),
                    "raw_p_value": float(test_result.pvalue),
                    "annotation": _significance_label(float(test_result.pvalue)),
                    "correction": "none (legacy figure behavior)",
                }
            )
    return SubtypeMetricResults(
        group_summary=pd.DataFrame(group_rows),
        pairwise_statistics=pd.DataFrame(pairwise_rows),
    )


def make_subtype_metric_figure(
    sample_metadata: pd.DataFrame,
    *,
    metric: str,
    alpha: float = 0.8,
) -> plt.Figure:
    """Render Figure 4B/C with the original pymaftools statistical plot API.

    Keep the validated T-only cohort, unpaired raw Mann–Whitney tests, palette
    and 6-by-5-inch canvas. No multiple-testing change is made during migration.
    """
    from pymaftools import PivotTable

    from .oncoplot_rendering import require_pymaftools_version

    require_pymaftools_version()
    calculate_subtype_metric_statistics(
        sample_metadata, metrics=(metric,), expected_group_counts=None
    )
    data = sample_metadata.loc[
        sample_metadata.sample_type.eq("T") & sample_metadata.subtype.isin(GROUP_ORDER),
        ["subtype", metric],
    ].copy()
    if data.index.has_duplicates:
        raise ValueError("Subtype metric samples must have unique identifiers")
    table = PivotTable(pd.DataFrame(index=[metric], columns=data.index))
    table.sample_metadata = data
    figure, axis = plt.subplots(figsize=(6, 5))
    table.plot.plot_boxplot_with_annot(
        group_col="subtype",
        test_col=metric,
        order=list(GROUP_ORDER),
        palette=PALETTE,
        ax=axis,
        fontsize=12,
        alpha=alpha,
        test="Mann-Whitney",
        comparisons_correction=None,
        is_paired=False,
    )
    return figure


def make_reference_subtype_metric_figure(
    sample_metadata: pd.DataFrame,
    *,
    metric: str,
) -> plt.Figure:
    """Draw the legacy boxplot and raw-p star annotations for one metric."""
    data = sample_metadata.loc[
        sample_metadata["sample_type"].eq("T")
        & sample_metadata["subtype"].isin(GROUP_ORDER),
        ["subtype", metric],
    ].copy()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.boxplot(
        data=data,
        x="subtype",
        y=metric,
        hue="subtype",
        palette=PALETTE,
        order=GROUP_ORDER,
        ax=ax,
    )
    for patch in ax.patches:
        face = patch.get_facecolor()
        patch.set_facecolor((face[0], face[1], face[2], 0.8))
    annotator = Annotator(
        ax,
        PAIRS,
        data=data,
        x="subtype",
        y=metric,
        order=GROUP_ORDER,
    )
    annotator.configure(
        test="Mann-Whitney",
        comparisons_correction=None,
        text_format="star",
        loc="inside",
        verbose=False,
    )
    annotator.apply_and_annotate()
    ax.set_title(f"Boxplot of {metric} by subtype", fontsize=12)
    ax.set_xlabel("subtype", fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    counts = data.groupby("subtype").size().to_dict()
    ax.set_xticks(range(len(GROUP_ORDER)))
    ax.set_xticklabels(
        [f"{group} (n={counts[group]})" for group in GROUP_ORDER], fontsize=12
    )
    return fig


def save_fixed_canvas_figure(fig: plt.Figure, output_base: Path) -> dict[str, object]:
    """Save the legacy 6×5-inch canvas as a decoded 600-dpi RGB TIFF."""
    png_path = Path(f"{output_base}.png")
    tiff_path = Path(f"{output_base}.tiff")
    fig.tight_layout()
    fig.savefig(png_path, dpi=150, facecolor="white")
    fig.set_dpi(600)
    fig.canvas.draw()
    rgb = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGB")
    rgb.save(tiff_path, compression="tiff_lzw", dpi=(600, 600))
    with Image.open(tiff_path) as image:
        image.load()
        if image.mode != "RGB":
            raise ValueError(f"Expected RGB TIFF, received mode {image.mode}")
        dpi = [int(round(float(value))) for value in image.info.get("dpi", ())]
        return {
            "tiff": tiff_path.name,
            "preview": png_path.name,
            "pixels": list(image.size),
            "dpi": dpi,
            "mode": image.mode,
            "decode": "passed",
        }


def run(data_root: Path, output_dir: Path) -> dict[str, object]:
    """Produce package-rendered panels and aggregate reproducibility records."""
    from .provenance import file_hash, runtime_identity

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty output directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = data_root / "data/WES/all_case_table.db"
    input_hash = file_hash(input_path)
    runtime = runtime_identity()
    cohort = read_pivot_sqlite(input_path)
    results = calculate_subtype_metric_statistics(cohort.sample_metadata)
    results.group_summary.to_csv(
        output_dir / "Figure4B_C_group_summary.csv", index=False
    )
    results.pairwise_statistics.to_csv(
        output_dir / "Figure4B_C_pairwise_raw_statistics.csv", index=False
    )

    manifests = []
    figure_names = {"TMB": "Figure4B_TMB", "HRD-sum": "Figure4C_HRD"}
    for metric, name in figure_names.items():
        figure = make_subtype_metric_figure(cohort.sample_metadata, metric=metric)
        manifests.append(save_fixed_canvas_figure(figure, output_dir / name))
        plt.close(figure)

    summary = {
        "renderer": "pymaftools 0.5.0 plot_boxplot_with_annot",
        "input_sha256": {"data/WES/all_case_table.db": input_hash},
        "runtime": runtime,
        "status": "complete_legacy_method_reproduced",
        "analysis": "T-sample subtype TMB and HRD distributions",
        "figure_panels": {"TMB": "Figure 4B", "HRD-sum": "Figure 4C"},
        "sample_selection": "all sample_type=T samples grouped by subtype",
        "subtype_counts": EXPECTED_GROUP_COUNTS,
        "test": "two-sided unpaired Mann–Whitney U for each pair",
        "multiple_testing_correction": "none in the legacy figure implementation",
        "figure_legacy_statistics_preserved": True,
        "manuscript_methods_correction_wording_needs_reconciliation": True,
        "group_summary_file": "Figure4B_C_group_summary.csv",
        "pairwise_statistics_file": "Figure4B_C_pairwise_raw_statistics.csv",
        "figure_manifest": manifests,
        "patient_or_sample_identifiers_written": False,
    }
    if file_hash(input_path) != input_hash or runtime_identity() != runtime:
        raise RuntimeError("Subtype metric inputs or source changed during run")
    summary["artifacts"] = {
        path.name: file_hash(path) for path in sorted(output_dir.iterdir())
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    from .figure_style import setup_plotting

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
