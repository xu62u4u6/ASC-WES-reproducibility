"""S8: paired component metrics, retaining legacy raw Wilcoxon annotations."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

from .c47_profile import _as_sample_mask
from .dataio import read_pivot_sqlite
from .figure_style import plt, save_tiff_and_preview
from .paired_screen import matched_part_matrices
from .subtype_metrics import _significance_label

METRICS = ("MSI", "TMB", "ti/tv", "HRD-sum")
PARTS = ("A", "T", "S")


def calculate_component_metrics(metadata, *, expected_cases=11):
    if metadata.index.has_duplicates:
        raise ValueError("Duplicate sample keys")
    selected = metadata.loc[_as_sample_mask(metadata["AS_sample"])]
    if len(selected) != expected_cases * 3 or not selected.subtype.eq("ASC").all():
        raise ValueError("Expected complete ASC triplets only")
    arrays = matched_part_matrices(
        selected[list(METRICS)].T, selected, expected_case_count=expected_cases
    )
    summaries, tests, overall = [], [], []
    for i, metric in enumerate(METRICS):
        values = [arrays[p][i] for p in PARTS]
        for p, x in zip(PARTS, values):
            summaries.append(
                dict(
                    metric=metric,
                    component=p,
                    n=len(x),
                    median=float(np.median(x)),
                    minimum=float(x.min()),
                    maximum=float(x.max()),
                )
            )
        for a, b in [(0, 1), (1, 2), (0, 2)]:
            p = (
                1.0
                if np.array_equal(values[a], values[b])
                else float(wilcoxon(values[a], values[b]).pvalue)
            )
            tests.append(
                dict(
                    metric=metric,
                    first=PARTS[a],
                    second=PARTS[b],
                    n_pairs=expected_cases,
                    p_raw_two_sided=p,
                    annotation=_significance_label(p),
                    test="paired Wilcoxon",
                    correction="none; legacy candidate",
                )
            )
        p = (
            1.0
            if all(np.array_equal(values[0], x) for x in values[1:])
            else float(friedmanchisquare(*values).pvalue)
        )
        overall.append(
            dict(
                metric=metric,
                n_cases=expected_cases,
                p_raw=p,
                test="Friedman",
                role="legacy descriptive audit; not boxplot annotation",
            )
        )
    return arrays, pd.DataFrame(summaries), pd.DataFrame(tests), pd.DataFrame(overall)


def draw_metric(ax, metadata, index):
    """Use the notebook's package boxplot with explicitly aligned case IDs."""
    from pymaftools import PivotTable

    from .figure_style import PART_COLORS
    from .oncoplot_rendering import require_pymaftools_version

    require_pymaftools_version()
    data = metadata.loc[_as_sample_mask(metadata["AS_sample"])].copy()
    data = data.sort_values(["case_ID", "sample_type"])
    assert not data.duplicated(["case_ID", "sample_type"]).any()
    case_orders = [
        data.loc[data.sample_type.eq(part), "case_ID"].tolist() for part in PARTS
    ]
    assert case_orders[0] == case_orders[1] == case_orders[2]
    assert len(case_orders[0]) == 11
    table = PivotTable(pd.DataFrame(index=[METRICS[index]], columns=data.index))
    table.sample_metadata = data
    table.plot.plot_boxplot_with_annot(
        group_col="sample_type",
        test_col=METRICS[index],
        test="Wilcoxon",
        is_paired=True,
        pair_col="case_ID",
        comparisons_correction=None,
        palette=PART_COLORS,
        order=list(PARTS),
        ax=ax,
        fontsize=12,
        alpha=0.6,
    )


def main():
    from .figure_style import setup_plotting

    setup_plotting()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Output must be new or empty")
    pivot = read_pivot_sqlite(args.data_root / "data/WES/all_case_table.db")
    arrays, groups, tests, overall = calculate_component_metrics(pivot.sample_metadata)
    args.output.mkdir(parents=True, exist_ok=True)
    figures = {}
    for i in range(4):
        fig, ax = plt.subplots(figsize=(6, 5))
        draw_metric(ax, pivot.sample_metadata, i)
        name = f"S8{chr(65 + i)}"
        figures[name] = save_tiff_and_preview(fig, args.output / name)
        plt.close(fig)
    groups.to_csv(args.output / "S8_group_summary.csv", index=False)
    tests.to_csv(args.output / "S8_pairwise_raw.csv", index=False)
    overall.to_csv(args.output / "S8_friedman_raw.csv", index=False)
    summary = {
        "panel": "S8",
        "renderer": "pymaftools 0.5.0 plot_boxplot_with_annot",
        "cases": 11,
        "metrics": list(METRICS),
        "figures": figures,
        "all_pairwise_raw_p_ge_005": bool(tests.p_raw_two_sided.ge(0.05).all()),
        "limitations": "Processed metric provenance pending; raw-p legacy annotations, manuscript BH wording unresolved. No IDs/per-sample tables exported.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
