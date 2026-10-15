"""Reproduce matched A/T/S arm and CNV-cluster Friedman screens for S9."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare
from statsmodels.stats.multitest import multipletests

from .dataio import read_pivot_sqlite
from .figure_style import plt, save_tiff_and_preview
from .paired_cnv import PARTS

EXPECTED_CASES = 11
EXPECTED_CLUSTER_COUNT = 114
EXPECTED_SIGNIFICANT_ARMS = ("3q",)
EXPECTED_SIGNIFICANT_CLUSTERS = ("8", "47", "63", "93", "106")
Q_CUTOFF = 0.1

X_AXIS_CANDIDATES = {
    "legacy_mean_pairwise_sum": {
        "label": "Mean within-case sum of 3 absolute pairwise differences",
        "title": "x-axis candidate 1: mean of within-case pairwise sums",
    },
    "mean_over_three_pairs": {
        "label": "Mean absolute difference across cases and 3 pairs",
        "title": "x-axis candidate 2: mean across the 3 pairs",
    },
}


def cluster_mean_matrix(
    gene_matrix: pd.DataFrame,
    cluster_map: pd.DataFrame,
    *,
    expected_cluster_count: int = EXPECTED_CLUSTER_COUNT,
) -> pd.DataFrame:
    """Mean gene-level log2 ratio within each frozen CNV cluster."""

    required = {"feature", "cluster"}
    missing_columns = required - set(cluster_map.columns)
    if missing_columns:
        raise ValueError(
            f"Cluster map is missing columns: {', '.join(sorted(missing_columns))}"
        )

    genes = gene_matrix.copy()
    genes.index = genes.index.astype(str)
    mapping = cluster_map.loc[:, ["feature", "cluster"]].copy()
    mapping["feature"] = mapping["feature"].astype(str)
    if mapping["feature"].duplicated().any():
        raise ValueError("Cluster map contains duplicate gene features")
    labels = mapping.set_index("feature")["cluster"].reindex(genes.index)
    if labels.isna().any():
        raise ValueError(
            f"Cluster map is missing {int(labels.isna().sum())} genes from the CNV matrix"
        )
    labels = labels.astype(str)
    numeric_labels = pd.to_numeric(labels, errors="raise")
    if not np.equal(numeric_labels, np.floor(numeric_labels)).all():
        raise ValueError("Cluster labels must be integer identifiers")

    means = genes.groupby(labels.to_numpy(), sort=False).mean()
    means.index = means.index.astype(str)
    means = means.reindex(sorted(means.index, key=int))
    if len(means) != expected_cluster_count:
        raise ValueError(
            f"Expected {expected_cluster_count} frozen clusters; found {len(means)}"
        )
    return means


def matched_part_matrices(
    matrix: pd.DataFrame,
    sample_metadata: pd.DataFrame,
    *,
    expected_case_count: int = EXPECTED_CASES,
) -> dict[str, np.ndarray]:
    """Return each component matrix with cases aligned and IDs kept in memory only."""

    required = {"case_ID", "sample_type"}
    absent = required - set(sample_metadata.columns)
    if absent:
        raise ValueError(f"Sample metadata is missing: {', '.join(sorted(absent))}")
    metadata = sample_metadata.loc[
        sample_metadata["sample_type"].isin(PARTS)
        & sample_metadata.index.isin(matrix.columns),
        ["case_ID", "sample_type"],
    ].copy()
    if metadata[["case_ID", "sample_type"]].isna().to_numpy().any():
        raise ValueError("Component sample metadata contains missing case/type")
    if metadata.duplicated(["case_ID", "sample_type"]).any():
        raise ValueError("A case has multiple samples for a component type")

    by_case = metadata.reset_index(names="sample").pivot(
        index="case_ID", columns="sample_type", values="sample"
    )
    if not set(PARTS).issubset(by_case.columns):
        raise ValueError("A/T/S complete matched cases could not be identified")
    complete = by_case.dropna(subset=list(PARTS)).sort_index()
    if len(complete) != expected_case_count:
        raise ValueError(
            f"Expected {expected_case_count} complete cases; found {len(complete)}"
        )

    result = {
        part: matrix.loc[:, complete[part].tolist()].to_numpy(dtype=float)
        for part in PARTS
    }
    if any(not np.isfinite(values).all() for values in result.values()):
        raise ValueError("Matched CNV values contain non-finite values")
    return result


def paired_feature_statistics(
    matrix: pd.DataFrame,
    sample_metadata: pd.DataFrame,
    *,
    expected_case_count: int = EXPECTED_CASES,
) -> pd.DataFrame:
    """Return Friedman/BH results, approved three-pair mean and legacy sum."""

    arrays = matched_part_matrices(
        matrix, sample_metadata, expected_case_count=expected_case_count
    )
    p_values = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for feature_index in range(matrix.shape[0]):
            test = friedmanchisquare(
                arrays["A"][feature_index],
                arrays["T"][feature_index],
                arrays["S"][feature_index],
            )
            p_values.append(float(test.pvalue))

    p = np.asarray(p_values, dtype=float)
    q = np.full(p.shape, np.nan, dtype=float)
    finite = np.isfinite(p)
    if finite.any():
        q[finite] = multipletests(p[finite], method="fdr_bh")[1]

    pairwise_sum = (
        np.abs(arrays["A"] - arrays["T"])
        + np.abs(arrays["T"] - arrays["S"])
        + np.abs(arrays["S"] - arrays["A"])
    ).mean(axis=1)
    result = pd.DataFrame(
        {
            "feature": matrix.index.astype(str),
            "p": p,
            "BH_adjusted_p": q,
            "legacy_mean_pairwise_sum": pairwise_sum,
            "mean_over_three_pairs": pairwise_sum / 3.0,
        }
    )
    return result


def make_s9_candidate(result: pd.DataFrame, *, level: str, metric: str) -> plt.Figure:
    """Keep points and fan out labels using the reviewed legacy seed positions."""
    if metric not in X_AXIS_CANDIDATES:
        raise ValueError(f"Unknown S9 x-axis candidate: {metric}")
    valid = result.loc[np.isfinite(result["BH_adjusted_p"])].copy()
    if valid.empty:
        raise ValueError(f"No valid adjusted p-values for {level}")
    q = valid["BH_adjusted_p"].clip(lower=np.finfo(float).tiny)
    y = -np.log10(q)
    significant = valid.loc[valid["BH_adjusted_p"] < Q_CUTOFF].sort_values(metric)

    np.testing.assert_allclose(
        result.mean_over_three_pairs, result.legacy_mean_pairwise_sum / 3
    )
    fig, ax = plt.subplots(figsize=(7, 5.4))
    fig.subplots_adjust(left=0.15, right=0.97, bottom=0.16, top=0.91)
    ax.scatter(
        valid.loc[valid.BH_adjusted_p >= Q_CUTOFF, metric],
        y[valid.BH_adjusted_p >= Q_CUTOFF],
        c="gray",
        s=45,
        alpha=0.6,
        label="ns",
    )
    ax.scatter(
        significant[metric],
        -np.log10(significant.BH_adjusted_p),
        c="red",
        edgecolors="white",
        s=65,
        zorder=3,
        label="BH-adjusted p < 0.1",
    )
    ax.axhline(1.0, color="#ff5555", linestyle="--", linewidth=1)
    seeds = {
        "63": (0.47, 1.19),
        "93": (0.55, 1.30),
        "8": (0.615, 1.20),
        "47": (0.69, 1.30),
        "106": (0.83, 1.20),
    }
    divisor = 3 if metric == "mean_over_three_pairs" else 1
    texts = []
    for index, row in enumerate(significant.itertuples(index=False)):
        px = getattr(row, metric)
        py = -np.log10(max(row.BH_adjusted_p, np.finfo(float).tiny))
        if level == "cluster":
            lx, ly = seeds[str(row.feature)]
            lx /= divisor
        else:
            lx, ly = px - 0.035 / divisor, py + 0.13
        text = ax.annotate(
            str(row.feature),
            (
                getattr(row, metric),
                -np.log10(max(row.BH_adjusted_p, np.finfo(float).tiny)),
            ),
            xytext=(lx, ly),
            ha="center",
            va="center",
            color="#d7191c",
            fontsize=12,
            arrowprops={"arrowstyle": "-", "color": ".4", "linewidth": 0.8},
        )
        texts.append(text)
    ax.set_ylim(-0.1, 1.5)
    ax.set_xlabel(
        "Mean absolute CNV difference (effect size)"
        if divisor == 3
        else X_AXIS_CANDIDATES[metric]["label"]
    )
    ax.set_ylabel("-log10(BH-adjusted p-value)")
    ax.set_title(f"CNV {level.title()} Level")
    ax.grid(False)
    ax.legend(loc="upper right", fontsize=11)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [plt.Text.get_window_extent(text, renderer) for text in texts]
    assert not any(
        a.overlaps(b) for i, a in enumerate(boxes) for b in boxes[i + 1 :]
    ), "S9 labels overlap"
    return fig


def _write_one_level(
    result: pd.DataFrame, *, level: str, output_dir: Path
) -> tuple[list[str], list[dict[str, object]]]:
    result.to_csv(output_dir / f"paired_{level}_statistics.csv", index=False)
    significant = (
        result.loc[result["BH_adjusted_p"] < Q_CUTOFF, "feature"].astype(str).tolist()
    )
    figures = []
    for metric in X_AXIS_CANDIDATES:
        figure = make_s9_candidate(result, level=level, metric=metric)
        output_base = output_dir / f"S9_{level}_{metric}"
        figures.append(
            save_tiff_and_preview(figure, output_base, plot_axis=figure.axes[0])
        )
        plt.close(figure)
    return significant, figures


def run(data_root: Path, output_dir: Path) -> dict[str, object]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty output directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    snv = read_pivot_sqlite(data_root / "data/WES/all_case_table.db")
    arm = read_pivot_sqlite(data_root / "data/CNV_0407/arm_cnv.db")
    gene = read_pivot_sqlite(data_root / "data/CNV_0407/gene_cnv.db")
    if set(arm.values.columns) != set(snv.sample_metadata.index):
        raise ValueError("SNV sample metadata does not cover the arm-CNV matrix")
    if set(gene.values.columns) != set(snv.sample_metadata.index):
        raise ValueError("SNV sample metadata does not cover the gene-CNV matrix")

    cluster_map = pd.read_csv(data_root / "data/clustering/best_k_cluster_labels.csv")
    clusters = cluster_mean_matrix(gene.values, cluster_map)
    results = {
        "arm": paired_feature_statistics(arm.values, snv.sample_metadata),
        "cluster": paired_feature_statistics(clusters, snv.sample_metadata),
    }
    significant = {}
    figure_manifest = []
    for level, result in results.items():
        significant[level], figures = _write_one_level(
            result, level=level, output_dir=output_dir
        )
        figure_manifest.extend(figures)

    if tuple(significant["arm"]) != EXPECTED_SIGNIFICANT_ARMS:
        raise AssertionError(
            f"Expected significant arms {EXPECTED_SIGNIFICANT_ARMS}; got {significant['arm']}"
        )
    if tuple(significant["cluster"]) != EXPECTED_SIGNIFICANT_CLUSTERS:
        raise AssertionError(
            f"Expected significant clusters {EXPECTED_SIGNIFICANT_CLUSTERS}; got {significant['cluster']}"
        )

    summary = {
        "status": "complete",
        "analysis": "matched A/T/S arm and CNV-cluster differences",
        "n_matched_cases": EXPECTED_CASES,
        "test": "Friedman test across A/T/S within case; BH separately within arm and cluster feature families",
        "significance_rule": "BH-adjusted p < 0.1",
        "significant_arms": significant["arm"],
        "significant_clusters": significant["cluster"],
        "x_axis_candidates": list(X_AXIS_CANDIDATES),
        "x_axis_definition_selected": "mean_over_three_pairs",
        "figure_manifest": figure_manifest,
        "patient_or_sample_identifiers_written": False,
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
