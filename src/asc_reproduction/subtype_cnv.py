"""Reproduce subtype-level arm/cluster CNV statistics and S15B/C panels."""

from __future__ import annotations

import argparse
import json
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import kruskal, mannwhitneyu
from statsmodels.stats.multitest import multipletests

from .dataio import read_pivot_sqlite
from .figure_style import plt
from .paired_screen import cluster_mean_matrix

GROUPS = ("LUAD", "ASC", "LUSC")
EXPECTED_GROUP_COUNTS = {"LUAD": 89, "ASC": 29, "LUSC": 82}
EXPECTED_ARM_COUNT = 44
EXPECTED_CLUSTER_COUNT = 114
EXPECTED_SIGNIFICANT_ARMS = 31
EXPECTED_SIGNIFICANT_CLUSTERS = 78
EXPECTED_SHARED_ARMS = 29
Q_CUTOFF = 0.1
TABLE2_TOP_N = 20


def chr_sort_key(arm: str) -> tuple[int, str]:
    """Natural chromosome-arm ordering for the overlap summary."""
    import re

    match = re.match(r"(\d+|X|Y)([pq]?)", str(arm))
    if not match:
        return (999, str(arm))
    chromosome, arm_part = match.groups()
    chromosome_number = (
        int(chromosome) if chromosome.isdigit() else {"X": 23, "Y": 24}[chromosome]
    )
    return chromosome_number, arm_part


def cluster_feature_metadata(
    gene_features: pd.Index,
    gene_metadata: pd.DataFrame,
    cluster_map: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize the arms and gene count represented by each frozen cluster."""
    required_gene = {"Chromosome", "Arm"}
    absent_gene = required_gene - set(gene_metadata.columns)
    if absent_gene:
        raise ValueError(
            f"Gene feature metadata is missing: {', '.join(sorted(absent_gene))}"
        )
    required_map = {"feature", "cluster"}
    absent_map = required_map - set(cluster_map.columns)
    if absent_map:
        raise ValueError(f"Cluster map is missing: {', '.join(sorted(absent_map))}")

    features = pd.Index(gene_features.astype(str), name="feature")
    metadata = gene_metadata.copy()
    metadata.index = metadata.index.astype(str)
    if not set(features).issubset(metadata.index):
        raise ValueError("Gene metadata does not cover the CNV matrix")
    mapping = cluster_map.loc[:, ["feature", "cluster"]].copy()
    mapping["feature"] = mapping["feature"].astype(str)
    if mapping["feature"].duplicated().any():
        raise ValueError("Cluster map contains duplicate genes")
    labels = mapping.set_index("feature")["cluster"].reindex(features)
    if labels.isna().any():
        raise ValueError("Cluster map does not cover all CNV matrix genes")

    gene_arms = metadata.loc[features, "Chromosome"].astype(str) + metadata.loc[
        features, "Arm"
    ].astype(str)
    rows: list[dict[str, object]] = []
    for cluster, members in labels.groupby(labels, sort=False).groups.items():
        member_index = pd.Index(members)
        rows.append(
            {
                "feature": str(cluster),
                "unique_chr_arm": list(pd.unique(gene_arms.reindex(member_index))),
                "features_count": int(len(member_index)),
            }
        )
    result = pd.DataFrame(rows).set_index("feature")
    result = result.reindex(sorted(result.index, key=int))
    return result


def subtype_cnv_statistics(
    matrix: pd.DataFrame,
    sample_metadata: pd.DataFrame,
    *,
    expected_group_counts: dict[str, int] | None = EXPECTED_GROUP_COUNTS,
) -> pd.DataFrame:
    """Kruskal-Wallis and pairwise Mann-Whitney tests, with separate BH families.

    The input matrix is feature-by-sample. Only T samples are analyzed. BH is
    applied across features separately for KW and for each of the three
    pairwise comparisons. Sample identifiers are used only for alignment and
    are never returned in the result.
    """
    required = {"sample_type", "subtype"}
    absent = required - set(sample_metadata.columns)
    if absent:
        raise ValueError(f"Sample metadata is missing: {', '.join(sorted(absent))}")
    if matrix.index.has_duplicates or matrix.columns.has_duplicates:
        raise ValueError("CNV matrix contains duplicate feature or sample keys")
    if not set(matrix.columns).issubset(sample_metadata.index):
        raise ValueError("Sample metadata does not cover the CNV matrix")

    metadata = sample_metadata.loc[matrix.columns]
    metadata = metadata.loc[metadata["sample_type"].eq("T")]
    metadata = metadata.loc[metadata["subtype"].isin(GROUPS)]
    observed = metadata["subtype"].value_counts().to_dict()
    if expected_group_counts is not None and observed != expected_group_counts:
        raise ValueError(
            f"Expected T-sample counts {expected_group_counts}; observed {observed}"
        )
    sample_indices = {
        group: metadata.index[metadata["subtype"].eq(group)].tolist()
        for group in GROUPS
    }
    if any(len(sample_indices[group]) < 2 for group in GROUPS):
        raise ValueError("Each subtype must have at least two T samples")

    records: list[dict[str, float | str]] = []
    comparisons = tuple(combinations(GROUPS, 2))
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        for feature, row in matrix.iterrows():
            values = {
                group: row.loc[sample_indices[group]].to_numpy(dtype=float)
                for group in GROUPS
            }
            if any(
                not np.isfinite(group_values).all() for group_values in values.values()
            ):
                raise ValueError(f"Feature {feature!s} contains a non-finite CNV value")
            kw_stat, kw_p = kruskal(*(values[group] for group in GROUPS))
            record: dict[str, float | str] = {
                "feature": str(feature),
                "KW_stat": float(kw_stat),
                "KW_pval": float(kw_p),
            }
            for first, second in comparisons:
                statistic, p_value = mannwhitneyu(
                    values[first], values[second], alternative="two-sided"
                )
                prefix = f"{first}_vs_{second}"
                record[f"{prefix}_stat"] = float(statistic)
                record[f"{prefix}_pval"] = float(p_value)
            records.append(record)

    result = pd.DataFrame.from_records(records)
    p_columns = [
        "KW_pval",
        *(f"{first}_vs_{second}_pval" for first, second in comparisons),
    ]
    for p_column in p_columns:
        result[p_column.replace("_pval", "_qval")] = multipletests(
            result[p_column].to_numpy(dtype=float), method="fdr_bh"
        )[1]
    return result.sort_values("KW_qval", kind="mergesort").reset_index(drop=True)


def _significant_sets(
    arm_results: pd.DataFrame,
    cluster_results: pd.DataFrame,
    cluster_metadata: pd.DataFrame,
) -> tuple[set[str], set[str], set[str]]:
    significant_arms = set(
        arm_results.loc[arm_results["KW_qval"] < Q_CUTOFF, "feature"].astype(str)
    )
    significant_clusters = set(
        cluster_results.loc[cluster_results["KW_qval"] < Q_CUTOFF, "feature"].astype(
            str
        )
    )
    represented_arms: set[str] = set()
    for cluster in significant_clusters:
        represented_arms.update(cluster_metadata.loc[cluster, "unique_chr_arm"])
    return significant_arms, significant_clusters, represented_arms


def _table2_candidate(
    ordered_results: pd.DataFrame,
    cluster_metadata: pd.DataFrame,
    *,
    include_cutoff_ties: bool = False,
) -> pd.DataFrame:
    significant = ordered_results.loc[ordered_results["KW_qval"] < Q_CUTOFF].copy()
    if len(significant) < TABLE2_TOP_N:
        raise ValueError(f"Expected at least {TABLE2_TOP_N} significant clusters")
    if include_cutoff_ties:
        cutoff_q = float(significant.iloc[TABLE2_TOP_N - 1]["KW_qval"])
        selected = significant.loc[significant["KW_qval"] <= cutoff_q]
    else:
        selected = significant.head(TABLE2_TOP_N)
    table = selected.merge(
        cluster_metadata, left_on="feature", right_index=True, how="left"
    )
    table = table.loc[
        :, ["feature", "unique_chr_arm", "features_count", "KW_qval"]
    ].copy()
    table["unique_chr_arm"] = table["unique_chr_arm"].map(
        lambda arms: str(np.asarray(arms, dtype=object))
    )
    return table


def make_table2_tie_candidates(
    cluster_results: pd.DataFrame,
    cluster_metadata: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    """Make explicitly non-final Table 2 variants for the tied twentieth rank."""
    stable = cluster_results.sort_values("KW_qval", kind="mergesort").reset_index(
        drop=True
    )
    numeric_feature_order = (
        cluster_results.assign(
            _numeric_feature=pd.to_numeric(cluster_results["feature"], errors="raise")
        )
        .sort_values("_numeric_feature", kind="mergesort")
        .drop(columns="_numeric_feature")
    )
    legacy = numeric_feature_order.sort_values("KW_qval", kind="quicksort").reset_index(
        drop=True
    )

    stable_top = _table2_candidate(stable, cluster_metadata)
    legacy_top = _table2_candidate(legacy, cluster_metadata)
    include_ties = _table2_candidate(stable, cluster_metadata, include_cutoff_ties=True)
    cutoff_q = float(stable_top.iloc[-1]["KW_qval"])
    tied = stable.loc[stable["KW_qval"].eq(cutoff_q), "feature"].astype(str).tolist()
    candidates = {
        "legacy_order": legacy_top,
        "stable_feature_order": stable_top,
        "include_cutoff_ties": include_ties,
    }
    audit = {
        "status": "pending_user_choice",
        "canonical_table_written": False,
        "requested_rows_before_ties": TABLE2_TOP_N,
        "twentieth_rank_q_value": cutoff_q,
        "clusters_tied_at_twentieth_rank": tied,
        "candidate_rows": {key: len(value) for key, value in candidates.items()},
        "candidate_files": {
            "legacy_order": "Supplementary_Table_2_candidate_legacy_order.csv",
            "stable_feature_order": "Supplementary_Table_2_candidate_stable_feature_order.csv",
            "include_cutoff_ties": "Supplementary_Table_2_candidate_include_cutoff_ties.csv",
        },
        "legacy_ordering_note": (
            "Reproduces the prior pinned-pandas default quicksort order from numeric cluster order; "
            "the tied cutoff itself has no unique biological ranking."
        ),
    }
    return candidates, audit


def make_arm_heatmap(
    arm_matrix: pd.DataFrame, sample_metadata: pd.DataFrame
) -> plt.Figure:
    metadata = sample_metadata.loc[arm_matrix.columns]
    metadata = metadata.loc[metadata["sample_type"].eq("T")]
    sample_order: list[str] = []
    group_boundaries = []
    for group in GROUPS:
        group_samples = sorted(metadata.index[metadata["subtype"].eq(group)])
        sample_order.extend(group_samples)
        group_boundaries.append(len(sample_order))
    if len(sample_order) != sum(EXPECTED_GROUP_COUNTS.values()):
        raise ValueError("S15B heatmap requires exactly the 200 reference T samples")

    arm_order = sorted(arm_matrix.index.astype(str), key=chr_sort_key)
    matrix = arm_matrix.copy()
    matrix.index = matrix.index.astype(str)
    values = matrix.loc[arm_order, sample_order].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(10, 9))
    image = ax.imshow(values, aspect="auto", cmap="coolwarm", vmin=-2, vmax=2)
    ax.set_yticks(range(len(arm_order)), arm_order)
    ax.tick_params(axis="y", labelsize=8)
    boundaries = np.cumsum([0, *[EXPECTED_GROUP_COUNTS[group] for group in GROUPS]])
    centers = [
        boundaries[i] + (boundaries[i + 1] - boundaries[i] - 1) / 2
        for i in range(len(GROUPS))
    ]
    labels = [f"{group} ({EXPECTED_GROUP_COUNTS[group]})" for group in GROUPS]
    ax.set_xticks(centers, labels)
    for boundary in group_boundaries[:-1]:
        ax.axvline(boundary - 0.5, color="black", linewidth=0.7)
    ax.set_title("Arm-level CNV profiles")
    fig.colorbar(image, ax=ax, label="CNV log2 ratio")
    return fig


def make_overlap_venn(
    significant_arms: set[str], represented_arms: set[str]
) -> plt.Figure:
    from matplotlib_venn import venn2

    fig, ax = plt.subplots(figsize=(7, 5))
    venn2(
        [significant_arms, represented_arms],
        set_labels=("Significant arms", "Arms represented by\nsignificant clusters"),
        ax=ax,
    )
    ax.set_title("Related CNV summaries")
    return fig


def save_fixed_canvas(fig: plt.Figure, output_base: Path) -> dict[str, object]:
    """Write the same full-canvas dimensions as the legacy S15 figure saver."""
    png_path = Path(f"{output_base}.png")
    tiff_path = Path(f"{output_base}.tiff")
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
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
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty output directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = read_pivot_sqlite(data_root / "data/WES/all_case_table.db")
    arms = read_pivot_sqlite(data_root / "data/CNV_0407/arm_cnv.db")
    genes = read_pivot_sqlite(data_root / "data/CNV_0407/gene_cnv.db")
    if set(arms.values.columns) != set(metadata.sample_metadata.index):
        raise ValueError("SNV sample metadata does not cover the arm-CNV matrix")
    if set(genes.values.columns) != set(metadata.sample_metadata.index):
        raise ValueError("SNV sample metadata does not cover the gene-CNV matrix")

    cluster_map = pd.read_csv(data_root / "data/clustering/best_k_cluster_labels.csv")
    clusters = cluster_mean_matrix(genes.values, cluster_map)
    cluster_metadata = cluster_feature_metadata(
        genes.values.index, genes.feature_metadata, cluster_map
    )
    arm_results = subtype_cnv_statistics(arms.values, metadata.sample_metadata)
    cluster_results = subtype_cnv_statistics(clusters, metadata.sample_metadata)
    if len(arms.values.index) != EXPECTED_ARM_COUNT:
        raise AssertionError(
            f"Expected {EXPECTED_ARM_COUNT} arms; got {len(arms.values.index)}"
        )
    if len(clusters.index) != EXPECTED_CLUSTER_COUNT:
        raise AssertionError(
            f"Expected {EXPECTED_CLUSTER_COUNT} clusters; got {len(clusters.index)}"
        )

    significant_arms, significant_clusters, represented_arms = _significant_sets(
        arm_results, cluster_results, cluster_metadata
    )
    if len(significant_arms) != EXPECTED_SIGNIFICANT_ARMS:
        raise AssertionError(
            f"Expected {EXPECTED_SIGNIFICANT_ARMS} significant arms; got {len(significant_arms)}"
        )
    if len(significant_clusters) != EXPECTED_SIGNIFICANT_CLUSTERS:
        raise AssertionError(
            f"Expected {EXPECTED_SIGNIFICANT_CLUSTERS} significant clusters; "
            f"got {len(significant_clusters)}"
        )
    shared_arms = significant_arms & represented_arms
    if len(shared_arms) != EXPECTED_SHARED_ARMS:
        raise AssertionError(
            f"Expected {EXPECTED_SHARED_ARMS} shared arm representations; got {len(shared_arms)}"
        )

    arm_results.to_csv(output_dir / "arm_all_tests.csv", index=False)
    cluster_results.to_csv(output_dir / "cluster_all_tests.csv", index=False)
    significant_arm_table = arm_results.loc[arm_results["KW_qval"] < Q_CUTOFF].copy()
    significant_arm_table.to_csv(output_dir / "Supplementary_Table_1.csv", index=False)
    table2_candidates, table2_audit = make_table2_tie_candidates(
        cluster_results, cluster_metadata
    )
    for candidate_name, candidate_table in table2_candidates.items():
        candidate_table.to_csv(
            output_dir / table2_audit["candidate_files"][candidate_name], index=False
        )
    (output_dir / "Supplementary_Table_2_tie_audit.json").write_text(
        json.dumps(table2_audit, indent=2) + "\n", encoding="utf-8"
    )

    represented_order = sorted(represented_arms, key=chr_sort_key)
    significant_order = sorted(significant_arms, key=chr_sort_key)
    figure_manifest = []
    for name, figure in (
        ("S15B_arm_heatmap", make_arm_heatmap(arms.values, metadata.sample_metadata)),
        ("S15C_CNV_overlap", make_overlap_venn(significant_arms, represented_arms)),
    ):
        figure_manifest.append(save_fixed_canvas(figure, output_dir / name))
        plt.close(figure)

    summary = {
        "status": "complete",
        "analysis": "subtype-level CNV tests and cluster/arm overlap",
        "sample_type": "T",
        "subtype_counts": EXPECTED_GROUP_COUNTS,
        "arm_feature_count": len(arms.values.index),
        "cluster_feature_count": len(clusters.index),
        "test": "Kruskal-Wallis; pairwise two-sided Mann-Whitney U",
        "correction": "Benjamini-Hochberg separately across features for KW and each pairwise family",
        "significance_rule": "BH-adjusted p < 0.1",
        "significant_arm_count": len(significant_arms),
        "significant_cluster_count": len(significant_clusters),
        "significant_cluster_represented_arm_count": len(represented_arms),
        "shared_arm_count": len(shared_arms),
        "shared_arms": sorted(shared_arms, key=chr_sort_key),
        "only_arm_level_significant": sorted(
            significant_arms - represented_arms, key=chr_sort_key
        ),
        "only_represented_by_significant_cluster": sorted(
            represented_arms - significant_arms, key=chr_sort_key
        ),
        "arm_level_significant_arms": significant_order,
        "cluster_represented_arms": represented_order,
        "supplementary_table_2": table2_audit,
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
