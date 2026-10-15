"""One pooled, CNV-standardized CCA fit for S18A and Tables 3/4."""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler

from .cgc_oncoplot import group_mutation_order, mutation_presence
from .dataio import read_pivot_sqlite
from .figure_style import TRACK_COLORS, plt, save_tiff_and_preview
from .paired_screen import cluster_mean_matrix


@dataclass(frozen=True)
class PooledCCAResult:
    correlation: float
    scores: pd.DataFrame
    snv_weights: pd.Series
    cnv_weights: pd.Series
    cnv_standardization: pd.DataFrame
    iterations: int


def fit_pooled_cca(
    snv: pd.DataFrame, cnv: pd.DataFrame, *, pcs: int = 20
) -> PooledCCAResult:
    if (
        snv.columns.has_duplicates
        or cnv.columns.has_duplicates
        or snv.index.has_duplicates
        or cnv.index.has_duplicates
    ):
        raise ValueError("CCA requires unique feature and sample keys")
    if set(snv.columns) != set(cnv.columns):
        raise ValueError("SNV and CNV sample sets differ")
    cnv = cnv.loc[:, snv.columns]
    x, y = snv.T.to_numpy(dtype=float), cnv.T.to_numpy(dtype=float)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("CCA inputs contain non-finite values")
    if not np.isin(x, [0, 1]).all():
        raise ValueError("SNV input must be binary")
    if pcs < 1 or pcs > min(x.shape[0] - 1, x.shape[1], y.shape[1]):
        raise ValueError("Too few samples/features for requested PCA dimensions")
    scaler = StandardScaler()
    ys = scaler.fit_transform(y)
    px = PCA(n_components=pcs, svd_solver="full", random_state=0)
    py = PCA(n_components=pcs, svd_solver="full", random_state=0)
    xp, yp = px.fit_transform(x), py.fit_transform(ys)
    cca = CCA(n_components=1, scale=False, max_iter=500, tol=1e-6)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        u, v = cca.fit_transform(xp, yp)
    wx = pd.Series(
        px.components_.T @ cca.x_weights_[:, 0],
        index=snv.index.astype(str),
        name="weight",
    )
    wy = pd.Series(
        py.components_.T @ cca.y_weights_[:, 0],
        index=cnv.index.astype(str),
        name="weight",
    )
    sign = -1 if "EGFR" in wx.index and wx.loc["EGFR"] < 0 else 1
    wx *= sign
    wy *= sign
    scores = pd.DataFrame(
        {"snv_c": u[:, 0] * sign, "cnv_c": v[:, 0] * sign}, index=snv.columns
    )
    if not np.isfinite(scores.to_numpy()).all():
        raise ValueError("CCA produced non-finite scores")
    correlation = float(scores.corr().iloc[0, 1])
    if not np.isfinite(correlation):
        raise ValueError("CCA correlation is undefined")
    standardization = pd.DataFrame(
        {"mean": scaler.mean_, "scale": scaler.scale_}, index=cnv.index.astype(str)
    )
    return PooledCCAResult(
        correlation, scores, wx, wy, standardization, int(cca.n_iter_[0])
    )


def ranked_weights(weights: pd.Series) -> pd.DataFrame:
    ranked = weights.loc[weights.abs().sort_values(ascending=False).index].to_frame()
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    return ranked


def make_cca_figure(
    result: PooledCCAResult, metadata: pd.DataFrame, *, square=False
) -> plt.Figure:
    """Render pooled CCA; optionally match the merged S18 square panel geometry."""
    if square:
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_axes([0.12, 0.12, 0.70, 0.80])
        ax.set_box_aspect(1)
    else:
        fig, ax = plt.subplots(figsize=(7, 5))
    groups = metadata.loc[result.scores.index, "subtype"]
    for name, color in TRACK_COLORS["subtype"].items():
        points = result.scores.loc[groups.eq(name)]
        ax.scatter(
            points.snv_c,
            points.cnv_c,
            color=color,
            label=name,
            s=28,
            edgecolors="white",
            linewidths=0.35,
        )
    ax.set(
        xlabel="Canonical var 1 (SNV)",
        ylabel="Canonical var 1 (CNV)",
        title=f"CCA: SNV vs CNV-cluster (r1={result.correlation:.3f})",
    )
    ax.grid(True, color="0.8", alpha=0.5, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(title="subtype", loc="upper left" if square else "best")
    return fig


def main() -> None:
    from .figure_style import setup_plotting

    setup_plotting()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--square", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Output directory must be new or empty")
    root = args.data_root / "data"
    snv = read_pivot_sqlite(root / "WES/all_case_table.db")
    genes = read_pivot_sqlite(root / "CNV_0407/gene_cnv.db")
    binary = mutation_presence(snv.values)
    order = group_mutation_order(binary, snv.sample_metadata)
    meta = snv.sample_metadata.loc[order]
    meta = meta.loc[meta.sample_type.eq("T")]
    if meta.subtype.value_counts().to_dict() != {"LUAD": 89, "ASC": 29, "LUSC": 82}:
        raise ValueError("Unexpected pooled T cohort")
    binary = binary.loc[:, meta.index]
    binary = binary.loc[binary.mean(axis=1).ge(0.05)]
    if len(binary) != 441:
        raise ValueError("Expected 441 SNV genes at >=5%")
    clusters = cluster_mean_matrix(
        genes.values, pd.read_csv(root / "clustering/best_k_cluster_labels.csv")
    )
    result = fit_pooled_cca(binary, clusters.loc[:, meta.index])
    if not np.isclose(result.correlation, 0.7586780177215268, atol=1e-9, rtol=0):
        raise ValueError(
            f"Pooled correlation differs from verified anchor: {result.correlation}"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    for number, modality, weights in (
        (3, "snv", result.snv_weights),
        (4, "cnv", result.cnv_weights),
    ):
        table = ranked_weights(weights)
        table.to_csv(
            args.output / f"pooled_CCA_{modality}_weights_all.csv",
            index_label="feature",
        )
        table.head(10).to_csv(
            args.output / f"Supplementary_Table_{number}.csv", index_label="feature"
        )
    result.cnv_standardization.to_csv(
        args.output / "CNV_standardization_parameters.csv", index_label="cluster"
    )
    figure = make_cca_figure(result, meta, square=args.square)
    manifest = save_tiff_and_preview(
        figure, args.output / "S18A_pooled_CCA_CNV_standardized"
    )
    plt.close(figure)
    summary = {
        "correlation": result.correlation,
        "sample_counts": {"LUAD": 89, "ASC": 29, "LUSC": 82},
        "SNV_features": 441,
        "CNV_clusters": 114,
        "SNV_frequency": ">=0.05",
        "preprocessing": "binary SNV; pooled StandardScaler for CNV; full PCA20 each",
        "CCA": {
            "components": 1,
            "scale": False,
            "max_iter": 500,
            "tol": 1e-6,
            "iterations": result.iterations,
        },
        "sign": "EGFR SNV weight nonnegative",
        "table_ranking": "descending absolute back-projected weight",
        "CNV_coefficient_units": "standardized cluster units; not raw log2-ratio units",
        "same_fit_for_figure_and_tables": True,
        "sample_scores_or_identifiers_written": False,
        "reference_only_CCA_changed": False,
        "figure": manifest,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
