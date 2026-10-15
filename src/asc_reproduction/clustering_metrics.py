"""Render saved five-fold clustering metrics; never recompute clustering."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .figure_style import plt, save_tiff_and_preview

FOLDS = [f"fold{i}_silhouette" for i in range(1, 6)]


def summarize_metrics(metrics, labels):
    if not set(FOLDS).issubset(metrics) or metrics.empty or not metrics.index.is_unique:
        raise ValueError("Missing metrics or duplicate k")
    k = np.asarray(metrics.index, dtype=float)
    values = metrics[FOLDS].to_numpy(dtype=float)
    if (
        not np.isfinite(k).all()
        or (k < 2).any()
        or (k != k.astype(int)).any()
        or not np.isfinite(values).all()
        or (np.abs(values) > 1).any()
    ):
        raise ValueError("Invalid k or silhouette values")
    if (
        "cluster" not in labels
        or labels.empty
        or not labels.index.is_unique
        or labels["cluster"].isna().any()
    ):
        raise ValueError("Invalid frozen cluster labels")
    result = pd.DataFrame(
        {
            "mean_silhouette": metrics[FOLDS].mean(axis=1),
            "sd_silhouette": metrics[FOLDS].std(axis=1, ddof=1),
        }
    )
    result.index.name = "k"
    best = int(result.mean_silhouette.idxmax())
    if labels.cluster.nunique() != best:
        raise ValueError("Selected k and frozen cluster count disagree")
    return result, {
        "best_k": best,
        "n_genes": len(labels),
        "n_clusters": int(labels.cluster.nunique()),
        "evaluated_k_count": len(result),
        "k_min": int(k.min()),
        "k_max": int(k.max()),
        "best_mean": float(result.loc[best, "mean_silhouette"]),
        "best_sd": float(result.loc[best, "sd_silhouette"]),
        "tied_maxima": [
            int(x)
            for x in result.index[
                result.mean_silhouette.eq(result.mean_silhouette.max())
            ]
        ],
        "selection": "maximum five-fold mean; first file-order k if tied",
        "uncertainty": "sample standard deviation of five saved folds, not confidence interval",
    }


def main():
    from .figure_style import setup_plotting

    setup_plotting()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {
        n: args.data_root / "data/clustering" / n
        for n in ("metric_df.csv", "best_k_cluster_labels.csv")
    }
    hashes = {n: hashlib.sha256(p.read_bytes()).hexdigest() for n, p in paths.items()}
    metrics = pd.read_csv(paths["metric_df.csv"], index_col=0)
    labels = pd.read_csv(paths["best_k_cluster_labels.csv"], index_col=0)
    result, report = summarize_metrics(metrics, labels)
    if (report["best_k"], report["n_genes"]) != (114, 25988):
        raise ValueError("Frozen study clustering anchors differ")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Output must be new or empty")
    args.output.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        result.index,
        result.mean_silhouette,
        color="#1f77b4",
        label="Mean silhouette (5-fold)",
    )
    ax.errorbar(
        result.index,
        result.mean_silhouette,
        yerr=result.sd_silhouette,
        fmt="none",
        capsize=2,
        capthick=1,
        color="#bbbbbb",
        alpha=0.6,
        label="5-fold mean ± SD",
    )
    ax.axvline(
        report["best_k"],
        color="red",
        linestyle=":",
        alpha=0.6,
        label=f"Best k={report['best_k']}",
    )
    ax.set(xlabel="Number of clusters (k)", ylabel="Silhouette score")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    report["figure"] = save_tiff_and_preview(fig, args.output / "S1_clustering_metrics")
    plt.close(fig)
    result.to_csv(args.output / "S1_saved_fold_summary.csv")
    if hashes != {
        n: hashlib.sha256(p.read_bytes()).hexdigest() for n, p in paths.items()
    }:
        raise ValueError("Clustering inputs changed during rendering")
    report.update(
        input_sha256=hashes,
        scope="Plot from supplied processed fold metrics and frozen labels; no clustering or silhouette refit",
        limitations="Exact upstream metric-generation provenance and manuscript panel coverage still require reconciliation",
    )
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
