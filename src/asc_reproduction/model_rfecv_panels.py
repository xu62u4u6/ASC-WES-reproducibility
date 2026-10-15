"""S19B/C from verified final-development diagnostics; never deserialize a model."""

import argparse
import json
from pathlib import Path

import pandas as pd

from .figure_style import plt, save_tiff_and_preview
from .model_checkpoints import validate_curve
from .model_learners import COMBINATIONS
from .model_runner import model_lock
from .model_summary import collect_complete_cv, snapshot_identity
from .provenance import file_hash, runtime_identity, write_json


def verify_final_curves(root, expected, winner):
    if root.is_symlink() or (root / "FAILED.json").exists():
        raise ValueError("Invalid final model directory")
    record = json.loads((root / "COMPLETE.json").read_text())
    required = {
        "selection_decision.json",
        "FROZEN_MODEL.json",
        "model.joblib",
        "final_selection_diagnostics.json",
        "selected_features.json",
        "snv_rfecv_curve.csv",
        "cnv_rfecv_curve.csv",
    }
    inventory = record.get("artifacts", {})
    if (
        record.get("status") != "succeeded"
        or record.get("model") != winner
        or record.get("refit_on_holdout") is not False
    ):
        raise ValueError("Final model does not match complete CV selection")
    if winner not in COMBINATIONS or not required.issubset(inventory):
        raise ValueError("Final model has no complete stacking RFECV diagnostics")
    for name, item in inventory.items():
        path = root / name
        if Path(name).name != name or path.is_symlink() or not path.is_file():
            raise ValueError("Unsafe final artifact")
        if path.stat().st_size != item["bytes"] or file_hash(path) != item["sha256"]:
            raise ValueError("Final artifact changed")
    decision = json.loads((root / "selection_decision.json").read_text())
    frozen = json.loads((root / "FROZEN_MODEL.json").read_text())
    if (
        decision.get("cv_identity") != expected
        or decision.get("model") != winner
        or decision.get("complete_folds") != 500
        or decision.get("candidate_rows") != 4000
        or decision.get("selected_using") != "complete_development_CV_only"
        or decision.get("model_seed") != 42
    ):
        raise ValueError("Final selection provenance differs")
    if (
        frozen.get("training_count") != 136
        or frozen.get("holdout_count") != 35
        or frozen.get("refit_on_holdout") is not False
        or frozen.get("sha256") != inventory["model.joblib"]["sha256"]
        or record.get("model_sha256") != frozen.get("sha256")
    ):
        raise ValueError("Frozen model provenance differs")
    diagnostics = json.loads((root / "final_selection_diagnostics.json").read_text())
    features = json.loads((root / "selected_features.json").read_text())
    branches = list(COMBINATIONS[winner])
    if (
        diagnostics.get("training_count") != 136
        or diagnostics.get("status") != "available"
        or diagnostics.get("selected_branches") != branches
        or set(features) != set(branches)
        or set(diagnostics.get("curves", {})) != {"snv", "cnv"}
    ):
        raise ValueError("Final curve scope differs")
    curves = {}
    for modality in ("snv", "cnv"):
        info = diagnostics["curves"][modality]
        name = f"{modality}_rfecv_curve.csv"
        if info.get("file") != name or any(
            info.get(k) != inventory[name][k] for k in ("bytes", "sha256")
        ):
            raise ValueError("Curve inventory differs")
        curve = pd.read_csv(root / name)
        validate_curve(curve)
        branch = modality + "_rfe"
        used = branch in branches
        n = info["selected_feature_count"]
        if info.get("used_in_selected_model") is not used or n not in set(
            curve.n_features
        ):
            raise ValueError("Curve selection differs")
        if used and (len(features[branch]) != n or len(set(features[branch])) != n):
            raise ValueError("Final selected features differ from curve")
        curves[modality] = curve
    return curves, diagnostics, file_hash(root / "COMPLETE.json")


def plot_curve(curve, modality, info, *, synthetic=False, ax=None):
    validate_curve(curve)
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure
    x = curve.n_features.to_numpy()
    mean = curve.mean_test_score.to_numpy()
    sd = curve.std_test_score.to_numpy()
    ax.plot(
        x,
        mean,
        color="steelblue",
        marker="o",
        markersize=3,
        label="Mean internal CV accuracy",
    )
    ax.fill_between(
        x,
        mean - sd,
        mean + sd,
        color="steelblue",
        alpha=0.2,
        label="± SD (5 RFECV folds)",
    )
    n = info["selected_feature_count"]
    ax.axvline(
        n, color="firebrick", linestyle="--", label=f"RFE-selected features: {n}"
    )
    title = f"{modality.upper()} RFECV — full development cohort"
    if synthetic:
        title = "SYNTHETIC TEST — not study results"
    ax.set(
        xlabel="Number of retained features",
        ylabel="RFECV internal accuracy",
        title=title,
        ylim=(
            min(0, float((mean - sd).min())) - 0.02,
            max(1, float((mean + sd).max())) + 0.02,
        ),
    )
    ax.grid(alpha=0.2)
    ax.legend(loc="lower right", fontsize=9)
    used = (
        "RFE branch used in selected model"
        if info["used_in_selected_model"]
        else "RFE branch NOT used in selected model"
    )
    note = (
        used
        + "\nFull-development feature selection; not holdout performance or a confidence interval"
    )
    if standalone:
        fig.text(0.5, 0.025, note, ha="center", fontsize=8)
        fig.tight_layout(rect=(0, 0.09, 1, 1))
    else:
        note = (
            used
            + "\nFull-development selection, not holdout performance\nShading is SD, not a confidence interval"
        )
        ax.text(
            0.5, -0.21, note, transform=ax.transAxes, ha="center", va="top", fontsize=8
        )
    return fig


def render_rfecv(cv_run, snapshot, data_root, final_run, output):
    with model_lock(cv_run):
        expected = snapshot_identity(snapshot, data_root)
        execution = runtime_identity()
        _, _, ranking = collect_complete_cv(cv_run, expected)
        winner = ranking.iloc[0]["model"]
        curves, diagnostics, digest = verify_final_curves(final_run, expected, winner)
        if (
            snapshot_identity(snapshot, data_root) != expected
            or runtime_identity() != execution
        ):
            raise ValueError("Inputs or source changed before rendering")
        output.mkdir(parents=True, exist_ok=False)
        try:
            figures = {}
            for panel, modality in [("S19B", "snv"), ("S19C", "cnv")]:
                fig = plot_curve(
                    curves[modality], modality, diagnostics["curves"][modality]
                )
                try:
                    figures[panel] = save_tiff_and_preview(fig, output / panel)
                finally:
                    plt.close(fig)
                curves[modality].to_csv(
                    output / f"{panel}_rfecv_values.csv", index=False
                )
            _, _, after = verify_final_curves(final_run, expected, winner)
            if (
                after != digest
                or snapshot_identity(snapshot, data_root) != expected
                or runtime_identity() != execution
            ):
                raise ValueError(
                    "Final model, inputs or source changed during rendering"
                )
            write_json(
                output / "COMPLETE.json",
                {
                    "status": "succeeded",
                    "figures": figures,
                    "scope": "S19B/C, full-development RFECV diagnostics; not holdout performance",
                    "model": winner,
                    "final_manifest_sha256": digest,
                    "diagnostics": diagnostics,
                    "cv_identity": expected,
                    "render_runtime": execution,
                    "artifacts": {
                        p.name: {"sha256": file_hash(p), "bytes": p.stat().st_size}
                        for p in sorted(output.iterdir())
                        if p.is_file()
                    },
                },
            )
        except BaseException as exc:
            write_json(output / "FAILED.json", {"error_type": type(exc).__name__})
            raise


def main():
    from .figure_style import setup_plotting

    setup_plotting()
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("cv-run", "snapshot", "data-root", "final-run", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    render_rfecv(a.cv_run, a.snapshot, a.data_root, a.final_run, a.output)


if __name__ == "__main__":
    main()
