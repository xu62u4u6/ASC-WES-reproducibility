"""S18B/C: original reference-only PCA_CCA with frozen-model score colors.

Source: result3.3.ipynb cells 39–41. PCA_CCA is pymaftools 0.5.0;
scatter rendering remains seaborn, as in the original notebook.
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, Normalize
from pymaftools.utils.reduction import PCA_CCA

from .dataio import read_pivot_sqlite
from .figure_style import TRACK_COLORS, plt, save_tiff_and_preview, setup_plotting
from .model_inputs import load_model_inputs
from .oncoplot_rendering import require_pymaftools_version
from .provenance import file_hash, write_json
from .score_inputs import keyed_matrices, select_score_panels


def draw_panel(frame, *, score, merged=False):
    """Render one unlettered panel; leave all multi-panel layout to assembly."""
    figure = plt.figure(figsize=(8, 7))
    axis = figure.add_axes([0.12, 0.12, 0.68, 0.80])
    if merged:
        axis.set_position([0.12, 0.12, 0.70, 0.80])
        axis.set_box_aspect(1)
    colors = TRACK_COLORS["subtype"]
    cmap = LinearSegmentedColormap.from_list(
        "soft_score", [colors[name] for name in ("LUAD", "ASC", "LUSC")]
    )
    norm = Normalize(0, 1)
    sns.scatterplot(
        data=frame,
        x="CCA_SNV",
        y="CCA_CNV",
        hue="soft_score" if score else "subtype",
        hue_order=None if score else ["LUAD", "ASC", "LUSC"],
        palette=cmap if score else colors,
        hue_norm=norm if score else None,
        legend=False if score else "auto",
        style="subtype" if merged else None,
        style_order=["LUAD", "ASC", "LUSC"] if merged else None,
        markers={"LUAD": "o", "ASC": "^", "LUSC": "s"} if merged else True,
        s=55 if merged else 36,
        edgecolor="white",
        linewidth=0.5,
        alpha=0.7,
        ax=axis,
    )
    if score:
        color_axis = figure.add_axes([0.85 if merged else 0.83, 0.12, 0.025, 0.80])
        figure.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=color_axis,
            label="LUAD–LUSC classification score (0–1)",
            ticks=np.linspace(0, 1, 6),
        )
    else:
        axis.legend(loc="upper left", title="subtype")
        axis.set_ylabel("")
    axis.grid(False)
    if merged:
        axis.grid(True, color="0.8", alpha=0.5, linewidth=0.6)
        axis.set_axisbelow(True)
        from matplotlib.lines import Line2D

        handles = [
            Line2D(
                [],
                [],
                linestyle="none",
                marker=marker,
                color="0.35",
                markersize=7,
                label=name,
            )
            for name, marker in (("LUAD", "o"), ("ASC", "^"), ("LUSC", "s"))
        ]
        axis.legend(handles=handles, title="subtype", loc="upper left")
    return figure, axis


def run(root, model_directory, output, merged=False):
    """Recompute descriptive CCA, replay frozen predictions, and save candidates."""
    require_pymaftools_version()
    if output.exists():
        raise FileExistsError(output)
    decision = json.loads((model_directory / "selection_decision.json").read_text())
    for name, digest in decision["cv_identity"]["inputs"].items():
        assert file_hash(root / name) == digest, name
    model_path = model_directory / "model.joblib"
    digest = file_hash(model_path)
    frozen = json.loads((model_directory / "FROZEN_MODEL.json").read_text())
    assert digest == frozen["sha256"] and not frozen["refit_on_holdout"]
    model = joblib.load(model_path)
    assert model.training_count == 136
    assert model.classes == ("LUAD", "LUSC")
    assert {b.name: len(b.features) for b in model.branches} == {
        "snv_rfe": 45,
        "cnv_rfe": 10,
    }
    data, split = load_model_inputs(root)
    ref_tables = keyed_matrices(data, asc=False)
    asc_tables = keyed_matrices(data, asc=True)
    reference, predicted = model.predict(ref_tables)
    asc_scores, _ = model.predict(asc_tables)
    expected = json.loads((model_directory / "holdout_metrics.json").read_text())
    np.testing.assert_allclose(
        (predicted[split.holdout] == data["y"][split.holdout]).mean(),
        expected["acc"],
        atol=1e-12,
        rtol=0,
    )
    metadata = read_pivot_sqlite(root / "data/WES/all_case_table.db").sample_metadata
    frame, _ = select_score_panels(
        pd.Series(reference, index=data["patients"]),
        pd.Series(asc_scores, index=data["asc_samples"]),
        metadata,
    )
    tables = {}
    for name in ("snv", "cnv"):
        table = pd.concat([ref_tables[name], asc_tables[name]])
        assert table.index.is_unique and table.columns.is_unique
        tables[name] = table.loc[frame.index].T
        assert np.isfinite(tables[name].to_numpy()).all()
    assert tables["snv"].shape == (13977, 200)
    assert tables["cnv"].shape == (114, 200)
    assert np.isin(tables["snv"], [0, 1]).all()
    fit_ids = frame.index[frame.subtype.isin(["LUAD", "LUSC"])]
    assert len(fit_ids) == 171
    cca = PCA_CCA(20, 1, random_state=42)
    cca.fit_transform(*(tables[k].loc[:, fit_ids] for k in ("snv", "cnv")))
    x, y = cca.transform(tables["snv"], tables["cnv"])
    frame["CCA_SNV"] = x[:, 0]
    frame["CCA_CNV"] = y[:, 0]
    assert np.isfinite(frame[["CCA_SNV", "CCA_CNV"]]).all().all()
    output.mkdir(parents=True)
    geometry = {}
    for label, score in (("B", True), ("C", False)):
        figure, axis = draw_panel(frame, score=score, merged=merged and score)
        geometry[label] = save_tiff_and_preview(
            figure, output / f"S18{label}", plot_axis=axis
        )
        plt.close(figure)
    write_json(output / "panel_geometry.json", geometry)
    assert file_hash(model_path) == digest
    write_json(
        output / "configuration.json",
        {
            "status": "candidate; not historical coordinate authentication",
            "source": "result3.3.ipynb cells 39–41",
            "pymaftools": "0.5.0 PCA_CCA; original seaborn scatter",
            "fit_n": 171,
            "project_n": 200,
            "features": [13977, 114],
            "PCA": "20 per modality; auto solver; random_state=42",
            "CNV_standardized_before_PCA": False,
            "CCA_scale": True,
            "model_sha256": digest,
            "classifier_refit": False,
            "reference_correlation": frame.loc[fit_ids, ["CCA_SNV", "CCA_CNV"]]
            .corr()
            .iloc[0, 1],
            "pooled_correlation": frame[["CCA_SNV", "CCA_CNV"]].corr().iloc[0, 1],
            "input_sha256": decision["cv_identity"]["inputs"],
            "interpretation": "descriptive; includes development and holdout samples",
        },
    )


def main():
    """Run with explicit processed input and trusted frozen-model directories."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--merged", action="store_true")
    args = parser.parse_args()
    setup_plotting()
    run(args.data_root, args.model_directory, args.output, args.merged)


if __name__ == "__main__":
    main()
