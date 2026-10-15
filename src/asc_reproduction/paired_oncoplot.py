"""S5A: frequency-ranked top 30 genes across matched A/T/S samples."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from pymaftools import OncoPlot, PivotTable
from pymaftools.plot.Track import FreqTrack

from .cgc_oncoplot import CGCData, mutation_presence
from .cohort_inputs import read_prepared_table, select_paired_metadata
from .oncoplot_rendering import (
    align_sample_label_gutter,
    require_pymaftools_version,
    validate_frequency_alignment,
)
from .figure_style import (
    MUTATION_COLORS,
    default_color_manager,
    plt,
    save_tiff_and_preview,
)

PARTS = ("A", "T", "S")
PALETTE = {
    name: default_color_manager().get_cmap(name)
    for name in ("sample_type", "sex", "smoke")
}


def prepare_paired_oncoplot(values, metadata, cases, *, expected_cases=11):
    """Select original paired top30 genes without changing their frequency."""
    if (
        metadata.index.has_duplicates
        or values.index.has_duplicates
        or values.columns.has_duplicates
        or cases.case_ID.duplicated().any()
    ):
        raise ValueError("Duplicate sample, gene or case keys")
    if set(values.columns) != set(metadata.index):
        raise ValueError("Sample keys differ")
    m = select_paired_metadata(metadata, cases, expected_cases=expected_cases)
    if (
        not m.subtype.eq("ASC").all()
        or not np.isfinite(m.TMB.to_numpy(dtype=float)).all()
    ):
        raise ValueError("Invalid ASC annotations")
    for name, colors in PALETTE.items():
        if not m[name].isin(colors).all():
            raise ValueError("Invalid categorical annotation")
    binary = mutation_presence(values.loc[:, m.index])
    frequencies = binary.mean(axis=1)
    genes = frequencies.sort_values(ascending=False).index[:30]
    if len(genes) != 30:
        raise ValueError("Fewer than 30 supplied genes")
    classes = values.loc[genes, m.index].replace({False: "WT"})
    if set(classes.to_numpy().ravel()) - set(MUTATION_COLORS):
        raise ValueError("Invalid mutation class")
    summary = frequencies.loc[genes].to_frame("freq")
    summary["count"] = binary.loc[genes].sum(axis=1)
    summary.index.name = "gene"
    labels = m.index.astype(str).tolist()
    return CGCData(classes, m, summary), labels


def plot_paired_oncoplot(data, labels):
    """Render original S5A tracks using pymaftools 0.5.0, preserving IDs."""
    require_pymaftools_version()
    if list(data.classes.columns) != list(labels):
        raise ValueError("S5A labels must be original aligned sample IDs")
    table = PivotTable(data.classes.where(data.classes.ne("WT"), False))
    table.sample_metadata = data.metadata.copy()
    table.sample_metadata["smoke"] = table.sample_metadata.smoke.astype(str)
    table = table.add_freq()
    np.testing.assert_allclose(table.feature_metadata.freq, data.gene_summary.freq)
    colors = {False: "white", **MUTATION_COLORS}
    palettes = {name: dict(palette) for name, palette in PALETTE.items()}
    palettes["smoke"] = {str(k): v for k, v in palettes["smoke"].items()}
    plot = OncoPlot(table).set_config(figsize=(20, 13))
    plot.main(cmap_dict=colors, linewidths=0.4)
    plot.add_freq(annot_fontsize=12, xtick_fontsize=12, linewidths=0.4)
    for track in plot.tracks:
        if isinstance(track, FreqTrack):
            track.size = 1.3
    plot.add_bar("TMB", size=1.2, ylabel_size=12)
    plot.add_sample_annotation(
        list(palettes), cmap_dict=palettes, alpha=0.7,
        size=0.7, linewidths=0.4,
    )
    plot.render(
        main_width=28, main_height=25, legend_width=5,
        feature_gutter_size=5, legend_pad=0.1, show_sample_labels=True,
    )
    align_sample_label_gutter(plot)
    plot.add_xticklabel(fontsize=11, rotation=90)
    for axis in plot.fig.axes:
        if [text.get_text() for text in axis.texts] == list(table.index):
            for text in axis.texts:
                text.set_fontsize(13)
    for name, palette in palettes.items():
        plot.legend_manager.legend_dict[name] = {
            key: to_rgba(color, 0.7) for key, color in palette.items()
        }
    plot.plot_all_legends(fontsize=12, title_fontsize=14)
    validate_frequency_alignment(plot)
    return plot.fig


def run(data_root: Path, output: Path) -> None:
    """Generate the existing S5A candidate for matched A/T/S samples."""
    from .figure_style import setup_plotting

    setup_plotting()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output must be new or empty")
    snv = read_prepared_table(data_root, "paired", "SNV")
    cases = pd.read_csv(data_root / "data/ASC_all_case_metadata_0312.csv")
    data, labels = prepare_paired_oncoplot(
        pd.DataFrame(snv), snv.sample_metadata, cases
    )
    output.mkdir(parents=True, exist_ok=True)
    fig = plot_paired_oncoplot(data, labels)
    info = save_tiff_and_preview(fig, output / "S5A_paired_oncoplot")
    plt.close(fig)
    data.gene_summary.to_csv(output / "S5A_gene_summary.csv")
    summary = {
        "panel": "S5A",
        "samples": 33,
        "genes": 30,
        "figure": info,
        "selection": "Top30 by binary mutation frequency across 33 matched A/T/S samples",
        "order": "Case CSV then A/T/S; frequency descending genes; pinned pandas tie behavior",
        "renderer": "pymaftools 0.5.0; metadata alpha=0.7",
        "sample_labels": "Original sample IDs; case CSV then A/T/S",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main():
    """Read CLI options and call the reusable analysis function."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    run(args.data_root, args.output)


if __name__ == "__main__":
    main()
