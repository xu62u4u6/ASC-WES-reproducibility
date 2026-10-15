"""Shared pymaftools 0.5.0 adapters for the manuscript oncoplots.

Figure 4A follows ``plot_fig5_cgc_oncoplot`` in the original
``reproduce_submission/analysis/polish_selected_figures.py``. The old chained
plotting API is translated to the 0.5.0 track API, not to a separate renderer.
Validated selection, sample order, frequency definitions and colors are kept.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pymaftools
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from pymaftools import OncoPlot, PivotTable
from pymaftools.plot.Track import FreqTrack

from .figure_style import patient_palette

if TYPE_CHECKING:
    from .cgc_oncoplot import CGCData


FIGURE_2A_METADATA_ALPHA = 0.7


def require_pymaftools_version() -> None:
    """Reject an unverified runtime before using the plotting API."""
    if pymaftools.__version__ != "0.5.0":
        raise RuntimeError("Manuscript rendering requires pymaftools 0.5.0")


def validate_frequency_alignment(plot: OncoPlot) -> None:
    """Reject overflowing frequency text or rows displaced from the matrix."""
    plot.fig.canvas.draw()
    renderer = plot.fig.canvas.get_renderer()
    main = plot.ax_heatmap.get_window_extent(renderer)
    axes = [
        axis for axis in plot.fig.axes
        if [label.get_text() for label in axis.get_xticklabels()] == ["freq"]
    ]
    if len(axes) != 1:
        raise ValueError("Expected exactly one frequency column")
    axis = axes[0]
    bounds = axis.get_window_extent(renderer)
    np.testing.assert_allclose(
        [bounds.y0, bounds.y1], [main.y0, main.y1], atol=0.5
    )
    for row, text in enumerate(axis.texts):
        box = text.get_window_extent(renderer)
        if box.x0 < bounds.x0 + 1 or box.x1 > bounds.x1 - 1:
            raise ValueError("Frequency annotation extends beyond its cell")
        expected_y = plot.ax_heatmap.transData.transform((0.5, row + 0.5))[1]
        actual_y = axis.transData.transform(text.get_position())[1]
        np.testing.assert_allclose(actual_y, expected_y, atol=0.5)


def align_sample_label_gutter(plot: OncoPlot) -> None:
    """Work around the 0.5.0 ungrouped sample-label half-cell offset.

    Heatmap cells and label anchors use i + 0.5, but the package label
    gutter uses xlim=(-0.5, n-0.5). Match the heatmap limits instead.
    Do not mutate the package, label strings, or sample ordering. This
    adapter deliberately rejects grouped layouts; review on upgrade.
    """
    require_pymaftools_version()
    gutters = plot._sample_gutter_axes
    if len(gutters) != 1:
        raise ValueError("Expected one ungrouped sample-label gutter")
    gutter = gutters[0]
    count = plot.pivot_table.shape[1]
    if len(gutter.texts) != count:
        raise ValueError("Sample-label count differs from matrix columns")
    np.testing.assert_allclose(plot.ax_heatmap.get_xlim(), (0, count))
    np.testing.assert_allclose(
        [text.get_position()[0] for text in gutter.texts],
        np.arange(count) + 0.5,
    )
    gutter.set_xlim(plot.ax_heatmap.get_xlim())


def matched_patient_annotation(metadata: pd.DataFrame) -> tuple[pd.Series, dict]:
    """Give each matched ASC case its registered color across A/T/S samples.

    Patient labels remain anonymous ordinal strings, but colors are looked up
    by case ID, independently of first-appearance order. The Figure 2A
    renderer applies its own shared metadata opacity.
    """
    required = ("case_ID", "sample_type")
    if not set(required).issubset(metadata.columns):
        raise ValueError("Matched patient annotation requires case and type")
    if metadata[list(required)].isna().to_numpy().any():
        raise ValueError("Matched patient annotation contains missing values")
    cases = metadata.case_ID.drop_duplicates().tolist()
    if len(cases) != 11 or len(metadata) != 33:
        raise ValueError("Figure 2A requires 11 matched A/T/S triplets")
    for _, group in metadata.groupby("case_ID", sort=False):
        if len(group) != 3 or set(group.sample_type) != {"A", "T", "S"}:
            raise ValueError("Figure 2A has an incomplete A/T/S patient triplet")
    ordinal = {case: str(index + 1) for index, case in enumerate(cases)}
    registered = patient_palette(cases)
    palette = {ordinal[case]: registered[case] for case in cases}
    return metadata.case_ID.map(ordinal), palette


def render_driver_oncoplot(data: CGCData, labels: list[str]) -> Figure:
    """Render Figure 2A variants using package tracks, without a TMB bar.

    Selection remains the original three prespecified genes, not a newly
    imposed shared-variant filter. Manuscript sample IDs preserve column order.
    """
    from .figure_style import MUTATION_COLORS
    from .paired_oncoplot import PALETTE

    require_pymaftools_version()
    if not data.classes.columns.equals(data.metadata.index):
        raise ValueError("Driver variant columns and metadata are not aligned")
    if len(labels) != data.classes.shape[1] or len(set(labels)) != len(labels):
        raise ValueError("Driver sample labels must be unique and complete")
    table = PivotTable(data.classes.where(data.classes.ne("WT"), False))
    table.sample_metadata = data.metadata.copy()
    table = table.add_freq()
    np.testing.assert_allclose(
        table.feature_metadata.freq, data.gene_summary.freq, rtol=0, atol=0
    )
    table.columns = labels
    table.sample_metadata.index = labels
    # Binary smoking status is categorical, not a continuous numeric track.
    table.sample_metadata["smoke"] = table.sample_metadata.smoke.astype(str)
    patient_labels, patient_colors = matched_patient_annotation(data.metadata)
    table.sample_metadata["patient"] = patient_labels.to_numpy()
    colors = {
        False: "white",
        **{key: value for key, value in MUTATION_COLORS.items() if key != "WT"},
    }
    plot = OncoPlot(table).set_config(figsize=(20, 12), ytick_fontsize=14)
    plot.main(cmap_dict=colors, linewidths=0.5).add_freq(
        annot_fontsize=13, xtick_fontsize=14
    )
    plot.add_sample_annotation(
        ["patient"],
        cmap_dict={"patient": patient_colors},
        alpha=FIGURE_2A_METADATA_ALPHA,
        size=0.7,
        linewidths=0.5,
    )
    for name, palette in PALETTE.items():
        if name == "smoke":
            palette = {str(key): color for key, color in palette.items()}
        plot.add_sample_annotation(
            [name],
            cmap_dict={name: palette},
            alpha=FIGURE_2A_METADATA_ALPHA,
            size=0.7,
            linewidths=0.5,
        )
    plot.render(
        main_width=28,
        main_height=15,
        legend_width=5,
        feature_gutter_size=7.5,
        legend_pad=0.1,
        show_sample_labels=True,
    )
    align_sample_label_gutter(plot)
    plot.add_xticklabel(fontsize=10, rotation=90)
    # pymaftools 0.5.0 draws feature names in a separate gutter with a fixed
    # 8-point font; set this panel's 15 names explicitly for manuscript size.
    feature_axes = [
        axis
        for axis in plot.fig.axes
        if [text.get_text() for text in axis.texts] == list(data.classes.index)
    ]
    if len(feature_axes) != 1:
        raise ValueError("Expected one complete driver feature-label gutter")
    for text in feature_axes[0].texts:
        text.set_fontsize(14)
    # S7 also presents patient identity as a strip without an 11-item legend.
    # Removing only that legend keeps the existing mutation/metadata legends.
    plot.legend_manager.legend_dict.pop("patient")
    for name, palette in PALETTE.items():
        plot.legend_manager.legend_dict[name] = {
            str(key) if name == "smoke" else key: to_rgba(color, 0.7)
            for key, color in palette.items()
        }
    plot.plot_all_legends(fontsize=14, title_fontsize=16)
    return plot.fig


def build_cgc_plot_table(data: CGCData) -> PivotTable:
    """Convert validated Figure 4A data to the package's mutation table.

    WT is a display token in the existing processed-input reader. pymaftools
    represents absent events with boolean False. Frequencies are recomputed by
    the package and checked against the saved analysis definitions before any
    plotting. Patient identifiers remain in memory only.
    """
    from .cgc_oncoplot import GROUPS

    require_pymaftools_version()
    if not data.classes.columns.equals(data.metadata.index):
        raise ValueError("Oncoplot sample order differs from metadata")
    if not data.classes.index.equals(data.gene_summary.index):
        raise ValueError("Oncoplot gene order differs from frequency metadata")
    if data.classes.index.has_duplicates or data.classes.columns.has_duplicates:
        raise ValueError("Oncoplot requires unique genes and samples")
    decoded = data.classes.where(data.classes.ne("WT"), False)
    table = PivotTable(decoded.copy())
    table.sample_metadata = data.metadata.copy()
    groups = {group: table.loc[:, data.metadata.subtype.eq(group)] for group in GROUPS}
    table = table.add_freq(groups=groups)
    columns = ["freq", *(f"{group}_freq" for group in GROUPS)]
    np.testing.assert_allclose(
        table.feature_metadata[columns],
        data.gene_summary[columns],
        rtol=0,
        atol=0,
        err_msg="pymaftools frequencies differ from the confirmed analysis",
    )
    pd.testing.assert_frame_equal(pd.DataFrame(table), decoded)
    table.feature_metadata["CGC role"] = data.gene_summary.CGC_role
    for group in GROUPS:
        table.feature_metadata[group] = table.feature_metadata[f"{group}_freq"]
    # Keep binary smoking status categorical in the track API.
    table.sample_metadata["smoke"] = table.sample_metadata.smoke.astype(str)
    return table


def render_frequency_oncoplot(
    data: CGCData,
    *,
    smoking: bool = False,
    cell_linewidth: float = 0.5,
    annotation_linewidth: float = 1.0,
    legend_width: float = 6.0,
    legend_fontsize: float = 12.0,
    legend_title_fontsize: float = 14.0,
    figsize: tuple[float, float] = (24, 18),
    main_width: float = 28.0,
    preserve_frequency_names: bool = False,
    show_sample_labels: bool = False,
) -> Figure:
    """Render S11/S13 with package tracks, preserving selected rows and order."""
    from .figure_style import MUTATION_COLORS, TRACK_COLORS

    require_pymaftools_version()
    if not data.classes.columns.equals(data.metadata.index):
        raise ValueError("Mutation columns and metadata differ")
    table = PivotTable(data.classes.where(data.classes.ne("WT"), False))
    table.sample_metadata = data.metadata.copy()
    table.sample_metadata["smoke"] = table.sample_metadata.smoke.astype(str)
    frequency_names = (
        {"freq": "All ASC", "smoker_freq": "Smoker", "non-smoker_freq": "Non-smoker"}
        if smoking
        else {"LUAD_freq": "LUAD", "ASC_freq": "ASC", "LUSC_freq": "LUSC"}
    )
    if preserve_frequency_names:
        frequency_names = {name: name for name in frequency_names}
    frequencies = data.gene_summary[list(frequency_names)].rename(
        columns=frequency_names
    )
    np.testing.assert_allclose(
        data.classes.ne("WT").mean(axis=1), data.gene_summary.freq
    )
    tracks = ["smoke"] if smoking else list(TRACK_COLORS)
    palettes = {name: TRACK_COLORS[name] for name in tracks}
    palettes["smoke"] = {str(k): v for k, v in TRACK_COLORS["smoke"].items()}
    colors = {False: "white", **{k: v for k, v in MUTATION_COLORS.items() if k != "WT"}}
    plot = OncoPlot(table).set_config(figsize=figsize, ytick_fontsize=12)
    plot.main(cmap_dict=colors, linewidths=cell_linewidth)
    plot.add_bar("TMB", size=1.5, ylabel_size=12)
    frequency_track = FreqTrack(frequencies, annot_fontsize=10, xtick_fontsize=12)
    frequency_track.size = 4.5
    plot.add_track(frequency_track)
    plot.add_sample_annotation(
        tracks,
        cmap_dict=palettes,
        alpha=0.7,
        size=0.7,
        linewidths=annotation_linewidth,
    )
    plot.render(
        main_width=main_width,
        main_height=26,
        legend_width=legend_width,
        legend_pad=0.2,
        show_sample_labels=show_sample_labels,
    )
    if show_sample_labels:
        align_sample_label_gutter(plot)
        for text in plot._sample_gutter_axes[0].texts:
            text.set_fontsize(12)
    for axis in plot.fig.axes:
        if [text.get_text() for text in axis.texts] == list(data.classes.index):
            for text in axis.texts:
                text.set_fontsize(12)
    for name in tracks:
        plot.legend_manager.legend_dict[name] = {
            key: to_rgba(color, 0.7) for key, color in palettes[name].items()
        }
    plot.plot_all_legends(
        fontsize=legend_fontsize, title_fontsize=legend_title_fontsize
    )
    return plot.fig


def render_cgc_oncoplot(data: CGCData) -> Figure:
    """Return Figure 4A with mutation, TMB, frequency and annotation tracks.

    Caller owns saving/closing the figure. No model fitting or gene selection
    occurs here. The function preserves the confirmed 0.7 annotation opacity
    and the existing LUAD/ASC/LUSC column blocks.
    """
    from .cgc_oncoplot import GROUPS, ROLE_COLORS
    from .figure_style import MUTATION_COLORS, TRACK_COLORS

    table = build_cgc_plot_table(data)
    mutation_colors = {
        False: "white",
        **{key: color for key, color in MUTATION_COLORS.items() if key != "WT"},
    }
    track_colors = {
        **TRACK_COLORS,
        "smoke": {str(key): color for key, color in TRACK_COLORS["smoke"].items()},
    }
    frequencies = FreqTrack(
        table.feature_metadata[list(GROUPS)],
        annot_fontsize=19,
        xtick_fontsize=16,
        linewidths=0.5,
    )
    frequencies.size = 4.5
    plot = (
        OncoPlot(table)
        .set_config(figsize=(30, 16), ytick_fontsize=15)
        .main(cmap_dict=mutation_colors, linewidths=0.5)
        .add_bar("TMB", size=1.5, ylabel_size=12)
        .add_track(frequencies)
        .add_feature_annotation(
            ["CGC role"],
            cmap_dict={"CGC role": ROLE_COLORS},
            alpha=0.7,
            size=1.1,
            linewidths=0.5,
        )
        .add_sample_annotation(
            list(TRACK_COLORS),
            cmap_dict=track_colors,
            alpha=0.7,
            size=0.7,
            linewidths=0.5,
        )
        .render(
            main_width=28,
            main_height=20,
            legend_width=5.5,
            legend_pad=0.1,
            show_sample_labels=False,
        )
    )
    # Match the reviewed legend order and opacity, not the API insertion order.
    legends = plot.legend_manager.legend_dict
    role_legend = legends.pop("CGC role")
    for track in TRACK_COLORS:
        legends[track] = {
            key: to_rgba(color, 0.7) for key, color in track_colors[track].items()
        }
    legends["CGC Role"] = {
        key: to_rgba(color, 0.7)
        for key, color in role_legend.items()
        if key in set(data.gene_summary.CGC_role)
    }
    plot.plot_all_legends(
        fontsize=19,
        title_fontsize=21,
        title_offset_y=0.05,
        item_offset_y=0.029,
        legend_gap=0.015,
    )
    # The new API's feature-label gutter otherwise uses a fixed 8-point font.
    for axis in plot.fig.axes:
        tick_labels = [label.get_text() for label in axis.get_xticklabels()]
        if tick_labels == list(GROUPS):
            axis.set_title("Frequency", fontsize=19)
        elif tick_labels == ["CGC role"]:
            axis.set_title("Role", fontsize=19)
            axis.set_xticks([])
        if axis.get_title() == "TMB":
            axis.set_title("")
            axis.set_ylabel("TMB")
        for label in axis.texts:
            if label.get_text() in data.classes.index:
                label.set_fontsize(14)
    return plot.fig
