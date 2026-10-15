"""S5B: reproduce the notebook's A-versus-S variant sets, not strict-only sets."""

from pathlib import Path

import pandas as pd
from pymaftools import MAF, OncoPlot, PivotTable
from pymaftools.plot.Track import FreqTrack

from .oncoplot_rendering import (
    align_sample_label_gutter,
    require_pymaftools_version,
    validate_frequency_alignment,
)

GROUP_TITLES = {
    "AST_shared": "Shared mutations across all three components (A, T, and S)",
    "A_private": "A-only mutations",
    "S_private": "S-only mutations",
}

# Original manuscript row order resolves equal-frequency ties only.
DISPLAY_ORDER = {
    "AST_shared": ["EGFR L858R"],
    "A_private": ["RPS3A Q157P", "TRBV6-8 3'Flank"],
    "S_private": [
        "HLA-DQB1 G234=", "UGT1A6 Intron", "KRTAP9-9 3'UTR",
        "AHNAK2 D3341=", "HLA-DQB1 S233G",
    ],
}


def mutation_label(key: str, annotations: pd.DataFrame) -> str:
    """Use the legacy gene/protein label, falling back to classification."""
    parts = key.split("|")
    original = "|".join(parts[:6])
    record = annotations.loc[original]
    protein = record["HGVSp_Short"]
    suffix = record["Variant_Classification"]
    if isinstance(protein, str) and protein not in ("", "."):
        suffix = protein.removeprefix("p.")
    return f"{parts[0]} {suffix}"


def component_variant_sets(a: set, t: set, s: set) -> dict:
    """Preserve the notebook's pairwise-private, not strictly-private, sets."""
    return {"AST_shared": a & t & s, "A_private": a - s, "S_private": s - a}


def build_shared_private_tables(maf_path: Path, case_ids: list[str]) -> dict:
    """Apply notebook cell24 sets, using all supplied mutation classes.

    A-private means A minus S; T presence is deliberately not excluded.
    Set keys include the mutation index and first classification per sample,
    exactly as the source notebook and earlier definition audit do.
    """
    require_pymaftools_version()
    if len(case_ids) != 11 or len(set(case_ids)) != len(case_ids):
        raise ValueError("S5B requires the eleven complete paired cases")
    maf = MAF.read_csv(maf_path)
    matrix = pd.DataFrame(maf.to_mutation_table())
    annotations = maf.loc[~maf.index.duplicated(keep="first")]
    groups = {name: {} for name in GROUP_TITLES}
    for case in case_ids:
        parts = {}
        for component in ("A", "T", "S"):
            column = matrix[f"{case}_{component}"]
            present = column.loc[column.ne(False)]
            parts[component] = set(
                present.index.astype(str) + "|" + present.astype(str)
            )
        a, t, s = (parts[component] for component in ("A", "T", "S"))
        for name, variants in component_variant_sets(a, t, s).items():
            groups[name][case] = variants
    tables = {}
    for name, per_case in groups.items():
        keys = sorted(set().union(*per_case.values()))
        values = pd.DataFrame(
            {case: [key in per_case[case] for key in keys] for case in case_ids},
            index=keys,
        )
        table = PivotTable(values).add_freq()
        keep = table.feature_metadata.index[table.feature_metadata.freq.ge(0.1)]
        table = table.subset(features=keep).sort_features("freq", ascending=False)
        labels = [mutation_label(key, annotations) for key in table.index]
        if len(set(labels)) != len(labels):
            raise ValueError("Ambiguous S5B variant display names")
        table.feature_metadata["display_label"] = labels
        if set(labels) != set(DISPLAY_ORDER[name]):
            raise ValueError(f"{name}: variants differ from the original figure")
        keys_by_label = dict(zip(labels, table.index))
        table = table.subset(
            features=[keys_by_label[label] for label in DISPLAY_ORDER[name]]
        )
        tables[name] = table
    if [len(table) for table in tables.values()] != [1, 2, 5]:
        raise ValueError("S5B no longer matches the source's 1/2/5 variant rows")
    return tables


def draw_shared_private_table(table: PivotTable) -> object:
    """Draw one S5B group through 0.5.0; assembly adds titles and letters."""
    require_pymaftools_version()
    plot = OncoPlot(table).set_config(figsize=(20.5, len(table) * 0.7 + 2))
    plot.main(
        cmap_dict={True: "gray", False: "white"},
        linewidths=0.4, linecolor="#dddddd",
    )
    plot.add_freq(annot_fontsize=13, xtick_fontsize=12, linewidths=0.4)
    for track in plot.tracks:
        if isinstance(track, FreqTrack):
            track.size = 2.3
    plot.render(
        main_width=32.6,
        main_height=len(table) * 1.4,
        feature_gutter_size=6.5,
        legend_width=0,
        legend_pad=0,
        sample_gutter_size=4,
        show_sample_labels=True,
        feature_labels=table.feature_metadata.display_label.tolist(),
    )
    align_sample_label_gutter(plot)
    plot.add_xticklabel(fontsize=12, rotation=90)
    # The complete panel has one shared legend, added by the assembler.
    plot.ax_legend.set_visible(False)
    labels = table.feature_metadata.display_label.tolist()
    for axis in plot.fig.axes:
        if [text.get_text() for text in axis.texts] == labels:
            for text in axis.texts:
                text.set_fontsize(14)
    validate_frequency_alignment(plot)
    return plot.fig
