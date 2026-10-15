"""S12A/B EGFR and TP53 with externally supplied domain annotation."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse

from .dataio import read_pivot_sqlite
from .figure_style import MUTATION_COLORS, plt, save_tiff_and_preview
from .protein_positions import protein_counts

GROUPS = ("LUAD", "ASC", "LUSC")


def select_domains(frame, gene, length):
    """Select the original first matching RefSeq and validate domain bounds."""
    if not {"HGNC", "aa.length", "refseq.ID", "Start", "End", "Label"}.issubset(frame):
        raise ValueError("Missing domain annotation fields")
    subset = frame.loc[frame.HGNC.eq(gene) & frame["aa.length"].eq(length)]
    if (
        subset.empty
        or subset[["refseq.ID", "Start", "End", "Label"]].isna().any().any()
    ):
        raise ValueError("Missing domain annotation")
    refs = subset["refseq.ID"].unique().tolist()
    selected = subset.loc[subset["refseq.ID"].eq(refs[0])]
    bounds = selected[["Start", "End"]].to_numpy(dtype=float)
    if (
        not np.isfinite(bounds).all()
        or (bounds[:, 0] < 1).any()
        or (bounds[:, 1] > length).any()
        or (bounds[:, 0] > bounds[:, 1]).any()
    ):
        raise ValueError("Domain outside protein coordinate range")
    return selected[["Start", "End", "Label"]].copy(), refs


def plot_protein(gene, counts, domains, length, refseq):
    """Use the original package multi-cohort API with fixed legend geometry."""
    from pymaftools import LollipopPlot

    from .oncoplot_rendering import require_pymaftools_version

    require_pymaftools_version()
    names = {
        "Recep_L_domain": "Receptor L domain",
        "FU": "Furin-like domain",
        "Furin-like": "Furin-like domain",
        "TM_ErbB1": "Transmembrane domain of ErbB1 (EGFR)",
        "PTKc_EGFR": "Protein tyrosine kinase (EGFR catalytic domain)",
        "TyrKc": "Tyrosine kinase domain",
        "P53_TAD": "p53 Transactivation domain",
        "P53": "p53 DNA-binding domain (core DBD)",
        "P53_tetramer": "p53 Tetramerization domain",
    }
    cohorts = {
        group: (length, counts[group].to_dict("records"),
                domains.to_dict("records"), refseq)
        for group in GROUPS
    }
    plot = LollipopPlot.plot_multi_cohort(
        gene, cohorts, figsize=(16, 6), width_ratios=[8, 2],
        domain_label_map=names,
    )
    fig = plot.fig
    axes = [axis for axis in fig.axes if axis is not plot.ax_legend]
    for i, axis in enumerate(axes):
        axis.set_position([0.075, 0.72 - i * 0.28, 0.71, 0.20])
        axis.grid(False)
        axis.tick_params(labelsize=10)
        assert axis.get_title() == GROUPS[i]
        points = [collection.get_offsets()[0] for collection in axis.collections]
        np.testing.assert_allclose(
            [point[0] for point in points], counts[GROUPS[i]].position
        )
    plot.ax_legend.set_position([0.787, 0.10, 0.208, 0.82])
    plot.ax_legend.clear()
    legends = plot.legend_manager.legend_dict
    observed = legends["Mutation Types"]
    legends["Mutation Types"] = {
        key: observed[key] for key in MUTATION_COLORS if key in observed
    }
    plot.plot_all_legends(
        ax=plot.ax_legend, fontsize=9, title_fontsize=11,
        start_y=0.98, title_offset_y=0.09, item_offset_y=0.050,
        legend_gap=0.055, rect_height=0.022, text_offset_x=0.12,
    )
    mutation_title = next(
        text for text in plot.ax_legend.texts
        if text.get_text() == "Mutation Types"
    )
    legend_box = plot.ax_legend.get_window_extent()
    for patch in list(plot.ax_legend.patches):
        if patch.get_y() < mutation_title.get_position()[1]:
            center = (patch.get_x() + patch.get_width() / 2,
                      patch.get_y() + patch.get_height() / 2)
            diameter = patch.get_height()
            marker = Ellipse(
                center, diameter * legend_box.height / legend_box.width,
                diameter, facecolor=patch.get_facecolor(), edgecolor="none",
                linewidth=0, transform=plot.ax_legend.transAxes,
            )
            plot.ax_legend.add_patch(marker)
            patch.remove()
    return fig



def main():
    """Validate source cohorts and export two package-rendered S12 panels."""
    from .figure_style import setup_plotting

    setup_plotting()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--domain-csv", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Output must be new or empty")
    paths = {
        "domains": args.domain_csv,
        "metadata": args.data_root / "data/WES/all_case_table.db",
    }
    paths.update({g: args.data_root / f"data/WES/{g}.maf" for g in GROUPS})
    fingerprints = {
        k: hashlib.sha256(v.read_bytes()).hexdigest() for k, v in paths.items()
    }
    metadata = read_pivot_sqlite(paths["metadata"]).sample_metadata
    mafs = {g: pd.read_csv(paths[g], sep="\t") for g in GROUPS}
    for g, m in mafs.items():
        if set(m.sample_ID) != set(
            metadata.index[metadata.subtype.eq(g) & metadata.sample_type.eq("T")]
        ):
            raise ValueError("MAF sample cohort mismatch")
    resource = pd.read_csv(args.domain_csv, index_col=0, low_memory=False)
    args.output.mkdir(parents=True, exist_ok=True)
    report = []
    for panel, gene in [("S12A", "EGFR"), ("S12B", "TP53")]:
        counts = {}
        stats = {}
        for g, m in mafs.items():
            counts[g], stats[g] = protein_counts(m, gene)
        lengths = {s["protein_length"] for s in stats.values()}
        if len(lengths) != 1:
            raise ValueError("Cohort protein lengths differ")
        length = lengths.pop()
        domains, refs = select_domains(resource, gene, length)
        fig = plot_protein(gene, counts, domains, length, refs[0])
        info = save_tiff_and_preview(fig, args.output / panel, plot_axis=fig.axes[1])
        plt.close(fig)
        pd.concat([d.assign(cohort=g) for g, d in counts.items()]).to_csv(
            args.output / f"{panel}_position_counts.csv", index=False
        )
        report.append(
            {
                "panel": panel,
                "gene": gene,
                "cohorts": stats,
                "chosen_refseq": refs[0],
                "matching_refseqs": refs,
                "domain_rows": len(domains),
                "figure": info,
            }
        )
    if fingerprints != {
        k: hashlib.sha256(v.read_bytes()).hexdigest() for k, v in paths.items()
    }:
        raise ValueError("Inputs changed during rendering")
    (args.output / "summary.json").write_text(
        json.dumps(
            {
                "panels": report,
                "input_sha256": fingerprints,
                "scope": "Legacy positions/counts/domains; pymaftools 0.5.0 LollipopPlot.plot_multi_cohort; package count-height mapping preserved",
                "limitations": "External domain resource provenance/license pending; resource not bundled. No sample IDs exported.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
