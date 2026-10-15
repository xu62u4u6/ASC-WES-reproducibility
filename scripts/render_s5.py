"""Produce S5A and three S5B panels from the original notebook definitions."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from asc_reproduction.cohort_inputs import read_prepared_table
from asc_reproduction.figure_style import plt, save_tiff_and_preview, setup_plotting
from asc_reproduction.paired_oncoplot import (
    plot_paired_oncoplot,
    prepare_paired_oncoplot,
)
from asc_reproduction.shared_private_figures import (
    build_shared_private_tables,
    draw_shared_private_table,
)


def matrix_axis(figure, cells):
    """Identify the actual matrix rectangle for aligned raster assembly."""
    axes = [
        axis for axis in figure.axes
        if any(mesh.get_array().size == cells for mesh in axis.collections)
    ]
    if len(axes) != 1:
        raise ValueError("Expected one main matrix axis")
    return axes[0]


def render(data_root: Path, output: Path) -> None:
    """Save verified panel inputs, images and provenance without replacing Word."""
    setup_plotting()
    output.mkdir(parents=True, exist_ok=False)
    pivot = read_prepared_table(data_root, "paired", "SNV")
    cases = pd.read_csv(data_root / "data/ASC_all_case_metadata_0312.csv")
    data, labels = prepare_paired_oncoplot(
        pd.DataFrame(pivot), pivot.sample_metadata, cases
    )
    figure = plot_paired_oncoplot(data, labels)
    records = {"A": save_tiff_and_preview(
        figure, output / "S5A", plot_axis=matrix_axis(figure, data.classes.size)
    )}
    plt.close(figure)
    data.gene_summary.to_csv(output / "S5A_gene_summary.csv")
    maf = data_root / "data/WES/all_case_maf_AST.csv"
    case_ids = data.metadata.case_ID.drop_duplicates().tolist()
    tables = build_shared_private_tables(maf, case_ids)
    for name, table in tables.items():
        table.to_h5(output / f"{name}.h5")
        table.feature_metadata.to_csv(output / f"{name}_features.csv")
        figure = draw_shared_private_table(table)
        records[name] = save_tiff_and_preview(
            figure, output / name, plot_axis=matrix_axis(figure, table.size)
        )
        plt.close(figure)
    records["provenance"] = {
        "pymaftools": "0.5.0",
        "MAF_sha256": hashlib.sha256(maf.read_bytes()).hexdigest(),
        "paired_SNV_sha256": hashlib.sha256(
            (data_root / "data/prepared/paired_SNV.h5").read_bytes()
        ).hexdigest(),
        "definitions": "AST_shared=A&T&S; A_private=A-S; S_private=S-A",
        "selection": "all mutation classes; at least 10% of 11 cases",
        "status": "candidate; original algorithm, not strict component-only",
    }
    (output / "panels.json").write_text(json.dumps(records, indent=2) + "\n")


def main() -> None:
    """Run S5 only; assembly is a separate script."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument(
        "--output", type=Path,
        default=Path("outputs/Supplementary_Figure_05/package_panels"),
    )
    args = parser.parse_args()
    render(args.data_root, args.output)


if __name__ == "__main__":
    main()
