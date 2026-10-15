"""Figure 2A legacy candidate: nonsynonymous EGFR/TP53/RBM10 variants."""

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from .cgc_oncoplot import CGCData
from .cohort_inputs import select_paired_metadata
from .dataio import read_pivot_sqlite
from .figure_style import MUTATION_COLORS, plt, save_tiff_and_preview
from .oncoplot_rendering import (
    FIGURE_2A_METADATA_ALPHA,
    render_driver_oncoplot,
)

TYPES = (
    "Frame_Shift_Del",
    "Frame_Shift_Ins",
    "In_Frame_Del",
    "In_Frame_Ins",
    "Missense_Mutation",
    "Nonsense_Mutation",
    "Splice_Site",
    "Translation_Start_Site",
    "Nonstop_Mutation",
)
GENES = ("EGFR", "TP53", "RBM10")


def prepare_driver_variants(maf, ordered_metadata):
    required = {"sample_ID", "Variant_Classification", "HGVSp_Short"}
    if not required.issubset(maf.columns):
        raise ValueError("Missing MAF fields")
    if (
        maf.index.isna().any()
        or maf[["sample_ID", "Variant_Classification"]].isna().any().any()
    ):
        raise ValueError("Missing variant keys or classification")
    if ordered_metadata.index.has_duplicates:
        raise ValueError("Duplicate sample keys")
    f = maf.loc[maf.Variant_Classification.isin(TYPES)]
    matrix = f.pivot_table(
        index=f.index,
        columns="sample_ID",
        values="Variant_Classification",
        aggfunc="first",
    ).fillna(False)
    if not set(ordered_metadata.index).issubset(matrix.columns):
        raise ValueError("Missing matched samples")
    matrix = matrix.loc[
        matrix.index.str.contains(r"^(?:EGFR|TP53|RBM10)\|"), ordered_metadata.index
    ]
    if matrix.empty:
        raise ValueError("No selected variants")
    meta = maf.loc[~maf.index.duplicated(keep="first")]
    labels = []
    for key in matrix.index:
        gene = str(key).split("|")[0]
        aa = meta.loc[key, "HGVSp_Short"]
        vc = meta.loc[key, "Variant_Classification"]
        suffix = (
            aa.removeprefix("p.") if isinstance(aa, str) and aa not in ("", ".") else vc
        )
        labels.append(f"{gene} {suffix}")
    duplicates = {label for label, n in Counter(labels).items() if n > 1}
    labels = [
        f"{label} ({str(key).split('|')[1]})" if label in duplicates else label
        for key, label in zip(matrix.index, labels)
    ]
    if len(set(labels)) != len(labels):
        raise ValueError("Ambiguous variant display labels")
    binary = matrix.ne(False)
    summary = binary.mean(axis=1).to_frame("freq")
    summary["count"] = binary.sum(axis=1)
    shared = pd.DataFrame(
        {
            case: binary.loc[:, group.index].all(axis=1)
            for case, group in ordered_metadata.groupby("case_ID")
        }
    )
    summary["shared_triplet_case_count"] = shared.sum(axis=1)
    summary.index = labels
    summary.index.name = "variant_label"
    classes = matrix.replace({False: "WT"})
    classes.index = labels
    if set(classes.to_numpy().ravel()) - set(MUTATION_COLORS):
        raise ValueError("Unsupported variant class")
    return CGCData(classes, ordered_metadata, summary)


def run(data_root: Path, output: Path) -> None:
    """Generate the Figure 2A candidate with the existing variant definition."""
    from .figure_style import setup_plotting

    setup_plotting()
    from .provenance import file_hash, runtime_identity

    if output.exists() and any(output.iterdir()):
        raise ValueError("Output must be new or empty")
    sources = [
        data_root / "data" / name
        for name in (
            "WES/all_case_table.db",
            "ASC_all_case_metadata_0312.csv",
            "WES/all_case_maf_AST.csv",
        )
    ]
    hashes = {str(path): file_hash(path) for path in sources}
    runtime = runtime_identity()
    snv = read_pivot_sqlite(data_root / "data/WES/all_case_table.db")
    cases = pd.read_csv(data_root / "data/ASC_all_case_metadata_0312.csv")
    paired_metadata = select_paired_metadata(snv.sample_metadata, cases)
    labels = paired_metadata.index.astype(str).tolist()
    maf = pd.read_csv(
        data_root / "data/WES/all_case_maf_AST.csv", sep="\t", index_col=0
    )
    data = prepare_driver_variants(maf, paired_metadata)
    if len(data.classes) != 15:
        raise ValueError("Expected 15 driver variants")
    output.mkdir(parents=True, exist_ok=True)
    fig = render_driver_oncoplot(data, labels)
    info = save_tiff_and_preview(fig, output / "Figure2A_driver_variants")
    plt.close(fig)
    data.gene_summary.to_csv(output / "Figure2A_variant_summary.csv")
    summary = {
        "panel": "Figure2A",
        "variants": 15,
        "samples": 33,
        "genes": list(GENES),
        "figure": info,
        "input_sha256": hashes,
        "runtime": runtime,
        "annotation_alpha": FIGURE_2A_METADATA_ALPHA,
        "renderer": "pymaftools 0.5.0 OncoPlot tracks",
        "selection": "Nonsynonymous variants in three prespecified genes; NOT an AST-shared filter",
        "never_shared_in_all_three_in_any_case": int(
            data.gene_summary.shared_triplet_case_count.eq(0).sum()
        ),
        "sample_labels": "Original manuscript sample IDs in the figure; no per-sample CSV",
        "limitations": "Caption selection wording requires reconciliation; not a shared-only filter.",
    }
    if hashes != {str(path): file_hash(path) for path in sources}:
        raise RuntimeError("Input changed during driver rendering")
    summary["artifacts"] = {path.name: file_hash(path) for path in output.iterdir()}
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
