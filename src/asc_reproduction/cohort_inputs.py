"""Build reusable pymaftools 0.5.0 cohorts from processed study inputs.

This follows ``paper/utils/setup_env.py`` without its working-directory or
``sys.path`` side effects. Input values and analysis-specific filters are not
changed here. Patient-level outputs remain under the Git-ignored ``data/``.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .provenance import file_hash as file_sha256

import pandas as pd
import pymaftools
from pymaftools import Cohort, PivotTable
from pymaftools.core.CopyNumberVariationTable import CopyNumberVariationTable
from pymaftools.core.SignatureTable import SignatureTable
from pymaftools.core.SmallVariationTable import SmallVariationTable

TABLE_PATHS = {
    "SNV": (SmallVariationTable, "WES/all_case_table.db"),
    "CNV_thresholded": (CopyNumberVariationTable, "CNV_0407/thresholded_cnv.db"),
    "CNV_gene": (CopyNumberVariationTable, "CNV_0407/gene_cnv.db"),
    "signature": (SignatureTable, "signature/signature_table.db"),
    "CNV_arm": (CopyNumberVariationTable, "CNV_0407/arm_cnv.db"),
}
METADATA_PATHS = (
    "Census All Months Jan 27 2025.csv",
    "all_case_metadata_0527.csv",
    "ASC_all_case_metadata_0312.csv",
    "clustering/best_k_cluster_labels.csv",
    "clustering/metric_df.csv",
    "WES/all_case_maf_AST.csv",
)
EXPECTED_SUBTYPES = {"LUAD": 89, "ASC": 29, "LUSC": 82}
SCOPES = ("tumor", "paired")
TABLE_NAMES = (*TABLE_PATHS, "CNV_cluster")


def as_sample_mask(values: pd.Series) -> pd.Series:
    """Decode AS_sample flags without treating the string 'False' as true."""
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.fillna(False)
    if pd.api.types.is_numeric_dtype(values.dtype):
        return values.eq(1).fillna(False)
    normalized = values.astype("string").str.strip().str.lower()
    return normalized.isin({"true", "1", "yes"}).fillna(False)


def select_paired_metadata(
    metadata: pd.DataFrame,
    cases: pd.DataFrame,
    *,
    expected_cases: int = 11,
) -> pd.DataFrame:
    """Select complete ATS triplets in case-file order without gene filtering.

    This is the shared equivalent of ASCDataLoader.get_ATS_subset. Original
    sample identifiers are retained; unrelated TMB or feature data is unused.
    """
    if metadata.index.has_duplicates or cases.case_ID.duplicated().any():
        raise ValueError("Duplicate sample or case keys")
    selected_cases = cases.loc[as_sample_mask(cases.AS_sample), "case_ID"]
    if len(selected_cases) != expected_cases or selected_cases.isna().any():
        raise ValueError("Unexpected complete-case cohort")
    parts = ("A", "T", "S")
    selected = metadata.loc[
        metadata.case_ID.isin(selected_cases) & metadata.sample_type.isin(parts)
    ]
    if selected.duplicated(["case_ID", "sample_type"]).any():
        raise ValueError("Multiple samples per case/component")
    lookup = selected.reset_index(names="sample").set_index(["case_ID", "sample_type"])[
        "sample"
    ]
    order = lookup.reindex(pd.MultiIndex.from_product([selected_cases, parts]))
    if order.isna().any():
        raise ValueError("Incomplete A/T/S triplet")
    return selected.loc[order.tolist()].copy()


def validate_prepared_inputs(data_root: Path, products: tuple[str, ...]) -> dict:
    """Verify source and HDF5 hashes for both direct and batch entry points."""
    prepared = data_root / "data/prepared"
    manifest_path = prepared / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"{manifest_path}; prepare the Cohort first")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("pymaftools") != "0.5.0" or pymaftools.__version__ != "0.5.0":
        raise ValueError("Prepared inputs require pymaftools 0.5.0")
    for relative, expected_hash in manifest["source_inputs_sha256"].items():
        if file_sha256(data_root / "data" / relative) != expected_hash:
            raise ValueError(f"Prepared Cohort source input is stale: {relative}")
    recorded = [manifest["cohort"]]
    for scope in manifest["scopes"].values():
        recorded.extend(scope["tables"].values())
    expected = {item["path"]: item["sha256"] for item in recorded}
    identity = {
        "data/prepared/manifest.json": {
            "bytes": manifest_path.stat().st_size,
            "sha256": file_sha256(manifest_path),
        }
    }
    for name in products:
        path = prepared / name
        digest = file_sha256(path)
        if digest != expected.get(name):
            raise ValueError(f"Prepared Cohort product differs from manifest: {name}")
        identity[f"data/prepared/{name}"] = {
            "bytes": path.stat().st_size,
            "sha256": digest,
        }
    return identity


def stage_processed_inputs(source_data: Path, local_data: Path) -> dict[str, str]:
    """Copy required processed inputs, refusing conflicting targets."""
    paths = [relative for _, relative in TABLE_PATHS.values()]
    paths.extend(METADATA_PATHS)
    checksums = {}
    for relative in paths:
        source = source_data / relative
        target = local_data / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        source_hash = file_sha256(source)
        if target.exists():
            if file_sha256(target) != source_hash:
                raise ValueError(f"Local input differs from source: {target}")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            if file_sha256(target) != source_hash:
                raise IOError(f"Copied input failed checksum validation: {target}")
        checksums[relative] = source_hash
    return checksums


def paired_sample_ids(case_metadata: pd.DataFrame) -> list[str]:
    """Reproduce the original case CSV order and A/T/S triplets."""
    cases = case_metadata.loc[as_sample_mask(case_metadata["AS_sample"]), "case_ID"]
    if len(cases) != 11 or cases.isna().any() or cases.duplicated().any():
        raise ValueError("Expected eleven unique complete ASC cases")
    return [f"{case}_{part}" for case in cases for part in ("A", "T", "S")]


def build_cohort(data_root: Path) -> tuple[Cohort, list[str]]:
    """Read the five original assay tables and construct the original cohort."""
    if pymaftools.__version__ != "0.5.0":
        raise RuntimeError("Cohort preparation requires pymaftools 0.5.0")
    data_dir = data_root / "data"
    labels = pd.read_csv(
        data_dir / "clustering/best_k_cluster_labels.csv", index_col=0
    )["cluster"]
    case_metadata = pd.read_csv(data_dir / "ASC_all_case_metadata_0312.csv")
    cohort = Cohort("ASC", "ASC study using LUAD/LUSC/ASC samples")
    for name, (table_type, relative) in TABLE_PATHS.items():
        table = table_type.read_sqlite(str(data_dir / relative))
        if name == "CNV_gene":
            if not table.feature_metadata.index.equals(labels.index):
                if set(table.feature_metadata.index) != set(labels.index):
                    raise ValueError("Gene CNV and cluster label features differ")
            table.feature_metadata["cluster"] = labels.reindex(
                table.feature_metadata.index
            )
            if table.feature_metadata["cluster"].isna().any():
                raise ValueError("Gene CNV contains unmapped cluster features")
        cohort.add_table(table, name)
    order = cohort.SNV.sort_samples_by_group(
        "subtype", ["LUAD", "ASC", "LUSC"]
    ).columns.tolist()
    cohort = cohort.subset(samples=order)
    if len(order) != 222 or len(set(order)) != len(order):
        raise ValueError("Expected 222 unique cohort samples")
    paired_ids = paired_sample_ids(case_metadata)
    if not set(paired_ids).issubset(order):
        raise ValueError("Complete-case A/T/S samples are missing from cohort")
    return cohort, paired_ids


def export_prepared_inputs(
    cohort: Cohort, paired_ids: list[str], prepared_dir: Path
) -> dict:
    """Save cohort and analysis-scope pivots, then verify every round trip."""
    prepared_dir.mkdir(parents=True, exist_ok=True)
    cohort_path = prepared_dir / "cohort.h5"
    cohort.to_hdf5(str(cohort_path))
    restored = Cohort.read_hdf5(str(cohort_path))
    if list(restored.sample_IDs) != list(cohort.sample_IDs):
        raise ValueError("Cohort HDF5 changed sample order")
    tumor_ids = cohort.sample_metadata.index[
        cohort.sample_metadata["sample_type"].eq("T")
    ].tolist()
    tumor = cohort.subset(samples=tumor_ids)
    paired = cohort.subset(samples=paired_ids)
    if tumor.sample_metadata["subtype"].value_counts().to_dict() != EXPECTED_SUBTYPES:
        raise ValueError("Unexpected T-only subtype counts")
    manifest = {"pymaftools": pymaftools.__version__, "scopes": {}}
    for scope_name, scope in (("tumor", tumor), ("paired", paired)):
        manifest["scopes"][scope_name] = {
            "samples": len(scope.sample_IDs),
            "tables": {},
        }
        tables = dict(scope.tables)
        tables["CNV_cluster"] = scope.CNV_gene.to_cluster_table()
        if len(tables["CNV_cluster"]) != 114:
            raise ValueError("Expected 114 CNV clusters")
        for name, table in tables.items():
            path = prepared_dir / f"{scope_name}_{name}.h5"
            table.to_h5(path)
            restored_table = type(table).read_h5(path)
            if not pd.DataFrame(table).equals(pd.DataFrame(restored_table)):
                raise ValueError(f"Pivot HDF5 changed matrix values: {path}")
            if list(restored_table.columns) != list(scope.sample_IDs):
                raise ValueError(f"Pivot HDF5 changed sample order: {path}")
            manifest["scopes"][scope_name]["tables"][name] = {
                "path": path.name,
                "features": len(table),
                "samples": len(table.columns),
                "sha256": file_sha256(path),
            }
    manifest["cohort"] = {
        "path": cohort_path.name,
        "samples": len(cohort.sample_IDs),
        "sha256": file_sha256(cohort_path),
    }
    return manifest


def read_prepared_table(data_root: Path, scope: str, name: str) -> PivotTable:
    """Load a checked analysis-scope pivot for any figure or model script.

    Callers must apply their own frequency cutoffs and model-fold feature
    selection; the shared pivots deliberately contain unfiltered features.
    """
    if scope not in SCOPES or name not in TABLE_NAMES:
        raise ValueError(f"Unknown prepared table: {scope}/{name}")
    path = data_root / "data/prepared" / f"{scope}_{name}.h5"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path}; run python -m asc_reproduction.cohort_inputs first"
        )
    validate_prepared_inputs(data_root, (path.name,))
    table = PivotTable.read_h5(path)
    if len(table.columns) != (200 if scope == "tumor" else 33):
        raise ValueError(f"Unexpected sample count in {path}")
    if not table.columns.equals(table.sample_metadata.index):
        raise ValueError(f"Sample metadata misaligned in {path}")
    return table


def read_prepared_cohort(data_root: Path) -> Cohort:
    """Load the complete ordered Cohort for analyses requiring all samples."""
    path = data_root / "data/prepared/cohort.h5"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path}; run python -m asc_reproduction.cohort_inputs first"
        )
    validate_prepared_inputs(data_root, (path.name,))
    cohort = Cohort.read_hdf5(str(path))
    if len(cohort.sample_IDs) != 222 or len(set(cohort.sample_IDs)) != 222:
        raise ValueError("Prepared cohort must contain 222 unique samples")
    for table in cohort.tables.values():
        if not table.columns.equals(cohort.sample_metadata.index):
            raise ValueError("Prepared cohort assays and metadata misaligned")
    return cohort


def main() -> None:
    """Stage local inputs and export verified Cohort/PivotTable HDF5 files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args()
    source_data = args.source_root.resolve() / "data"
    local_data = args.repo_root.resolve() / "data"
    checksums = stage_processed_inputs(source_data, local_data)
    cohort, paired_ids = build_cohort(args.repo_root.resolve())
    manifest = export_prepared_inputs(cohort, paired_ids, local_data / "prepared")
    manifest["source_inputs_sha256"] = checksums
    manifest_path = local_data / "prepared/manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared cohort and pivots: {manifest_path}")


if __name__ == "__main__":
    main()
