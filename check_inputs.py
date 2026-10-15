"""Validate required processed inputs without printing or saving patient rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parent
SPEC_PATH = REPO / "input_spec.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_header(path: Path, delimiter: str = ",") -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return next(csv.reader(stream, delimiter=delimiter), [])


def inspect_csv(path: Path, item: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    if path.suffix == ".csv" and "all_case_maf_AST" in path.name:
        header = read_header(path, delimiter="\t")
        header = [column.strip() for column in header if column.strip()]
    else:
        header = read_header(path)
    missing_columns = sorted(set(item.get("required_columns", [])) - set(header))
    if missing_columns:
        errors.append("required column(s) missing: " + ", ".join(missing_columns))

    empty_required_counts: dict[str, int] = {}
    for column in item.get("nonempty_columns", []):
        if column not in header:
            continue
        index = header.index(column)
        missing_count = 0
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            next(reader, None)
            for row in reader:
                if index >= len(row) or not row[index].strip():
                    missing_count += 1
        empty_required_counts[column] = missing_count
        if missing_count:
            errors.append(f"{missing_count} empty value(s) in required field {column}")

    row_count = 0
    duplicate_keys = 0
    categories: Counter[str] = Counter()
    key = item.get("unique_key")
    category = item.get("category_column")
    if key:
        key_index = header.index(key) if key in header else None
        category_index = header.index(category) if category in header else None
        seen: set[str] = set()
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            next(reader, None)
            for row in reader:
                row_count += 1
                if category_index is not None and category_index < len(row):
                    categories[row[category_index].strip()] += 1
                if key_index is None or key_index >= len(row):
                    continue
                value = row[key_index].strip()
                if not value or value in seen:
                    duplicate_keys += 1
                seen.add(value)
        if duplicate_keys:
            errors.append(f"{duplicate_keys} blank or duplicate {key} value(s)")
        if category:
            expected_count = item.get("expected_category_count")
            if "" in categories:
                errors.append(f"{categories['']} blank {category} value(s)")
            if expected_count is not None and len(categories) != expected_count:
                errors.append(f"expected {expected_count} distinct {category} labels, found {len(categories)}")
    else:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            row_count = max(sum(1 for _ in csv.reader(stream)) - 1, 0)

    return {"kind": "csv", "bytes": path.stat().st_size, "sha256": sha256(path),
            "header_column_count": len(header), "row_count": row_count,
            "duplicate_key_count": duplicate_keys,
            "empty_required_counts": empty_required_counts,
            "category_count": len(categories) if category else None}, errors


def inspect_database(path: Path, item: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    uri = f"file:{path.resolve()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        table_names = {row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        requested = [item["table"], item["feature_metadata_table"], item["sample_metadata_table"]]
        absent = [name for name in requested if name not in table_names]
        if absent:
            errors.append("required SQLite table(s) missing: " + ", ".join(absent))
            return {"kind": "sqlite", "bytes": path.stat().st_size, "sha256": sha256(path),
                    "tables_present": False}, errors

        table_summary: dict[str, Any] = {}
        matrix_sample_columns: list[str] = []
        sample_metadata_values: list[str] = []
        for table_name in requested:
            quoted = '"' + table_name.replace('"', '""') + '"'
            columns = [row[1] for row in connection.execute(f"PRAGMA table_info({quoted})")]
            row_count = connection.execute(f"SELECT COUNT(*) FROM {quoted}").fetchone()[0]
            key = "feature" if table_name == item["table"] or table_name == item["feature_metadata_table"] else "sample"
            if key not in columns:
                errors.append(f"table {table_name} is missing key column {key}")
            if table_name == item["table"]:
                matrix_sample_columns = [column for column in columns if column != "feature"]
                table_summary[table_name] = {"rows": row_count, "feature_count": row_count,
                                             "sample_count": len(matrix_sample_columns),
                                             "has_feature_key": "feature" in columns}
            else:
                table_summary[table_name] = {"rows": row_count,
                                             "column_count": len(columns),
                                             "has_key": key in columns}
                if table_name == item["sample_metadata_table"] and key in columns:
                    sample_metadata_values = [row[0] for row in connection.execute(f"SELECT sample FROM {quoted}")]
                if table_name == item["feature_metadata_table"] and key in columns:
                    feature_metadata_count = connection.execute(
                        f"SELECT COUNT(*) FROM {quoted}"
                    ).fetchone()[0]
                    if feature_metadata_count != table_summary[item["table"]]["feature_count"]:
                        errors.append(f"table {table_name} feature count differs from matrix")

        feature_table = '"' + item["table"].replace('"', '""') + '"'
        duplicate_feature_count = connection.execute(
            f"SELECT COUNT(*) FROM (SELECT feature FROM {feature_table} GROUP BY feature HAVING COUNT(*) > 1)"
        ).fetchone()[0]
        if duplicate_feature_count:
            errors.append(f"matrix has {duplicate_feature_count} duplicated feature key(s)")
        missing_feature_count = connection.execute(
            f"SELECT COUNT(*) FROM {feature_table} WHERE feature IS NULL OR TRIM(feature) = ''"
        ).fetchone()[0]
        if missing_feature_count:
            errors.append(f"matrix has {missing_feature_count} blank feature key(s)")
        feature_metadata = '"' + item["feature_metadata_table"].replace('"', '""') + '"'
        feature_set_difference = connection.execute(
            f"SELECT COUNT(*) FROM (SELECT feature FROM {feature_table} EXCEPT SELECT feature FROM {feature_metadata})"
        ).fetchone()[0]
        metadata_set_difference = connection.execute(
            f"SELECT COUNT(*) FROM (SELECT feature FROM {feature_metadata} EXCEPT SELECT feature FROM {feature_table})"
        ).fetchone()[0]
        if feature_set_difference or metadata_set_difference:
            errors.append("matrix and feature metadata feature keys differ")
        quoted_columns = ["\"" + column.replace('"', '""') + "\"" for column in matrix_sample_columns]
        null_cells = connection.execute(
            f"SELECT {' + '.join(f'SUM({column} IS NULL)' for column in quoted_columns)} FROM {feature_table}"
        ).fetchone()
        missing_matrix_values = sum(value or 0 for value in null_cells)
        if missing_matrix_values:
            errors.append(f"matrix contains {missing_matrix_values} NULL value(s)")
        if len(sample_metadata_values) != len(set(sample_metadata_values)):
            errors.append("sample metadata contains duplicated sample key(s)")
        if set(sample_metadata_values) != set(matrix_sample_columns):
            errors.append("sample metadata keys differ from matrix sample columns")
        return {"kind": "sqlite", "bytes": path.stat().st_size, "sha256": sha256(path),
                "tables": table_summary, "matrix_sample_columns": matrix_sample_columns,
                "sample_metadata_values": sample_metadata_values,
                "duplicate_matrix_features": duplicate_feature_count,
                "missing_matrix_values": missing_matrix_values,
                "matrix_feature_metadata_key_differences": feature_set_difference + metadata_set_difference}, errors
    finally:
        connection.close()


def inspect_input(data_root: Path, item: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    path = data_root / item["path"]
    if not path.is_file():
        return {"path": item["path"], "exists": False}, ["required file missing"]
    try:
        if item["format"].startswith("SQLite"):
            details, errors = inspect_database(path, item)
        else:
            details, errors = inspect_csv(path, item)
    except (OSError, sqlite3.Error, UnicodeError, csv.Error, StopIteration) as error:
        details, errors = {"kind": item["format"], "bytes": path.stat().st_size}, [f"cannot inspect file: {error}"]
    details.update({"path": item["path"], "exists": True})
    return details, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="directory that contains the external data/ folder")
    parser.add_argument("--output", type=Path, default=REPO / "local" / "input_inventory.json",
                        help="local inventory destination (default is Git-ignored)")
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    entries: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for item in spec["required"]:
        entry, item_errors = inspect_input(data_root, item)
        entries.append(entry)
        errors.extend({"path": item["path"], "error": error} for error in item_errors)

    # Ensure all matrices encode the same cohort without persisting its identifiers.
    matrix_sets = [entry.get("matrix_sample_columns") for entry in entries if "matrix_sample_columns" in entry]
    sample_set_matches = bool(matrix_sets)
    if matrix_sets:
        reference = set(matrix_sets[0])
        for entry in entries:
            samples = entry.get("matrix_sample_columns")
            if samples is not None and set(samples) != reference:
                sample_set_matches = False
                errors.append({"path": entry["path"], "error": "matrix sample set differs from the first cohort matrix"})
        for entry in entries:
            entry.pop("matrix_sample_columns", None)
            entry.pop("sample_metadata_values", None)

    result = {
        "schema_version": 1,
        "data_root_label": "external data root (absolute path intentionally omitted)",
        "inputs": entries,
        "matrix_sample_sets_match": sample_set_matches,
        "validation_errors": errors,
        "patient_level_values_or_identifiers_written": False,
        "note": "SHA-256 values are local data fingerprints. Do not publish this generated manifest without data-sharing review."
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"required_files": len(entries), "present": sum(x.get("exists", False) for x in entries),
                      "validation_errors": errors, "inventory": str(args.output)}, indent=2, ensure_ascii=False))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
