"""Read the project's processed SQLite pivot tables without changing cwd."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sqlite3

import pandas as pd


@dataclass(frozen=True)
class PivotData:
    """A feature-by-sample matrix and its row/column metadata."""

    values: pd.DataFrame
    feature_metadata: pd.DataFrame
    sample_metadata: pd.DataFrame


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _read_table(connection: sqlite3.Connection, table: str, index: str) -> pd.DataFrame:
    quoted_table = _quote_identifier(table)
    quoted_index = _quote_identifier(index)
    columns = [row[1] for row in connection.execute(f"PRAGMA table_info({quoted_table})")]
    if index not in columns:
        raise ValueError(f"SQLite table {table!r} lacks required key {index!r}")
    frame = pd.read_sql_query(f"SELECT * FROM {quoted_table}", connection, index_col=index)
    if frame.index.has_duplicates:
        raise ValueError(f"SQLite table {table!r} contains duplicate {index!r} keys")
    if frame.index.isna().any() or (frame.index.astype(str).str.strip() == "").any():
        raise ValueError(f"SQLite table {table!r} contains blank {index!r} keys")
    return frame


def read_pivot_sqlite(
    path: str | Path,
    *,
    matrix_table: str = "data",
    feature_metadata_table: str = "feature_metadata",
    sample_metadata_table: str = "sample_metadata",
) -> PivotData:
    """Load one read-only SQLite matrix and verify metadata key alignment.

    The input uses the pivot schema recorded in ``input_spec.json``. Data rows
    are features and columns are samples. The returned sample metadata is
    reordered to match the matrix columns; features are likewise aligned to
    the matrix row order.
    """

    database = Path(path).expanduser().resolve()
    if not database.is_file():
        raise FileNotFoundError(database)

    uri = database.as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        available = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        required = {matrix_table, feature_metadata_table, sample_metadata_table}
        missing = sorted(required - available)
        if missing:
            raise ValueError(f"SQLite database is missing required table(s): {', '.join(missing)}")

        values = _read_table(connection, matrix_table, "feature")
        feature_metadata = _read_table(connection, feature_metadata_table, "feature")
        sample_metadata = _read_table(connection, sample_metadata_table, "sample")

    if values.isna().to_numpy().any():
        raise ValueError("Matrix contains missing values")
    if set(values.index) != set(feature_metadata.index):
        raise ValueError("Matrix and feature metadata feature keys differ")
    matrix_samples = list(values.columns)
    if len(matrix_samples) != len(set(matrix_samples)):
        raise ValueError("Matrix contains duplicate sample columns")
    if set(matrix_samples) != set(sample_metadata.index):
        raise ValueError("Matrix columns and sample metadata keys differ")

    feature_metadata = feature_metadata.loc[values.index].copy()
    sample_metadata = sample_metadata.loc[matrix_samples].copy()
    return PivotData(values=values, feature_metadata=feature_metadata,
                     sample_metadata=sample_metadata)
