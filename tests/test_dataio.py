import sqlite3

import pandas as pd
import pytest

from asc_reproduction import read_pivot_sqlite


def write_database(path, *, duplicate_sample=False):
    with sqlite3.connect(path) as connection:
        pd.DataFrame(
            {"feature": ["g2", "g1"], "S2": [2.0, 1.0], "S1": [4.0, 3.0]}
        ).to_sql("data", connection, index=False)
        pd.DataFrame(
            {"feature": ["g1", "g2"], "cluster": [1, 2]}
        ).to_sql("feature_metadata", connection, index=False)
        samples = ["S1", "S1"] if duplicate_sample else ["S1", "S2"]
        pd.DataFrame(
            {"sample": samples, "subtype": ["LUAD", "LUSC"]}
        ).to_sql("sample_metadata", connection, index=False)


def test_reads_and_aligns_metadata_to_matrix(tmp_path):
    database = tmp_path / "matrix.db"
    write_database(database)

    result = read_pivot_sqlite(database)

    assert result.values.index.tolist() == ["g2", "g1"]
    assert result.values.columns.tolist() == ["S2", "S1"]
    assert result.feature_metadata.index.tolist() == ["g2", "g1"]
    assert result.sample_metadata.index.tolist() == ["S2", "S1"]
    assert result.sample_metadata["subtype"].tolist() == ["LUSC", "LUAD"]


def test_rejects_duplicate_metadata_keys(tmp_path):
    database = tmp_path / "duplicate.db"
    write_database(database, duplicate_sample=True)

    with pytest.raises(ValueError, match="duplicate 'sample' keys"):
        read_pivot_sqlite(database)
