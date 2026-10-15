import numpy as np
import pandas as pd
import pytest

from asc_reproduction.paired_screen import (
    cluster_mean_matrix,
    paired_feature_statistics,
)


def test_cluster_matrix_is_gene_mean_in_numeric_cluster_order():
    matrix = pd.DataFrame(
        [[1.0, 3.0], [3.0, 5.0], [8.0, 6.0]],
        index=["g1", "g2", "g3"],
        columns=["s1", "s2"],
    )
    cluster_map = pd.DataFrame({"feature": ["g3", "g1", "g2"], "cluster": [8, 47, 47]})

    result = cluster_mean_matrix(matrix, cluster_map, expected_cluster_count=2)

    assert result.index.tolist() == ["8", "47"]
    assert result.loc["47"].tolist() == [2.0, 4.0]
    assert result.loc["8"].tolist() == [8.0, 6.0]


def test_paired_screen_aligns_cases_and_preserves_both_x_axis_candidates():
    columns = ["c2_S", "c1_T", "c3_A", "c2_A", "c1_S", "c3_T", "c2_T", "c1_A", "c3_S"]
    matrix = pd.DataFrame(
        [
            [2, 1, 0, 0, 2, 1, 1, 0, 2],
            [3, 2, 1, 1, 3, 2, 2, 1, 3],
        ],
        index=["arm1q", "arm3q"],
        columns=columns,
        dtype=float,
    )
    metadata = pd.DataFrame(
        {
            "case_ID": [sample.split("_")[0] for sample in columns],
            "sample_type": [sample.split("_")[1] for sample in columns],
        },
        index=columns,
    )

    result = paired_feature_statistics(matrix, metadata, expected_case_count=3)

    assert result["legacy_mean_pairwise_sum"].tolist() == [4.0, 4.0]
    assert np.allclose(
        result["legacy_mean_pairwise_sum"], 3 * result["mean_over_three_pairs"]
    )
    assert result["p"].notna().all()
    assert result["BH_adjusted_p"].between(0, 1).all()
    np.testing.assert_allclose(result["mean_over_three_pairs"], [4 / 3, 4 / 3])


def test_paired_screen_requires_complete_case_triplets():
    matrix = pd.DataFrame([[0.0, 0.2]], index=["3q"], columns=["c1_A", "c1_S"])
    metadata = pd.DataFrame(
        {"case_ID": ["c1", "c1"], "sample_type": ["A", "S"]}, index=matrix.columns
    )

    with pytest.raises(ValueError, match="complete matched cases"):
        paired_feature_statistics(matrix, metadata, expected_case_count=1)
