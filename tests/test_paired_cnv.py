import json

import numpy as np
import pandas as pd
import pytest

from asc_reproduction.figure_style import save_tiff_and_preview
from asc_reproduction.paired_cnv import calculate_paired_3q, make_gain_figure


def test_3q_pairs_by_case_and_includes_the_threshold_boundary():
    matrix = pd.DataFrame(
        [[0.10, -0.20, 0.10, 0.10, 0.20, 0.10]],
        index=["3q"],
        columns=["c2_S", "c1_A", "c1_T", "c2_A", "c2_T", "c1_S"],
    )
    metadata = pd.DataFrame(
        {
            "case_ID": ["c2", "c1", "c1", "c2", "c2", "c1"],
            "sample_type": ["S", "A", "T", "A", "T", "S"],
        },
        index=matrix.columns,
    )

    result = calculate_paired_3q(matrix, metadata, expected_case_count=2)

    assert result.values[["A", "T", "S"]].to_numpy().tolist() == [
        [-0.2, 0.1, 0.1],
        [0.1, 0.2, 0.1],
    ]
    assert result.gain_summary["n_gain"].tolist() == [1, 2, 2]
    assert result.gain_summary["n_total"].tolist() == [2, 2, 2]
    assert np.isfinite(result.paired_p_value)
    assert all(isinstance(index, int) for index in result.values.index)


def test_3q_rejects_incomplete_or_duplicated_case_components():
    matrix = pd.DataFrame([[0.0, 0.2]], index=["3q"], columns=["c1_A", "c1_S"])
    metadata = pd.DataFrame(
        {"case_ID": ["c1", "c1"], "sample_type": ["A", "S"]},
        index=matrix.columns,
    )

    with pytest.raises(ValueError, match="complete matched cases"):
        calculate_paired_3q(matrix, metadata, expected_case_count=1)


def test_gain_panel_is_written_as_decodable_rgb_tiff(tmp_path):
    summary = pd.DataFrame(
        {
            "component": ["A", "T", "S"],
            "n_gain": [2, 4, 8],
            "n_total": [11, 11, 11],
            "percent": [200 / 11, 400 / 11, 800 / 11],
        }
    )
    figure = make_gain_figure(summary)
    manifest = save_tiff_and_preview(figure, tmp_path / "gain")

    assert manifest["mode"] == "RGB"
    assert manifest["dpi"] == [600, 600]
    assert manifest["decode"] == "passed"
    json.dumps(manifest)
    assert (tmp_path / "gain.png").is_file()
    assert (tmp_path / "gain.tiff").is_file()
