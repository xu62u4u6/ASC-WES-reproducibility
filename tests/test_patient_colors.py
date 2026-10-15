"""Cross-panel patient-color registration tests."""

import numpy as np
import pytest
from matplotlib import colormaps
from matplotlib.colors import to_hex

from asc_reproduction.figure_style import (
    CASE_COLORS,
    default_color_manager,
    patient_palette,
)


def test_registered_colors_preserve_approved_s6_palette():
    """The fixed map retains S6C/D colors rather than rotating by order."""
    expected = [
        to_hex(colormaps["tab20"](position)) for position in np.linspace(0, 1, 11)
    ]
    assert list(CASE_COLORS.values()) == expected
    assert default_color_manager().get_cmap("patient") == CASE_COLORS


def test_patient_palette_is_case_id_based():
    """Subsets and reversed orders preserve a patient's exact color."""
    first = patient_palette(["AS_005", "AS_001", "AS_005"])
    second = patient_palette(["AS_001", "AS_005"])
    assert list(first) == ["AS_005", "AS_001"]
    assert first["AS_001"] == second["AS_001"] == CASE_COLORS["AS_001"]
    assert first["AS_005"] == second["AS_005"] == CASE_COLORS["AS_005"]
    with pytest.raises(ValueError, match="Unregistered"):
        patient_palette(["unknown"])
