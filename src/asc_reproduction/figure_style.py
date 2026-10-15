"""Shared plotting setup, registered colors and manuscript image export.

Use regular DejaVu Serif for panel letters and DejaVu Sans for plot text.
Both raster assemblers and Matplotlib obtain fonts from the pinned Matplotlib
installation, avoiding different system fonts or silent fallback.
"""

from collections.abc import Iterable
from io import BytesIO
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.font_manager import FontProperties
from PIL import Image, ImageFont
from pymaftools.plot.ColorManager import ColorManager

BODY_FONT_FAMILY = "DejaVu Sans"
PANEL_FONT_FAMILY = "DejaVu Serif"
PANEL_FONT_WEIGHT = "normal"

MUTATION_COLORS = {
    "WT": "white",
    "Missense_Mutation": "gray",
    "Frame_Shift_Ins": "#FF4500",
    "Frame_Shift_Del": "#3E85C0",
    "In_Frame_Ins": "#E8656E",
    "In_Frame_Del": "#ADDBEA",
    "Nonsense_Mutation": "#98E28F",
    "Splice_Site": "#D0875D",
    "Multi_Hit": "#222222",
}

CASE_COLORS = {
    "AS_001": "#1f77b4",
    "AS_005": "#ff7f0e",
    "AS_006": "#2ca02c",
    "AS_007": "#d62728",
    "AS_008": "#9467bd",
    "AS_009": "#8c564b",
    "AS_010": "#e377c2",
    "AS_016": "#7f7f7f",
    "AS_020": "#bcbd22",
    "AS_022": "#17becf",
    "AS_027": "#9edae5",
}


def default_color_manager() -> ColorManager:
    """Register the original ASC categories and approved S6 patient colors.

    Return a fresh manager so a caller cannot mutate another figure's maps.
    Transparency is applied by each renderer using its approved settings.
    """
    cm = ColorManager()
    cm.add_cmap("subtype", {"LUAD": "orange", "ASC": "green", "LUSC": "blue"})
    cm.add_cmap("smoke", {0: "gray", 1: "red"})
    cm.add_cmap("sex", {"M": "blue", "F": "red"})
    cm.add_cmap(
        "sample_type",
        {
            "A": "salmon",
            "T": "seagreen",
            "S": "steelblue",
        },
    )
    cm.add_cmap("patient", CASE_COLORS.copy())
    return cm


PART_COLORS = default_color_manager().get_cmap("sample_type")
SUBTYPE_COLORS = default_color_manager().get_cmap("subtype")
TRACK_COLORS = {
    name: default_color_manager().get_cmap(name) for name in ("subtype", "sex", "smoke")
}


def patient_palette(case_ids: Iterable[str]) -> dict[str, str]:
    """Look up fixed case colors in display order, rejecting unknown IDs."""
    cases = list(dict.fromkeys(case_ids))
    if not cases or any(pd.isna(case) for case in cases):
        raise ValueError("Patient colors require nonempty, nonmissing case IDs")
    registered = default_color_manager().get_cmap("patient")
    unknown = set(cases).difference(registered)
    if unknown:
        raise ValueError(f"Unregistered patient case IDs: {sorted(unknown)}")
    return {case: registered[case] for case in cases}


def setup_plotting(style=None, palette=None, figsize=None) -> None:
    """Initialize each plotting process explicitly, preserving approved sizes.

    Optional style/palette/figsize follow ASCDataLoader.setup_plotting.
    None preserves panel-specific settings. The headless backend and shared
    fonts are always set; importing an analysis module performs no setup.
    """
    plt.switch_backend("Agg")
    if style is not None or palette is not None:
        import seaborn as sns

        if style is not None:
            sns.set_style(style)
        if palette is not None:
            sns.set_palette(palette)
    if figsize is not None:
        plt.rcParams["figure.figsize"] = figsize
    configure_body_font()


def save_tiff_and_preview(
    fig: plt.Figure, output_base: Path, *, plot_axis=None
) -> dict[str, object]:
    """Save PNG/TIFF; optionally record a data rectangle for aligned assembly.

    With plot_axis, freeze the tight bounding box at TIFF resolution so its
    pixel coordinates can be used without guessing borders from image colors.
    Untagged figures retain the existing export behavior.
    """
    png_path = Path(f"{output_base}.png")
    tiff_path = Path(f"{output_base}.tiff")
    bounds = "tight"
    geometry = {}
    if plot_axis is not None:
        if plot_axis.figure is not fig:
            raise ValueError("Assembly axis belongs to a different figure")
        fig.set_dpi(600)
        fig.canvas.draw()
        bounds = fig.get_tightbbox(fig.canvas.get_renderer()).padded(0.1)
        axis_bounds = plot_axis.get_window_extent().transformed(
            fig.dpi_scale_trans.inverted()
        )
        geometry["plot_box_pixels"] = [
            (axis_bounds.x0 - bounds.x0) * 600,
            (bounds.y1 - axis_bounds.y1) * 600,
            (axis_bounds.x1 - bounds.x0) * 600,
            (bounds.y1 - axis_bounds.y0) * 600,
        ]
    fig.savefig(png_path, dpi=150, bbox_inches=bounds, facecolor="white")
    buffer = BytesIO()
    fig.savefig(
        buffer,
        format="tiff",
        dpi=600,
        bbox_inches=bounds,
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    buffer.seek(0)
    with Image.open(buffer) as image:
        image.convert("RGB").save(tiff_path, compression="tiff_lzw", dpi=(600, 600))
    with Image.open(tiff_path) as image:
        image.load()
        if image.mode != "RGB":
            raise ValueError(f"Expected RGB TIFF, received mode {image.mode}")
        dpi = [int(round(float(value))) for value in image.info.get("dpi", ())]
        return {
            **geometry,
            "tiff": tiff_path.name,
            "preview": png_path.name,
            "pixels": list(image.size),
            "dpi": dpi,
            "mode": image.mode,
            "decode": "passed",
        }


def configure_body_font() -> None:
    """Set shared plot-text defaults without changing sizes or other styles."""
    matplotlib.rcParams["font.family"] = [BODY_FONT_FAMILY]
    matplotlib.rcParams["font.sans-serif"] = [BODY_FONT_FAMILY]


def panel_font_path() -> Path:
    """Return the bundled regular serif font or fail instead of substituting."""
    path = Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSerif.ttf"
    if not path.is_file():
        raise FileNotFoundError(f"Required manuscript panel font absent: {path}")
    return path


def panel_font_properties(size: float) -> FontProperties:
    """Return Matplotlib panel-letter properties at the requested point size."""
    return FontProperties(
        fname=str(panel_font_path()),
        size=size,
        family=PANEL_FONT_FAMILY,
        weight=PANEL_FONT_WEIGHT,
    )


def panel_raster_font(size: int) -> ImageFont.FreeTypeFont:
    """Return the identical font for Pillow assemblers, with size in pixels."""
    return ImageFont.truetype(str(panel_font_path()), size)
