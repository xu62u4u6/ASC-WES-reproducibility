"""Compose saved pooled A left of merged score/shape B, without analysis."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from asc_reproduction.figure_style import panel_raster_font
from asc_reproduction.provenance import file_hash, write_json


def assemble(panel_a, source, output):
    """Preserve panel aspect ratios and shared regular-serif lettering."""
    if output.exists():
        raise FileExistsError(output)
    with Image.open(panel_a) as image:
        a = image.convert("RGB")
    with Image.open(source / "S18B.tiff") as image:
        b = image.convert("RGB")
    geometry = json.loads((source / "panel_geometry.json").read_text())["B"]
    # Locate A's long dark axes, not the image boundary or title.
    dark = np.asarray(a.convert("L")) < 110
    rows = np.flatnonzero(dark.sum(axis=1) > a.width * 0.65)
    columns = np.flatnonzero(dark.sum(axis=0) > a.height * 0.60)
    assert len(rows) > 1 and len(columns) > 1
    a_top, a_bottom = int(rows[0]), int(rows[-1])
    a_left = int(columns[0])
    b_box = geometry["plot_box_pixels"]
    scale = (b_box[3] - b_box[1]) / (a_bottom - a_top)
    a = a.resize(
        (round(a.width * scale), round(a.height * scale)), Image.Resampling.LANCZOS
    )
    top = max(a_top * scale, b_box[1]) + 60
    ax, ay = 420, round(top - a_top * scale)
    bx, by = ax + a.width + 440, round(top - b_box[1])
    canvas = Image.new(
        "RGB", (bx + b.width + 80, max(ay + a.height, by + b.height) + 60), "white"
    )
    canvas.paste(a, (ax, ay))
    canvas.paste(b, (bx, by))
    draw = ImageDraw.Draw(canvas)
    font = panel_raster_font(160)
    for label, label_x in (
        ("A", ax + a_left * scale - 540),
        ("B", bx + b_box[0] - 540),
    ):
        box = draw.textbbox((0, 0), label, font=font)
        draw.text(
            (round(label_x) - box[0], round(top - 70) - box[1]),
            label,
            font=font,
            fill="black",
        )
    output.mkdir(parents=True)
    canvas.save(output / "S18AB_merged.tiff", compression="tiff_lzw", dpi=(600, 600))
    canvas.thumbnail((1800, 1800), Image.Resampling.LANCZOS)
    canvas.save(output / "S18AB_merged.png")
    write_json(
        output / "assembly.json",
        {
            "A_sha256": file_hash(panel_a),
            "B_sha256": file_hash(source / "S18B.tiff"),
            "status": "generated; approval recorded separately",
            "A_source": str(panel_a),
            "B_source": str(source / "S18B.tiff"),
            "encoding": "B color=soft_score; shape=LUAD circle/ASC triangle/LUSC square",
            "analysis_recomputed": False,
            "layout": "A left, merged B right; equal data-axis heights",
            "A_axis_box_source": [a_left, a_top, int(columns[-1]), a_bottom],
            "A_scale": scale,
        },
    )


def main():
    """Read saved panels and write a new full-figure candidate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-a", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assemble(args.panel_a, args.source, args.output)


if __name__ == "__main__":
    main()
