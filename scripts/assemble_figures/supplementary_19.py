"""Assemble four saved S19 rows without fitting or testing any model."""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from asc_reproduction.figure_style import panel_raster_font
from asc_reproduction.provenance import file_hash, write_json

PANEL_LABEL_X = 160


def assemble(source, output, approved_on=None):
    """Stack A–D at equal width and keep labels outside the saved panels."""
    output.mkdir(parents=True, exist_ok=False)
    panels = []
    for letter in "ABCD":
        with Image.open(source / f"S19{letter}.tiff") as image:
            panel = image.convert("RGB")
        panels.append(panel)
    assert len({panel.width for panel in panels}) == 1
    width = panels[0].width
    canvas = Image.new(
        "RGB", (width + 240, sum(p.height for p in panels) + 240), "white"
    )
    draw = ImageDraw.Draw(canvas)
    top = 60
    for letter, panel in zip("ABCD", panels):
        canvas.paste(panel, (220, top))
        draw.text(
            (PANEL_LABEL_X, top + 20),
            letter,
            font=panel_raster_font(140),
            fill="black",
        )
        top += panel.height + 40
    canvas.save(output / "S19.tiff", compression="tiff_lzw", dpi=(600, 600))
    canvas.thumbnail((1800, 2200), Image.Resampling.LANCZOS)
    canvas.save(output / "S19.png")
    write_json(
        output / "assembly.json",
        {
            "sources": {
                letter: file_hash(source / f"S19{letter}.tiff") for letter in "ABCD"
            },
            "status": (
                f"approved by user on {approved_on}"
                if approved_on
                else "candidate; user approval pending"
            ),
            "approved_on": approved_on,
            "model_fitting": False,
            "label_x_pixels": PANEL_LABEL_X,
            "dpi": 600,
        },
    )


def main():
    """Read saved rows and compose a new candidate directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--approved-on",
        help="Record the date of explicit user approval, if approved.",
    )
    args = parser.parse_args()
    assemble(args.source, args.output, approved_on=args.approved_on)


if __name__ == "__main__":
    main()
