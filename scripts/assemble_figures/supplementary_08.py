"""Assemble saved S8 panels in original A/B above C/D order."""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from asc_reproduction.figure_style import panel_raster_font
from asc_reproduction.provenance import file_hash, write_json


def assemble(source, output):
    """Keep source proportions and reserve shared space for aligned letters."""
    panels = []
    for letter in "ABCD":
        with Image.open(source / f"S8{letter}.tiff") as image:
            panels.append(image.convert("RGB"))
    width = max(p.width for p in panels) + 180
    height = max(p.height for p in panels) + 80
    canvas = Image.new("RGB", (width * 2, height * 2), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (letter, panel) in enumerate(zip("ABCD", panels)):
        x, y = (i % 2) * width, (i // 2) * height
        canvas.paste(panel, (x + 160, y + 40))
        draw.text((x + 260, y + 40), letter, font=panel_raster_font(140), fill="black")
    output.mkdir(parents=True, exist_ok=False)
    canvas.save(output / "S8.tiff", compression="tiff_lzw", dpi=(600, 600))
    canvas.thumbnail((1800, 1800), Image.Resampling.LANCZOS)
    canvas.save(output / "S8.png")
    write_json(
        output / "assembly.json",
        {
            "sources": {p: file_hash(source / f"S8{p}.tiff") for p in "ABCD"},
            "status": "candidate; not user approved",
            "layout": "A/B above C/D",
        },
    )


def main():
    """Read independent panels without repeating tests or calculations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assemble(args.source, args.output)


if __name__ == "__main__":
    main()
