"""Assemble saved package-rendered S5 panels; no analysis is rerun here."""

import argparse
import hashlib
import json
from pathlib import Path

from matplotlib.font_manager import FontProperties, findfont
from PIL import Image, ImageDraw, ImageFont

from asc_reproduction.figure_style import BODY_FONT_FAMILY, panel_raster_font
from asc_reproduction.shared_private_figures import GROUP_TITLES


def assemble(source: Path, output: Path) -> None:
    """Keep A above B and align the three B matrices to identical case widths."""
    records = json.loads((source / "panels.json").read_text())
    output.mkdir(parents=True, exist_ok=False)
    canvas = Image.new("RGB", (10000, 12000), "white")
    draw = ImageDraw.Draw(canvas)
    heading = ImageFont.truetype(
        findfont(FontProperties(family=BODY_FONT_FAMILY, weight="bold")), 112
    )
    body = ImageFont.truetype(
        findfont(FontProperties(family=BODY_FONT_FAMILY)), 108
    )
    letter = panel_raster_font(220)
    with Image.open(source / records["A"]["tiff"]) as image:
        enlarged = image.convert("RGB").resize(
            (round(image.width * 1.04), round(image.height * 1.04)),
            Image.Resampling.LANCZOS,
        )
        canvas.paste(enlarged, (150, 230))
        y = enlarged.height + 380
    draw.text((520, 180), "A", font=letter, fill="black", anchor="lt")
    draw.text((520, y), "B", font=letter, fill="black", anchor="lt")
    draw.rectangle((6200, y + 30, 6360, y + 100), fill="gray")
    draw.text((6410, y), "Mutation present", font=body, fill="black")
    y += 210
    a_left, _, a_right, _ = records["A"]["plot_box_pixels"]
    matrix_left = round(150 + a_left * 1.04)
    matrix_width = round((a_right - a_left) * 1.04)
    geometry = {}
    for name, title in GROUP_TITLES.items():
        record = records[name]
        left, top, right, bottom = record["plot_box_pixels"]
        scale = matrix_width / (right - left)
        with Image.open(source / record["tiff"]) as image:
            # Upper groups share the bottom group's case labels and freq title.
            extra = 650 if name == "S_private" else 0
            crop = image.convert("RGB").crop(
                (0, 0, image.width, min(image.height, round(bottom + extra)))
            )
        panel = crop.resize(
            (round(crop.width * scale), round(crop.height * scale)),
            Image.Resampling.LANCZOS,
        )
        x = round(matrix_left - left * scale)
        if x < 0 or x + panel.width > canvas.width:
            raise ValueError("B labels or frequency column exceed the canvas")
        draw.text((matrix_left, y), title, font=heading, fill="black", anchor="lt")
        image_y = round(y + 160 - top * scale)
        canvas.paste(panel, (x, image_y))
        geometry[name] = {
            "matrix_left": matrix_left, "matrix_width": matrix_width,
            "matrix_top": y + 160, "uniform_scale": scale,
        }
        y = image_y + panel.height + 120
    canvas = canvas.crop((0, 0, canvas.width, y))
    canvas.save(output / "S5.tiff", compression="tiff_lzw", dpi=(600, 600))
    canvas.thumbnail((1900, 2200), Image.Resampling.LANCZOS)
    canvas.save(output / "S5.png")
    manifest = {
        "status": "candidate; requires user approval before Word replacement",
        "geometry": geometry,
        "A_uniform_scale": 1.04,
        "layout_change": "B data edges aligned to A; wide native render keeps short rows",
        "sources": {
            name: hashlib.sha256((source / record["tiff"]).read_bytes()).hexdigest()
            for name, record in records.items() if name != "provenance"
        },
    }
    (output / "assembly.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    """Provide a separate, explicit entrypoint for S5 page layout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assemble(args.source, args.output)


if __name__ == "__main__":
    main()
