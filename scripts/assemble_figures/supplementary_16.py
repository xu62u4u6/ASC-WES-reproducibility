"""Compose S16 descriptive audit; keep missing inference explicitly visible."""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

from asc_reproduction.figure_style import panel_raster_font


def assemble(source: Path, output: Path, summaries: Path | None = None) -> None:
    """Align sample matrices and mean heatmaps from saved data-area geometry."""
    metadata = json.loads((source / "panels.json").read_text())
    if summaries is not None:
        metadata.update(json.loads((summaries / "panels.json").read_text()))
    kinds = ("matrix", "mean", "p") if summaries else ("matrix", "mean")
    canvas = Image.new("RGB", (200 + 3700 * len(kinds), 10600), "white")
    draw = ImageDraw.Draw(canvas)
    for row, letter in enumerate("ABC"):
        for column, kind in enumerate(kinds):
            folder = summaries if summaries and kind != "matrix" else source
            with Image.open(folder / f"S16{letter}_{kind}.tiff") as image:
                panel = image.convert("RGB")
            left, top, right, bottom = metadata[f"{letter}_{kind}"]["plot_box_pixels"]
            scale = 2800 / (right - left)
            panel = panel.resize(
                (round(panel.width * scale), round(panel.height * scale)),
                Image.Resampling.LANCZOS,
            )
            x = 450 + column * 3700
            y = 300 + row * 3400
            origin = (round(x - left * scale), round(y - top * scale))
            assert origin[0] >= 0 and origin[1] >= 0
            assert origin[0] + panel.width <= canvas.width
            assert origin[1] + panel.height <= canvas.height
            canvas.paste(panel, origin)
        draw.text((30, 190 + row * 3400), letter,
                  font=panel_raster_font(230), fill="black", anchor="lt")
    output.mkdir(parents=True, exist_ok=False)
    name = "S16" if summaries else "S16_descriptive_only"
    canvas.save(output / f"{name}.tiff", compression="tiff_lzw",
                dpi=(600, 600))
    canvas.thumbnail((1500, 2100), Image.Resampling.LANCZOS)
    canvas.save(output / f"{name}.png")


def main():
    """Read six panels without changing their data or performing tests."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summaries", type=Path)
    args = parser.parse_args()
    assemble(args.source, args.output, args.summaries)


if __name__ == "__main__":
    main()
