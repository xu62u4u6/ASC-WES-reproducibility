"""Assemble existing Figure 4 panels: full-width A, B/C, full-width D.

This entry performs no analysis or fitting. Inputs are explicit TIFF paths;
their aspect ratios are preserved and never upscaled. Use a new output folder.
"""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from asc_reproduction.provenance import file_hash, write_json
from asc_reproduction.figure_style import panel_font_path, panel_raster_font


def assemble_figure_04(panels: list[Path], output: Path) -> None:
    """Save a 600-dpi RGB composite plus provenance for the four source panels."""
    if output.exists():
        raise FileExistsError("Use a new output directory")
    if len(panels) != 4 or not all(path.is_file() for path in panels):
        raise ValueError("Provide exactly four existing TIFFs in A–D order")
    hashes = [file_hash(path) for path in panels]
    width, margin, label_height = 7200, 120, 180
    prepared = []
    for index, path in enumerate(panels):
        with Image.open(path) as source:
            panel = source.convert("RGB")
        original = panel.size
        target = width if index in (0, 3) else (width - margin) // 2
        if panel.width < target:
            raise ValueError(f"Source panel {index} too small; no upscaling")
        panel.thumbnail((target, 10000), Image.Resampling.LANCZOS)
        prepared.append((panel, original))
    heights = [prepared[0][0].height,
               max(prepared[1][0].height, prepared[2][0].height),
               prepared[3][0].height]
    canvas = Image.new(
        "RGB", (width + 2 * margin,
                sum(heights) + 3 * label_height + 4 * margin), "white"
    )
    font_path = panel_font_path()
    font = panel_raster_font(170)
    drawing = ImageDraw.Draw(canvas)
    rows = [margin,
            margin + heights[0] + label_height + margin,
            margin + sum(heights[:2]) + 2 * (label_height + margin)]
    placements = []
    for index, letter in enumerate("ABCD"):
        row = (0, 1, 1, 2)[index]
        x = margin + ((width + margin) // 2 if index == 2 else 0)
        panel, original = prepared[index]
        position = (x, rows[row] + label_height)
        canvas.paste(panel, position)
        drawing.text((x, rows[row]), letter, fill="black", font=font)
        placements.append({
            "panel": letter, "source": str(panels[index]),
            "sha256": hashes[index], "original_pixels": original,
            "placed_pixels": panel.size, "position": position,
            "cropping": False,
        })
    if hashes != [file_hash(path) for path in panels]:
        raise RuntimeError("Panel inputs changed during assembly")
    output.mkdir(parents=True)
    canvas.save(output / "Figure_04.tiff", dpi=(600, 600),
                compression="tiff_lzw")
    canvas.thumbnail((1600, 2000), Image.Resampling.LANCZOS)
    canvas.save(output / "Figure_04.png")
    write_json(output / "assembly.json", {
        "layout": "original manuscript: A / B,C / D",
        "panels": placements, "model_fitting": False,
        "script_sha256": file_hash(Path(__file__)),
        "font_sha256": file_hash(font_path),
        "visual_review": "required; successful rendering is not approval",
        "artifacts": {p.name: file_hash(p) for p in output.iterdir()},
    })


def main() -> None:
    """Read four explicit source TIFFs and a nonexisting staging destination."""
    parser = argparse.ArgumentParser(description=__doc__)
    for letter in "abcd":
        parser.add_argument(f"--{letter}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assemble_figure_04([args.a, args.b, args.c, args.d], args.output)


if __name__ == "__main__":
    main()
