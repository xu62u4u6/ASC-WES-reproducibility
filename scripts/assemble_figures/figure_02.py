"""Assemble Figure 2 from explicit TIFFs: driver oncoplot above CCF panels."""

import argparse
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont

from asc_reproduction.provenance import file_hash, write_json
from asc_reproduction.figure_style import panel_raster_font


def assemble(a: Path, b: Path, output: Path) -> None:
    """Preserve both panel aspect ratios and save a 600-dpi RGB composite."""
    if output.exists():
        raise FileExistsError("Use a new output directory")
    sources = [a, b]
    hashes = [file_hash(path) for path in sources]
    panels = []
    width, margin, label_height = 7200, 100, 180
    for path in sources:
        with Image.open(path) as image:
            panel = image.convert("RGB")
        original = panel.size
        bounds = ImageChops.difference(
            panel, Image.new("RGB", panel.size, "white")
        ).getbbox()
        if bounds is None:
            raise ValueError("Cannot assemble a blank panel")
        left, top, right, bottom = bounds
        crop = (max(0, left - 40), max(0, top - 40),
                min(panel.width, right + 40), min(panel.height, bottom + 40))
        panel = panel.crop(crop)
        if panel.width < width:
            raise ValueError("Refusing to upscale a low-resolution source")
        target_width = width - 220 if len(panels) == 1 else width
        panel.thumbnail((target_width, 10000), Image.Resampling.LANCZOS)
        panels.append((panel, original, crop))
    canvas = Image.new("RGB", (width + 2 * margin,
                       sum(p.height for p, _, _ in panels)
                       + 2 * label_height + 3 * margin), "white")
    drawing = ImageDraw.Draw(canvas)
    font = panel_raster_font(170)
    y = margin
    transforms = []
    for index, (panel, original, crop) in enumerate(panels):
        drawing.text((margin, y), "AB"[index], font=font, fill="black")
        position = (margin + (220 if index == 1 else 0), y + label_height)
        canvas.paste(panel, position)
        transforms.append({"panel": "AB"[index], "source": str(sources[index]),
                           "sha256": hashes[index], "original_pixels": original,
                           "placed_pixels": panel.size, "position": position,
                           "white_margin_crop": crop})
        y += label_height + panel.height + margin
    if hashes != [file_hash(path) for path in sources]:
        raise RuntimeError("Input panel changed during assembly")
    output.mkdir(parents=True)
    canvas.save(output / "Figure_02.tiff", dpi=(600, 600), compression="tiff_lzw")
    canvas.thumbnail((1600, 1800), Image.Resampling.LANCZOS)
    canvas.save(output / "Figure_02.png")
    write_json(output / "assembly.json", {
        "layout": "original manuscript: A above B", "panels": transforms,
        "script_sha256": file_hash(Path(__file__)), "model_fitting": False,
        "artifacts": {p.name: file_hash(p) for p in output.iterdir()},
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("a", "b", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    args = parser.parse_args()
    assemble(args.a, args.b, args.output)
