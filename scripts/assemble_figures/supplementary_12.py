"""Assemble package-rendered S12 panels with matched main and legend columns."""

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw

from asc_reproduction.figure_style import panel_raster_font


def assemble(source: Path, output: Path) -> None:
    """Align saved axes without rescaling or recalculating protein events."""
    report = json.loads((source / "summary.json").read_text())
    panels = []
    for record in report["panels"]:
        path = source / record["figure"]["tiff"]
        with Image.open(path) as image:
            panel = image.convert("RGB")
        box = record["figure"]["plot_box_pixels"]
        panels.append((panel, box, path))
    assert len(panels) == 2
    widths = [box[2] - box[0] for _, box, _ in panels]
    assert abs(widths[0] - widths[1]) < 1
    offsets = [(round(500 - box[0]), round(270 - box[1]))
               for _, box, _ in panels]
    width = max(im.width + dx for (im, _, _), (dx, _) in zip(panels, offsets))
    height = max(im.height + dy for (im, _, _), (_, dy) in zip(panels, offsets))
    canvas = Image.new("RGB", (width, height * 2 + 100), "white")
    draw = ImageDraw.Draw(canvas)
    for i, ((panel, _, _), (dx, dy)) in enumerate(zip(panels, offsets)):
        top = i * (height + 100)
        canvas.paste(panel, (dx, top + dy))
        draw.text((100, top + 160), "AB"[i], fill="black",
                  font=panel_raster_font(140), anchor="lt")
    output.mkdir(parents=True, exist_ok=False)
    canvas.save(output / "S12.tiff", compression="tiff_lzw", dpi=(600, 600))
    canvas.thumbnail((2200, 2200), Image.Resampling.LANCZOS)
    canvas.save(output / "S12.png")
    manifest = {
        "status": "candidate; awaiting approval",
        "order": ["A: EGFR", "B: TP53"],
        "sources": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for _, _, p in panels},
        "axis_origin_pixels": [500, 270],
        "label_origin_pixels": [100, 160],
        "equal_main_width_pixels": widths,
        "dpi": 600,
    }
    (output / "assembly.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    """Parse independent panel and assembled-output directories."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assemble(args.source, args.output)


if __name__ == "__main__":
    main()
