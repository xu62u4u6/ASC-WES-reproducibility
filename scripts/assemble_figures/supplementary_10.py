"""Assemble saved S10 C47 boxplot (A) and 3q gain proportions (B)."""

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw

from asc_reproduction.figure_style import panel_raster_font


def assemble(source: Path, output: Path) -> None:
    """Align saved data rectangles without resizing panels or repeating tests."""
    summary = json.loads((source / "summary.json").read_text())
    records = {r["tiff"]: r for r in summary["figures"]}
    panels = []
    for letter in "AB":
        name = f"S10{letter}.tiff"
        path = source / name
        with Image.open(path) as image:
            panel = image.convert("RGB")
        box = records[name]["plot_box_pixels"]
        panels.append((panel, box, path))
    sizes = [(b[2] - b[0], b[3] - b[1]) for _, b, _ in panels]
    assert all(abs(a - b) < 1 for a, b in zip(sizes[0], sizes[1]))
    offsets = [(round(540 - b[0]), round(300 - b[1])) for _, b, _ in panels]
    width = max(im.width + dx for (im, _, _), (dx, _) in zip(panels, offsets))
    height = max(im.height + dy for (im, _, _), (_, dy) in zip(panels, offsets))
    canvas = Image.new("RGB", (width * 2, height), "white")
    draw = ImageDraw.Draw(canvas)
    for i, ((panel, _, _), (dx, dy)) in enumerate(zip(panels, offsets)):
        canvas.paste(panel, (i * width + dx, dy))
        draw.text(
            (i * width + 140, 150),
            "AB"[i],
            fill="black",
            font=panel_raster_font(140),
            anchor="lt",
        )
    output.mkdir(parents=True, exist_ok=False)
    canvas.save(output / "S10.tiff", compression="tiff_lzw", dpi=(600, 600))
    canvas.thumbnail((2200, 2200), Image.Resampling.LANCZOS)
    canvas.save(output / "S10.png")
    manifest = {
        "status": "candidate; awaiting full-figure approval",
        "order": {"A": "C47 component boxplot", "B": "3q gain proportion"},
        "sources": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for _, _, p in panels
        },
        "axis_origin_pixels": [540, 300],
        "label_origin_pixels": [140, 150],
        "dpi": 600,
    }
    (output / "assembly.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    """Read independent panels and write the two-panel manuscript candidate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assemble(args.source, args.output)


if __name__ == "__main__":
    main()
