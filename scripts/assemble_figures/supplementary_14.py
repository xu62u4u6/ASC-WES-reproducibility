"""Assemble saved S14 package panels: A/B above C/D/E/F."""

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw

from asc_reproduction.figure_style import panel_raster_font


def assemble(source: Path, output: Path) -> None:
    """Preserve source proportions and record the six input image hashes."""
    output.mkdir(parents=True, exist_ok=False)
    canvas = Image.new("RGB", (15000, 7700), "white")
    draw = ImageDraw.Draw(canvas)
    hashes = {}
    for index, letter in enumerate("ABCDEF"):
        path = source / f"S14{letter}.tiff"
        hashes[letter] = hashlib.sha256(path.read_bytes()).hexdigest()
        with Image.open(path) as image:
            assert image.size == (3600, 3000), image.size
            panel = image.convert("RGB")
        if index < 2:
            panel = panel.resize((5040, 4200), Image.Resampling.LANCZOS)
            x, y = 150 + index * 5400, 100
        else:
            x, y = 150 + (index - 2) * 3750, 4500
        canvas.paste(panel, (x, y))
        # Place regular-serif letters left of each panel's plotting region.
        draw.text((x - 110, y + 80), letter,
                  font=panel_raster_font(230), fill="black")
    canvas.save(output / "S14.tiff", compression="tiff_lzw", dpi=(600, 600))
    canvas.thumbnail((2000, 2000), Image.Resampling.LANCZOS)
    canvas.save(output / "S14.png")
    manifest = {
        "sources": hashes,
        "layout": "A/B above C/D/E/F, preserving original upper-right space",
        "status": "candidate, not user approved",
        "statistics_recomputed_by_assembly": False,
    }
    (output / "assembly.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    """Read six existing panels; do not rerun statistics during assembly."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assemble(args.source, args.output)


if __name__ == "__main__":
    main()
