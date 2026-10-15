"""Compose S17 in the original five-boxplot plus two-summary-row layout."""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

from asc_reproduction.figure_style import panel_raster_font


def assemble(source: Path, output: Path):
    """Place saved panels by data rectangles without recomputing statistics."""
    records = json.loads((source / 'panels.json').read_text())
    canvas = Image.new('RGB', (11350, 14000), 'white')
    positions = [
        ('jaccard', 0, 0), ('smc', 0, 1),
        ('gene', 1, 0), ('cluster', 1, 1), ('arm', 1, 2),
    ]
    for row, name in enumerate(['gene', 'arm'], start=2):
        for col, kind in enumerate(['matrix', 'mean', 'p']):
            positions.append((f'{name}_{kind}', row, col))
    for name, row, col in positions:
        record = records[name]
        with Image.open(source / record['tiff']) as image:
            panel = image.convert('RGB')
        left, top, right, _ = record['plot_box_pixels']
        scale = 2800 / (right - left)
        panel = panel.resize(
            (round(panel.width * scale), round(panel.height * scale)),
            Image.Resampling.LANCZOS,
        )
        x = round(650 + col * 3650 - left * scale)
        y = round(350 + row * 3380 - top * scale)
        assert min(x, y) >= 0
        assert x + panel.width <= canvas.width
        assert y + panel.height <= canvas.height
        canvas.paste(panel, (x, y))
    draw = ImageDraw.Draw(canvas)
    for label, y in [('A', 170), ('B', 6930)]:
        draw.text((30, y), label, font=panel_raster_font(290),
                  fill='black', anchor='lt')
    output.mkdir(parents=True, exist_ok=False)
    canvas.save(output / 'S17.tiff', compression='tiff_lzw', dpi=(600, 600))
    canvas.thumbnail((1600, 2000), Image.Resampling.LANCZOS)
    canvas.save(output / 'S17.png')


def main():
    """Compose saved S17 panels into a new candidate directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assemble(args.source, args.output)


if __name__ == '__main__':
    main()
