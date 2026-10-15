"""Compose S15 using saved axis geometry, without recalculating CNV."""

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw

from asc_reproduction.figure_style import panel_raster_font


def assemble(source: Path, output: Path) -> None:
    """Align A/B heatmaps and arrange C beside the D/E, F/G comparison grid."""
    metadata = json.loads((source / "panels.json").read_text())
    f_box = metadata["F"]["plot_box_pixels"]
    g_box = metadata["G"]["plot_box_pixels"]
    f_ratio = (f_box[3] - f_box[1]) / (f_box[2] - f_box[0])
    g_ratio = (g_box[3] - g_box[1]) / (g_box[2] - g_box[0])
    assert abs(f_ratio - g_ratio) < 1e-6, "F/G data areas must have equal aspect"
    placements = {}
    for letter in "ABCDEFG":
        path = source / f"S15{letter}.tiff"
        with Image.open(path) as image:
            panel = image.convert("RGB")
        left, top, right, bottom = metadata[letter]["plot_box_pixels"]
        if letter in "AB":
            target_width = 8100
            axis_x, axis_y = 550, 200 if letter == "A" else 4200
        elif letter == "C":
            target_width = 2650
            axis_x, axis_y = 300, 8300
        else:
            column = 0 if letter in "DF" else 1
            row = 0 if letter in "DE" else 1
            target_width = 2750
            axis_x, axis_y = 3600 + column * 3250, 8300 + row * 3000
        scale = target_width / (right - left)
        size = (round(panel.width * scale), round(panel.height * scale))
        panel = panel.resize(size, Image.Resampling.LANCZOS)
        x, y = round(axis_x - left * scale), round(axis_y - top * scale)
        placements[letter] = {
            "image": panel, "xy": (x, y), "axis": (axis_x, axis_y),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "scale": scale,
        }
    width = max(p["xy"][0] + p["image"].width for p in placements.values()) + 100
    height = max(p["xy"][1] + p["image"].height for p in placements.values()) + 100
    canvas = Image.new("RGB", (max(10000, width), height), "white")
    draw = ImageDraw.Draw(canvas)
    records = {}
    for letter, placement in placements.items():
        canvas.paste(placement["image"], placement["xy"])
        axis_x, axis_y = placement["axis"]
        label_x = 30 if letter in "ABC" else axis_x - 430
        draw.text((label_x, axis_y - 35), letter,
                  font=panel_raster_font(190), fill="black", anchor="lt")
        records[letter] = {k: v for k, v in placement.items() if k != "image"}
    output.mkdir(parents=True, exist_ok=False)
    canvas.save(output / "S15.tiff", compression="tiff_lzw", dpi=(600, 600))
    canvas.thumbnail((1800, 2400), Image.Resampling.LANCZOS)
    canvas.save(output / "S15.png")
    (output / "assembly.json").write_text(json.dumps(records, indent=2) + "\n")


def main():
    """Assemble seven existing panels into a candidate figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assemble(args.source, args.output)


if __name__ == "__main__":
    main()
