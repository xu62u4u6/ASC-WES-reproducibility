"""Assemble Figure 5 as the inspected original: A/B above C/D.

Reads high-resolution, already-validated panels; no analytical recomputation.
Downsamples to a fixed 600-dpi page without stretching plot content.
Only empty outer margins of A/B are trimmed to bring the PCA plots closer.
The original manuscript shares the subtype legend through panel B.
"""

import argparse
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont, __version__ as pillow_version

from asc_reproduction.provenance import file_hash, write_json
from asc_reproduction.figure_style import panel_font_path, panel_raster_font


def trim_white_margin(panel: Image.Image) -> tuple[Image.Image, tuple]:
    """Trim only exactly white borders, retaining all nonwhite pixels and padding."""
    background = Image.new("RGB", panel.size, "white")
    bounds = ImageChops.difference(panel, background).getbbox()
    if bounds is None:
        raise ValueError("A blank panel cannot be assembled")
    left, top, right, bottom = bounds
    padded = (
        max(0, left - 40), max(0, top - 40),
        min(panel.width, right + 40), min(panel.height, bottom + 40),
    )
    return panel.crop(padded), padded


def assemble_figure_05(pca: Path, scores: Path, output: Path) -> None:
    """Save a numbered TIFF/PNG and record exact panel hashes and transforms."""
    if output.exists():
        raise FileExistsError("Use a new assembly staging directory")
    panels = [
        pca / "Figure_5A_SNV_PCA.tiff",
        pca / "Figure_5B_CNV_PCA.tiff",
        scores / "Figure_5C_softscore.tiff",
        scores / "Figure_5D_softscore.tiff",
    ]
    if not all(path.is_file() for path in panels):
        raise FileNotFoundError("All four high-resolution panels are required")
    hashes = [file_hash(path) for path in panels]
    upstream_hashes = {
        str(path): file_hash(path)
        for path in (pca / "summary.json", scores / "COMPLETE.json")
    }
    # Fixed width at 600 dpi; derive row heights to avoid artificial gaps.
    cell_width = 3600
    margin, label_height = 120, 180
    prepared = []
    for index, path in enumerate(panels):
        with Image.open(path) as source:
            panel = source.convert("RGB")
        original_size = panel.size
        if panel.width < cell_width:
            raise ValueError("Panel is too small; refusing to upscale")
        crop = (0, 0, panel.width, panel.height)
        if index < 2:
            panel, crop = trim_white_margin(panel)
        else:
            panel.thumbnail((3060, 2380), Image.Resampling.LANCZOS)
        prepared.append((panel, original_size, crop))
    # A/B share a scale so equal-size source plot areas remain equal-size.
    top_scale = min(
        1.0,
        (2 * cell_width) / sum(item[0].width for item in prepared[:2]),
        3000 / max(item[0].height for item in prepared[:2]),
    )
    for index in (0, 1):
        panel, original_size, crop = prepared[index]
        size = (round(panel.width * top_scale), round(panel.height * top_scale))
        prepared[index] = (
            panel.resize(size, Image.Resampling.LANCZOS), original_size, crop
        )
    row_heights = [
        max(prepared[index][0].height for index in pair)
        for pair in ((0, 1), (2, 3))
    ]
    canvas = Image.new(
        "RGB", (2 * cell_width + 3 * margin,
                sum(row_heights) + label_height + 100 + 3 * margin), "white"
    )
    font_path = panel_font_path()
    font = panel_raster_font(170)
    drawing = ImageDraw.Draw(canvas)
    transforms = []
    for index, (letter, path) in enumerate(zip("ABCD", panels)):
        panel, original_size, crop = prepared[index]
        x = margin + (index % 2) * (cell_width + margin)
        if index in (1, 3):
            x = 2 * margin + prepared[0][0].width
        y = margin
        if index >= 2:
            y += row_heights[0] + label_height + margin
        # Anchor labels to their panel, not the wider reserved column.
        # Bottom labels occupy the blank upper-left corner beside the y-axis.
        panel_label_height = label_height if index < 2 else 100
        paste_at = (x, y + panel_label_height)
        canvas.paste(panel, paste_at)
        drawing.text((x, y), letter, font=font, fill="black")
        transforms.append({
            "panel": letter, "source": str(path), "sha256": hashes[index],
            "original_pixels": original_size, "placed_pixels": panel.size,
            "position": paste_at, "source_crop_box": crop,
            "label_position": (x, y),
            "cropping": "white outer margin only" if index < 2 else False,
        })
    if hashes != [file_hash(path) for path in panels]:
        raise ValueError("Panel source changed during assembly")
    output.mkdir(parents=True)
    canvas.save(output / "Figure_05.tiff", dpi=(600, 600),
                compression="tiff_lzw")
    canvas.thumbnail((1800, 1800), Image.Resampling.LANCZOS)
    canvas.save(output / "Figure_05.png")
    write_json(output / "assembly.json", {
        "layout": "original manuscript: 2 rows x 2 columns, A/B then C/D",
        "source_reference": "reproduce_submission/audit/manuscript/image5.tif",
        "panels": transforms, "model_fitting": False,
        "script_sha256": file_hash(Path(__file__)),
        "font_sha256": file_hash(font_path),
        "Pillow_version": pillow_version,
        "upstream_manifests": upstream_hashes,
        "layout_adjustment": (
            "larger/closer A/B, smaller C/D, panel-anchored labels "
            "and reduced bottom label whitespace"
        ),
        "visual_review": "required after rendering; not automated approval",
        "artifacts": {path.name: file_hash(path) for path in output.iterdir()},
    })


def main() -> None:
    """Read explicit panel directories and a new assembly output location."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pca", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assemble_figure_05(args.pca, args.scores, args.output)


if __name__ == "__main__":
    main()
