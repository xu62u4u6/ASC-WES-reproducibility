"""Assemble Supplementary Figure 6 from its four verified panel TIFFs.

The original manuscript places CNV heatmaps A/B above paired PCA panels C/D.
This script only crops outer white margins and scales each panel uniformly.
It never recomputes CNV or PCA values and never overwrites an existing output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, __version__ as pillow_version

from asc_reproduction.figure_style import panel_font_path, panel_raster_font
from asc_reproduction.provenance import file_hash, write_json


PANEL_FILES = (
    ("A", "S6A_arm_CNV.tiff", "cnv"),
    ("B", "S6B_cluster_CNV.tiff", "cnv"),
    ("C", "S6C_SNV_PCA.tiff", "pca"),
    ("D", "S6D_CNV_PCA.tiff", "pca"),
)


def crop_outer_white(panel: Image.Image) -> tuple[Image.Image, tuple[int, ...]]:
    """Remove only empty outer pixels, retaining 35 pixels of white padding."""
    background = Image.new("RGB", panel.size, "white")
    bounds = ImageChops.difference(panel, background).getbbox()
    if bounds is None:
        raise ValueError("A blank panel cannot be assembled")
    left, top, right, bottom = bounds
    box = (
        max(0, left - 35),
        max(0, top - 35),
        min(panel.width, right + 35),
        min(panel.height, bottom + 35),
    )
    return panel.crop(box), box


def assemble_supplementary_06(cnv: Path, pca: Path, output: Path) -> None:
    """Save the A/B-over-C/D composite with source and transform provenance."""
    if output.exists():
        raise FileExistsError("Use a new, empty numbered assembly directory")
    directories = {"cnv": cnv, "pca": pca}
    sources = [directories[group] / name for _, name, group in PANEL_FILES]
    if not all(path.is_file() for path in sources):
        raise FileNotFoundError("All four S6 high-resolution TIFFs are required")
    source_hashes = [file_hash(path) for path in sources]
    manifests = [cnv / "summary.json", pca / "summary.json"]
    if not all(path.is_file() for path in manifests):
        raise FileNotFoundError("Both upstream panel summaries are required")

    cell_width = 3200
    margin = 120
    gap = 100
    label_height = 160
    prepared = []
    cnv_summary = json.loads((cnv / "summary.json").read_text())
    pca_summary = json.loads((pca / "summary.json").read_text())
    cnv_boxes = []
    data_boxes = []
    for index, path in enumerate(sources):
        with Image.open(path) as source:
            if source.n_frames != 1:
                raise ValueError(f"Expected one image frame: {path}")
            original_size = source.size
            panel, crop = crop_outer_white(source.convert("RGB"))
        if panel.width < cell_width:
            raise ValueError(f"Refusing to upscale low-resolution panel: {path}")
        if index < 2:
            entry = cnv_summary["panels"][path.stem]["figure"]
            left, top, right, bottom = entry["plot_box_pixels"]
            cnv_boxes.append(
                (left - crop[0], top - crop[1], right - crop[0], bottom - crop[1])
            )
        else:
            entry = pca_summary["panels"][path.stem]["figure"]
            if "plot_box_pixels" not in entry:
                raise ValueError(
                    "Regenerate PCA panels with data-axis geometry before assembly"
                )
            left, top, right, bottom = entry["plot_box_pixels"]
            data_boxes.append(
                (left - crop[0], top - crop[1], right - crop[0], bottom - crop[1])
            )
        prepared.append((panel, original_size, crop))

    # A omits the categorical legend; B carries one shared legend at right.
    # Match heatmap widths instead of enlarging A to match B's whole image.
    cnv_width = (
        2
        * cell_width
        / sum(
            prepared[index][0].width / (box[2] - box[0])
            for index, box in enumerate(cnv_boxes)
        )
    )
    for index, box in enumerate(cnv_boxes):
        panel, original_size, crop = prepared[index]
        factor = cnv_width / (box[2] - box[0])
        dimensions = (round(panel.width * factor), round(panel.height * factor))
        prepared[index] = (
            panel.resize(dimensions, Image.Resampling.LANCZOS),
            original_size,
            crop,
        )

    # Allocate space jointly: C has no legend, D carries the shared legend.
    # Scale by the data rectangle, not by the whole PNG/TIFF width.
    data_width = (
        2
        * cell_width
        / sum(
            prepared[index + 2][0].width / (box[2] - box[0])
            for index, box in enumerate(data_boxes)
        )
    )
    scaled_boxes = []
    for index, box in enumerate(data_boxes, 2):
        panel, original_size, crop = prepared[index]
        factor = data_width / (box[2] - box[0])
        dimensions = (round(panel.width * factor), round(panel.height * factor))
        scale_x = dimensions[0] / panel.width
        scale_y = dimensions[1] / panel.height
        scaled_boxes.append(
            (box[0] * scale_x, box[1] * scale_y, box[2] * scale_x, box[3] * scale_y)
        )
        panel = panel.resize(dimensions, Image.Resampling.LANCZOS)
        prepared[index] = (panel, original_size, crop)
    data_top = max(box[1] for box in scaled_boxes)
    bottom_offsets = [round(data_top - box[1]) for box in scaled_boxes]

    top_height = max(prepared[index][0].height for index in (0, 1))
    bottom_height = max(
        prepared[index + 2][0].height + offset
        for index, offset in enumerate(bottom_offsets)
    )
    width = 2 * cell_width + 2 * margin + gap
    height = top_height + bottom_height + 2 * label_height + 2 * margin + gap
    canvas = Image.new("RGB", (width, height), "white")
    drawing = ImageDraw.Draw(canvas)
    letter_font = panel_raster_font(140)
    # Use one column anchor per letter column, independent of panel crops.
    label_columns = (margin, margin + prepared[0][0].width + gap)
    transforms = []
    for index, ((letter, _, _), path) in enumerate(zip(PANEL_FILES, sources)):
        panel, original_size, crop = prepared[index]
        column = index % 2
        row = index // 2
        x = margin + column * (cell_width + gap)
        if index == 1:
            x = margin + prepared[0][0].width + gap
        if index == 3:
            x = margin + prepared[2][0].width + gap
        row_start = margin if row == 0 else margin + label_height + top_height + gap
        label_at = (label_columns[column], row_start)
        vertical_offset = (
            top_height - panel.height if row == 0 else bottom_offsets[column]
        )
        paste_at = (x, row_start + label_height + vertical_offset)
        canvas.paste(panel, paste_at)
        drawing.text(label_at, letter, font=letter_font, fill="black")
        transforms.append(
            {
                "panel": letter,
                "source": str(path),
                "sha256": source_hashes[index],
                "original_pixels": original_size,
                "crop_box": crop,
                "placed_pixels": panel.size,
                "position": paste_at,
                "label_position": label_at,
            }
        )
        if row == 1:
            box = scaled_boxes[column]
            transforms[-1]["data_box_pixels"] = [
                x + box[0],
                paste_at[1] + box[1],
                x + box[2],
                paste_at[1] + box[3],
            ]
    c_box, d_box = (item["data_box_pixels"] for item in transforms[2:])
    differences = [
        abs(c_box[1] - d_box[1]),
        abs(c_box[3] - d_box[3]),
        abs((c_box[2] - c_box[0]) - (d_box[2] - d_box[0])),
    ]
    if max(differences) > 2:
        raise ValueError(
            "C/D data rectangles must share width and top/bottom within 2 pixels"
        )
    if source_hashes != [file_hash(path) for path in sources]:
        raise ValueError("A panel source changed during assembly")

    output.mkdir(parents=True)
    full_path = output / "Supplementary_Figure_06.tiff"
    canvas.save(full_path, dpi=(600, 600), compression="tiff_lzw")
    canvas.thumbnail((2000, 2000), Image.Resampling.LANCZOS)
    canvas.save(output / "Supplementary_Figure_06.png")
    write_json(
        output / "assembly.json",
        {
            "layout": "manuscript A/B top, C/D bottom",
            "source_reference": "outputs/Supplementary_Figure_06/reference/S6_original_preview.png",
            "panels": transforms,
            "upstream_manifests": {str(path): file_hash(path) for path in manifests},
            "script_sha256": file_hash(Path(__file__)),
            "font_sha256": file_hash(panel_font_path()),
            "Pillow_version": pillow_version,
            "analysis_recomputed": False,
            "visual_review": "required after rendering; not automated approval",
            "artifacts": {full_path.name: file_hash(full_path)},
        },
    )


def main() -> None:
    """Read explicit S6 panel locations and write a new numbered composite."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cnv", type=Path, required=True)
    parser.add_argument("--pca", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assemble_supplementary_06(args.cnv, args.pca, args.output)


if __name__ == "__main__":
    main()
