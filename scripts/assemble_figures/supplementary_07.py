"""Assemble S7 in manuscript order: A/B, C/D, then E/F/G."""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

from asc_reproduction.figure_style import panel_raster_font
from asc_reproduction.provenance import file_hash


def assemble(panels: Path, output: Path) -> None:
    """Compose existing small TIFFs without recalculating or adding tests."""
    if output.exists():
        raise FileExistsError("Use a new candidate directory")
    names = [
        "S7A_Jaccard",
        "S7B_SMC",
        "S7C_cluster_cosine",
        "S7D_arm_cosine",
        "S7E",
        "S7F",
        "S7G",
    ]
    rows = []
    provenance = []
    width, margin, gap, label_height = 6600, 100, 100, 160
    for indices in ((0, 1), (2, 3), (4, 5, 6)):
        row = []
        target = (width - 2 * margin - (len(indices) - 1) * gap) // len(indices)
        for index in indices:
            path = panels / f"{names[index]}.tiff"
            with Image.open(path) as source:
                image = source.convert("RGB")
            assert image.width >= target
            image = image.resize(
                (target, round(image.height * target / image.width)),
                Image.Resampling.LANCZOS,
            )
            row.append((index, image))
            provenance.append(
                {"panel": names[index], "source": str(path), "sha256": file_hash(path)}
            )
        rows.append(row)
    height = (
        2 * margin
        + sum(max(im.height for _, im in row) + label_height for row in rows)
        + 2 * gap
    )
    canvas = Image.new("RGB", (width, height), "white")
    drawing = ImageDraw.Draw(canvas)
    y = margin
    for row in rows:
        x = margin
        for index, image in row:
            drawing.text(
                (x, y), "ABCDEFG"[index], font=panel_raster_font(140), fill="black"
            )
            canvas.paste(image, (x, y + label_height))
            x += image.width + gap
        y += max(im.height for _, im in row) + label_height + gap
    output.mkdir(parents=True)
    canvas.save(
        output / "Supplementary_Figure_07.tiff", dpi=(600, 600), compression="tiff_lzw"
    )
    canvas.thumbnail((2000, 2400), Image.Resampling.LANCZOS)
    canvas.save(output / "Supplementary_Figure_07.png")
    (output / "assembly.json").write_text(
        json.dumps(
            {
                "panels": provenance,
            "status": "candidate; see upstream summary for E–G test specification",
                "script_sha256": file_hash(Path(__file__)),
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    """Read source and destination directories."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assemble(args.panels, args.output)


if __name__ == "__main__":
    main()
