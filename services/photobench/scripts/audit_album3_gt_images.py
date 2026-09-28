#!/usr/bin/env python3
"""Render an album3 contact sheet and list QA retrieval/evidence image coverage."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
ALBUM_DIR = ROOT / "data" / "album3"
PHOTO_DIR = ALBUM_DIR / "photos"
QA_PATH = ALBUM_DIR / "qa" / "full-album3.jsonl"


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        Path("/System/Library/Fonts/Hiragino Sans GB.ttc"),
        Path("/System/Library/Fonts/STHeiti Medium.ttc"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    )
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def render_contact_sheet(output: Path) -> None:
    photos = sorted(path for path in PHOTO_DIR.iterdir() if path.is_file())
    columns = 5
    cell_width, cell_height = 420, 350
    header_height = 70
    rows = math.ceil(len(photos) / columns)
    canvas = Image.new(
        "RGB", (columns * cell_width, header_height + rows * cell_height), "white"
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 16), "PhotoBench album3 · 全部 53 张照片", fill="black", font=load_font(22))
    label_font = load_font(17)
    for index, path in enumerate(photos):
        row, column = divmod(index, columns)
        x = column * cell_width
        y = header_height + row * cell_height
        image = ImageOps.exif_transpose(Image.open(path).convert("RGB"))
        image.thumbnail((cell_width - 24, cell_height - 68), Image.Resampling.LANCZOS)
        canvas.paste(image, (x + (cell_width - image.width) // 2, y + 8))
        draw.rectangle((x, y + cell_height - 53, x + cell_width, y + cell_height), fill="white")
        draw.text(
            (x + 10, y + cell_height - 45),
            f"{index + 1:02d}  {path.name}",
            fill="black",
            font=label_font,
        )
        draw.rectangle((x, y, x + cell_width - 1, y + cell_height - 1), outline="#999999")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=92)


def write_coverage(output: Path) -> None:
    rows = [json.loads(line) for line in QA_PATH.read_text().splitlines() if line.strip()]
    report = []
    for row in rows[:26]:
        report.append(
            {
                "qa_id": row["qa_id"],
                "question": row["question"],
                "retrieval_image_ids": row.get("retrieval_image_ids", []),
                "answer_evidence_image_ids": row.get("answer_evidence_image_ids", []),
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


def validate_coverage() -> None:
    rows = [json.loads(line) for line in QA_PATH.read_text().splitlines() if line.strip()]
    for row in rows:
        retrieval = set(row.get("retrieval_image_ids", []))
        direct = set(row.get("answer_evidence_image_ids", []))
        for claim in row.get("answer_claims", []):
            direct.update(claim.get("evidence_image_ids", []))
        if not direct.issubset(retrieval):
            missing = sorted(direct - retrieval)
            raise ValueError(f"{row['qa_id']} direct evidence missing from retrieval GT: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "album3-gt-audit")
    args = parser.parse_args()
    validate_coverage()
    render_contact_sheet(args.output_dir / "all-53-contact-sheet.jpg")
    write_coverage(args.output_dir / "qa-image-coverage.json")
    print(args.output_dir)


if __name__ == "__main__":
    main()
