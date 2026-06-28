from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

# ============================================================
# Paths
# ============================================================
MANIFEST = Path(
    r"D:\MFE204_RoadDetection\subsets\bdd100k_road200_v1\manifests"
    r"\bdd100k_candidate_pool_280.csv"
)

OUTPUT_ROOT = Path(
    r"D:\MFE204_RoadDetection\subsets\bdd100k_road200_v1\review_assets"
)

CONTACT_SHEET_DIR = OUTPUT_ROOT / "contact_sheets"
REVIEW_CSV_DIR = OUTPUT_ROOT / "review_csv"

CONTACT_SHEET_DIR.mkdir(parents=True, exist_ok=True)
REVIEW_CSV_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Contact-sheet layout
# ============================================================
COLUMNS = 4
ROWS_PER_PAGE = 5
IMAGES_PER_PAGE = COLUMNS * ROWS_PER_PAGE

THUMB_WIDTH = 300
THUMB_HEIGHT = 170
CAPTION_HEIGHT = 82

OUTER_MARGIN = 24
GAP = 14

JPEG_QUALITY = 92


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Try common Windows fonts; fall back safely if unavailable."""
    candidates = []

    if bold:
        candidates.extend(
            [
                r"C:\Windows\Fonts\arialbd.ttf",
                r"C:\Windows\Fonts\segoeuib.ttf",
                r"C:\Windows\Fonts\msyhbd.ttc",
            ]
        )
    else:
        candidates.extend(
            [
                r"C:\Windows\Fonts\arial.ttf",
                r"C:\Windows\Fonts\segoeui.ttf",
                r"C:\Windows\Fonts\msyh.ttc",
            ]
        )

    for font_path in candidates:
        path = Path(font_path)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)

    return ImageFont.load_default()


TITLE_FONT = load_font(23, bold=True)
ID_FONT = load_font(17, bold=True)
INFO_FONT = load_font(14, bold=False)
SMALL_FONT = load_font(13, bold=False)


def read_manifest() -> list[dict]:
    if not MANIFEST.exists():
        raise FileNotFoundError(
            f"Candidate manifest was not found:\n{MANIFEST}"
        )

    with open(MANIFEST, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    if not rows:
        raise RuntimeError("The candidate manifest is empty.")

    return rows


def draw_text_block(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    row: dict,
) -> None:
    """Draw a compact metadata panel under each thumbnail."""
    image_id = row["image_id"]
    scenario = row["scenario_candidate"]
    timeofday = row["timeofday"]
    weather = row["weather"]
    scene = row["scene"]

    dynamic_count = row["dynamic_box_count"]
    occlusion_ratio = row["dynamic_occlusion_ratio"]
    small_count = row["small_dynamic_count"]

    draw.text(
        (x, y + 5),
        image_id,
        font=ID_FONT,
        fill="black",
    )

    draw.text(
        (x, y + 28),
        f"{scenario} | {timeofday}",
        font=INFO_FONT,
        fill="black",
    )

    draw.text(
        (x, y + 47),
        f"{scene} | {weather}",
        font=SMALL_FONT,
        fill="black",
    )

    draw.text(
        (x, y + 64),
        (
            f"dyn={dynamic_count} | occ={occlusion_ratio} | "
            f"small={small_count}"
        ),
        font=SMALL_FONT,
        fill="black",
    )


def render_page(
    scenario: str,
    page_rows: list[dict],
    page_index: int,
    total_pages: int,
) -> Path:
    page_width = (
        OUTER_MARGIN * 2
        + COLUMNS * THUMB_WIDTH
        + (COLUMNS - 1) * GAP
    )

    title_height = 58

    page_height = (
        title_height
        + OUTER_MARGIN
        + ROWS_PER_PAGE * (THUMB_HEIGHT + CAPTION_HEIGHT)
        + (ROWS_PER_PAGE - 1) * GAP
        + OUTER_MARGIN
    )

    canvas = Image.new("RGB", (page_width, page_height), "white")
    draw = ImageDraw.Draw(canvas)

    title = (
        f"BDD100K candidate review | {scenario} | "
        f"page {page_index}/{total_pages}"
    )

    draw.text(
        (OUTER_MARGIN, 16),
        title,
        font=TITLE_FONT,
        fill="black",
    )

    for local_index, row in enumerate(page_rows):
        col = local_index % COLUMNS
        row_index = local_index // COLUMNS

        x = OUTER_MARGIN + col * (THUMB_WIDTH + GAP)
        y = (
            title_height
            + OUTER_MARGIN
            + row_index * (THUMB_HEIGHT + CAPTION_HEIGHT + GAP)
        )

        image_path = Path(row["image_path"])

        try:
            with Image.open(image_path) as source:
                source = source.convert("RGB")

                thumbnail = ImageOps.contain(
                    source,
                    (THUMB_WIDTH, THUMB_HEIGHT),
                )

                image_panel = Image.new(
                    "RGB",
                    (THUMB_WIDTH, THUMB_HEIGHT),
                    "lightgray",
                )

                paste_x = (THUMB_WIDTH - thumbnail.width) // 2
                paste_y = (THUMB_HEIGHT - thumbnail.height) // 2

                image_panel.paste(thumbnail, (paste_x, paste_y))

        except Exception as exc:
            image_panel = Image.new(
                "RGB",
                (THUMB_WIDTH, THUMB_HEIGHT),
                "lightgray",
            )

            error_draw = ImageDraw.Draw(image_panel)
            error_draw.text(
                (10, 10),
                "IMAGE READ ERROR",
                font=ID_FONT,
                fill="black",
            )
            error_draw.text(
                (10, 38),
                str(exc)[:80],
                font=SMALL_FONT,
                fill="black",
            )

        canvas.paste(image_panel, (x, y))

        draw.rectangle(
            (
                x,
                y,
                x + THUMB_WIDTH,
                y + THUMB_HEIGHT,
            ),
            outline="black",
            width=1,
        )

        draw_text_block(
            draw=draw,
            x=x,
            y=y + THUMB_HEIGHT,
            row=row,
        )

    filename = (
        f"{scenario}_page_{page_index:02d}_of_{total_pages:02d}.jpg"
    )

    output_path = CONTACT_SHEET_DIR / filename

    canvas.save(output_path, quality=JPEG_QUALITY)

    return output_path


def write_review_csv(scenario: str, rows: list[dict]) -> Path:
    """
    Write one scenario-specific review sheet.
    Do not overwrite the master manifest.
    """
    output_path = REVIEW_CSV_DIR / f"review_{scenario}_70.csv"

    fieldnames = [
        "image_id",
        "image_path",
        "scenario_candidate",
        "timeofday",
        "weather",
        "scene",
        "mapped_box_count",
        "dynamic_box_count",
        "occluded_dynamic_count",
        "dynamic_occlusion_ratio",
        "small_dynamic_count",
        "mapped_classes_present",
        "candidate_status",
        "visual_review_decision",
        "visual_review_reason",
        "reviewer_initials",
        "review_notes",
    ]

    with open(
        output_path,
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    "image_id": row["image_id"],
                    "image_path": row["image_path"],
                    "scenario_candidate": row["scenario_candidate"],
                    "timeofday": row["timeofday"],
                    "weather": row["weather"],
                    "scene": row["scene"],
                    "mapped_box_count": row["mapped_box_count"],
                    "dynamic_box_count": row["dynamic_box_count"],
                    "occluded_dynamic_count": row[
                        "occluded_dynamic_count"
                    ],
                    "dynamic_occlusion_ratio": row[
                        "dynamic_occlusion_ratio"
                    ],
                    "small_dynamic_count": row[
                        "small_dynamic_count"
                    ],
                    "mapped_classes_present": row[
                        "mapped_classes_present"
                    ],
                    "candidate_status": "pending_visual_review",
                    "visual_review_decision": "",
                    "visual_review_reason": "",
                    "reviewer_initials": "",
                    "review_notes": "",
                }
            )

    return output_path


def main() -> None:
    rows = read_manifest()

    grouped: dict[str, list[dict]] = defaultdict(list)

    for row in rows:
        grouped[row["scenario_candidate"]].append(row)

    expected_scenarios = [
        "daytime_normal",
        "night_lowlight",
        "crowded_occluded",
        "small_distant",
    ]

    print("Generating BDD100K review assets...\n")

    for scenario in expected_scenarios:
        scenario_rows = grouped.get(scenario, [])

        if len(scenario_rows) != 70:
            raise RuntimeError(
                f"{scenario}: expected 70 candidates, "
                f"but found {len(scenario_rows)}."
            )

        review_csv = write_review_csv(scenario, scenario_rows)

        total_pages = math.ceil(
            len(scenario_rows) / IMAGES_PER_PAGE
        )

        print(
            f"{scenario}: {len(scenario_rows)} images, "
            f"{total_pages} contact-sheet pages"
        )

        for page_number in range(1, total_pages + 1):
            start = (page_number - 1) * IMAGES_PER_PAGE
            end = start + IMAGES_PER_PAGE

            page_rows = scenario_rows[start:end]

            output_path = render_page(
                scenario=scenario,
                page_rows=page_rows,
                page_index=page_number,
                total_pages=total_pages,
            )

            print(f"  Saved: {output_path.name}")

        print(f"  Review CSV: {review_csv.name}\n")

    print("Done.")
    print(f"Contact sheets: {CONTACT_SHEET_DIR}")
    print(f"Review CSV files: {REVIEW_CSV_DIR}")


if __name__ == "__main__":
    main()