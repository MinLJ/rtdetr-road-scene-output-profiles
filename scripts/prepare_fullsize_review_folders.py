from __future__ import annotations

import csv
import shutil
from collections import defaultdict
from pathlib import Path

# ============================================================
# Input / output paths
# ============================================================
MANIFEST = Path(
    r"D:\MFE204_RoadDetection\subsets\bdd100k_road200_v1\manifests"
    r"\bdd100k_candidate_pool_280.csv"
)

OUTPUT_ROOT = Path(
    r"D:\MFE204_RoadDetection\subsets\bdd100k_road200_v1"
    r"\fullsize_review"
)

IMAGE_OUTPUT_ROOT = OUTPUT_ROOT / "images"
CSV_OUTPUT_ROOT = OUTPUT_ROOT / "review_csv"

IMAGE_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
CSV_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

SCENARIOS = [
    "daytime_normal",
    "night_lowlight",
    "crowded_occluded",
    "small_distant",
]


def read_manifest() -> list[dict]:
    if not MANIFEST.exists():
        raise FileNotFoundError(f"Manifest not found:\n{MANIFEST}")

    with open(MANIFEST, "r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def clear_folder(folder: Path) -> None:
    """Remove old generated files but keep the folder itself."""
    if folder.exists():
        for item in folder.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
    else:
        folder.mkdir(parents=True, exist_ok=True)


def copy_candidate_images(scenario: str, rows: list[dict]) -> list[dict]:
    scenario_dir = IMAGE_OUTPUT_ROOT / scenario
    clear_folder(scenario_dir)

    enriched_rows = []

    for index, row in enumerate(rows, start=1):
        source_path = Path(row["image_path"])

        if not source_path.exists():
            raise FileNotFoundError(
                f"Image missing for {row['image_id']}:\n{source_path}"
            )

        image_id = row["image_id"]
        dynamic_count = row["dynamic_box_count"]
        occ_ratio = row["dynamic_occlusion_ratio"]
        small_count = row["small_dynamic_count"]

        # Keep filename readable while ensuring review order is stable.
        new_filename = (
            f"{index:03d}__{image_id}"
            f"__dyn{dynamic_count}"
            f"__occ{occ_ratio}"
            f"__small{small_count}.jpg"
        )

        destination_path = scenario_dir / new_filename

        # copy2 preserves timestamps and avoids modifying original BDD files.
        shutil.copy2(source_path, destination_path)

        new_row = dict(row)
        new_row["review_index"] = index
        new_row["review_image_filename"] = new_filename
        new_row["review_image_path"] = str(destination_path)
        new_row["visual_review_decision"] = ""
        new_row["visual_review_reason"] = ""
        new_row["reviewer_initials"] = ""
        new_row["review_notes"] = ""

        enriched_rows.append(new_row)

    return enriched_rows


def write_review_csv(scenario: str, rows: list[dict]) -> Path:
    output_csv = CSV_OUTPUT_ROOT / f"review_{scenario}_70_fullsize.csv"

    fields = [
        "review_index",
        "review_image_filename",
        "image_id",
        "review_image_path",
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
        "visual_review_decision",
        "visual_review_reason",
        "reviewer_initials",
        "review_notes",
    ]

    with open(output_csv, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})

    return output_csv


def write_instructions() -> None:
    instruction_file = OUTPUT_ROOT / "README_REVIEW_INSTRUCTIONS.txt"

    content = """BDD100K Road200 Full-Size Visual Review

IMPORTANT:
- Do NOT make final decisions based on the contact sheets.
- Open the individual original-resolution JPG files in the scenario folder.
- Start with 001, then use the left/right arrow keys in Windows Photos
  or another image viewer to review sequentially.

Folder structure:
images/
  daytime_normal/
  night_lowlight/
  crowded_occluded/
  small_distant/

CSV review files:
review_csv/
  review_daytime_normal_70_fullsize.csv
  review_night_lowlight_70_fullsize.csv
  review_crowded_occluded_70_fullsize.csv
  review_small_distant_70_fullsize.csv

For each image:
1. Find the corresponding review_index in the CSV.
2. Fill:
   visual_review_decision = keep OR exclude
   visual_review_reason = one standard reason if excluded
   reviewer_initials = your initials
   review_notes = optional short note

Use only these exclusion reasons:
- not_visually_representative
- too_similar_to_normal_scene
- weak_small_object_challenge
- weak_crowding_or_occlusion_evidence
- low_visual_quality
- extreme_reflection_or_blur
- duplicate_visual_pattern
- other

Target:
- Retain exactly 50 images per scenario.
- Exclude exactly 20 images per scenario.
"""

    instruction_file.write_text(content, encoding="utf-8")


def main() -> None:
    rows = read_manifest()

    grouped: dict[str, list[dict]] = defaultdict(list)

    for row in rows:
        grouped[row["scenario_candidate"]].append(row)

    for scenario in SCENARIOS:
        scenario_rows = grouped.get(scenario, [])

        if len(scenario_rows) != 70:
            raise RuntimeError(
                f"{scenario}: expected 70 rows, found {len(scenario_rows)}."
            )

        # Keep same order as existing candidate manifest.
        reviewed_rows = copy_candidate_images(scenario, scenario_rows)
        review_csv = write_review_csv(scenario, reviewed_rows)

        print(f"{scenario}:")
        print(f"  Images: {IMAGE_OUTPUT_ROOT / scenario}")
        print(f"  CSV:    {review_csv}")

    write_instructions()

    print("\nDone.")
    print(f"Full-size review package: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()