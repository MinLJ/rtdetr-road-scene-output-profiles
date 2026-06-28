from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

# ============================================================
# Fixed project paths
# ============================================================
IMAGE_ROOT = Path(
    r"D:\MFE204_RoadDetection\data\bdd100k_images\bdd100k\images\100k"
)
LABEL_ROOT = Path(
    r"D:\MFE204_RoadDetection\data\bdd100k_labels\bdd100k\labels\100k"
)

SPLIT = "val"

OUTPUT_DIR = Path(r"D:\MFE204_RoadDetection\docs\bdd100k_audit")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "bdd100k_val_audit.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "bdd100k_val_audit_summary.txt"

# BDD100K image resolution is normally 1280 x 720.
# The script will report this assumption clearly in the summary.
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FRAME_AREA = FRAME_WIDTH * FRAME_HEIGHT

# These classes can be mapped reasonably to common COCO-pretrained
# detector categories used by YOLOv8n / RT-DETR-L.
# We exclude BDD100K "rider" and generic "traffic sign" for now because
# they do not have a clean one-to-one COCO label mapping.
EVAL_CLASSES = {
    "pedestrian",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
    "traffic light",
}

# Small object definition for the first audit:
# bbox area < 0.5% of a 1280 x 720 image.
SMALL_OBJECT_AREA_RATIO = 0.005
SMALL_OBJECT_MAX_AREA = FRAME_AREA * SMALL_OBJECT_AREA_RATIO


def to_bool(value) -> bool:
    """Convert BDD-style boolean-like values safely."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return False


def get_objects(record: dict) -> list[dict]:
    """
    Supports common BDD100K per-image JSON layouts:
    record['frames'][0]['objects'] or record['frames'][0]['labels'].
    """
    frames = record.get("frames", [])
    if not frames:
        return []

    first_frame = frames[0]

    if isinstance(first_frame.get("objects"), list):
        return first_frame["objects"]

    if isinstance(first_frame.get("labels"), list):
        return first_frame["labels"]

    return []


def get_box_area(obj: dict) -> float | None:
    """Return valid bbox area from BDD box2d format, otherwise None."""
    box = obj.get("box2d", {})

    try:
        x1 = float(box["x1"])
        y1 = float(box["y1"])
        x2 = float(box["x2"])
        y2 = float(box["y2"])
    except (KeyError, TypeError, ValueError):
        return None

    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    area = width * height

    return area if area > 0 else None


def main() -> None:
    image_dir = IMAGE_ROOT / SPLIT
    label_dir = LABEL_ROOT / SPLIT

    if not image_dir.exists():
        raise FileNotFoundError(f"Image folder not found: {image_dir}")

    if not label_dir.exists():
        raise FileNotFoundError(f"Label folder not found: {label_dir}")

    label_files = sorted(label_dir.glob("*.json"))

    if not label_files:
        raise RuntimeError(f"No JSON labels found in: {label_dir}")

    print(f"Scanning {len(label_files):,} BDD100K {SPLIT} labels...")
    print(f"CSV output: {OUTPUT_CSV}")

    timeofday_counter = Counter()
    weather_counter = Counter()
    scene_counter = Counter()
    class_counter = Counter()

    total_images = 0
    matched_images = 0
    unreadable_json = 0

    total_eval_boxes = 0
    total_small_boxes = 0
    total_occluded_boxes = 0
    total_truncated_boxes = 0

    rows = []

    for index, json_path in enumerate(label_files, start=1):
        try:
            with open(json_path, "r", encoding="utf-8") as file:
                record = json.load(file)
        except Exception as exc:
            unreadable_json += 1
            print(f"[WARN] Cannot read {json_path.name}: {exc}")
            continue

        image_id = record.get("name", json_path.stem)
        image_path = image_dir / f"{image_id}.jpg"
        image_exists = image_path.exists()

        total_images += 1
        matched_images += int(image_exists)

        attributes = record.get("attributes", {})
        timeofday = str(attributes.get("timeofday", "unknown")).strip().lower()
        weather = str(attributes.get("weather", "unknown")).strip().lower()
        scene = str(attributes.get("scene", "unknown")).strip().lower()

        timeofday_counter[timeofday] += 1
        weather_counter[weather] += 1
        scene_counter[scene] += 1

        objects = get_objects(record)

        eval_box_count = 0
        small_box_count = 0
        occluded_box_count = 0
        truncated_box_count = 0
        image_class_counter = Counter()

        for obj in objects:
            category = str(obj.get("category", "")).strip().lower()

            if category not in EVAL_CLASSES:
                continue

            area = get_box_area(obj)

            # Ignore objects without a valid 2D box.
            if area is None:
                continue

            eval_box_count += 1
            total_eval_boxes += 1

            image_class_counter[category] += 1
            class_counter[category] += 1

            obj_attributes = obj.get("attributes", {})

            is_occluded = to_bool(obj_attributes.get("occluded", False))
            is_truncated = to_bool(obj_attributes.get("truncated", False))

            if is_occluded:
                occluded_box_count += 1
                total_occluded_boxes += 1

            if is_truncated:
                truncated_box_count += 1
                total_truncated_boxes += 1

            if area < SMALL_OBJECT_MAX_AREA:
                small_box_count += 1
                total_small_boxes += 1

        occlusion_ratio = (
            occluded_box_count / eval_box_count
            if eval_box_count > 0
            else 0.0
        )

        # These are only preliminary candidate flags.
        # We will decide final thresholds after reading the statistics.
        candidate_daytime_normal = (
            timeofday == "daytime"
            and 3 <= eval_box_count <= 8
            and occlusion_ratio <= 0.20
        )

        candidate_night_lowlight = (
            timeofday == "night"
            and eval_box_count >= 2
        )

        candidate_crowded_occluded = (
            eval_box_count >= 10
            or occluded_box_count >= 3
        )

        candidate_small_distant = (
            small_box_count >= 2
        )

        rows.append(
            {
                "image_id": image_id,
                "split": SPLIT,
                "image_path": str(image_path),
                "image_exists": image_exists,
                "timeofday": timeofday,
                "weather": weather,
                "scene": scene,
                "eval_box_count": eval_box_count,
                "small_box_count": small_box_count,
                "occluded_box_count": occluded_box_count,
                "truncated_box_count": truncated_box_count,
                "occlusion_ratio": round(occlusion_ratio, 4),
                "classes_present": ";".join(sorted(image_class_counter)),
                "candidate_daytime_normal": candidate_daytime_normal,
                "candidate_night_lowlight": candidate_night_lowlight,
                "candidate_crowded_occluded": candidate_crowded_occluded,
                "candidate_small_distant": candidate_small_distant,
            }
        )

        if index % 1000 == 0:
            print(f"Processed {index:,}/{len(label_files):,} labels...")

    fieldnames = list(rows[0].keys())

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    candidate_counts = {
        "daytime_normal": sum(row["candidate_daytime_normal"] for row in rows),
        "night_lowlight": sum(row["candidate_night_lowlight"] for row in rows),
        "crowded_occluded": sum(row["candidate_crowded_occluded"] for row in rows),
        "small_distant": sum(row["candidate_small_distant"] for row in rows),
    }

    with open(OUTPUT_SUMMARY, "w", encoding="utf-8") as file:
        file.write("BDD100K Validation Split Audit Summary\n")
        file.write("=" * 60 + "\n\n")

        file.write(f"Image root: {image_dir}\n")
        file.write(f"Label root: {label_dir}\n")
        file.write(f"Expected frame size: {FRAME_WIDTH} x {FRAME_HEIGHT}\n")
        file.write(
            f"Small-object threshold: < {SMALL_OBJECT_MAX_AREA:.1f} pixels "
            f"({SMALL_OBJECT_AREA_RATIO * 100:.2f}% of image area)\n\n"
        )

        file.write(f"Readable label records: {total_images:,}\n")
        file.write(f"Matching image files: {matched_images:,}\n")
        file.write(f"Unreadable JSON files: {unreadable_json:,}\n\n")

        file.write("Time of day distribution:\n")
        for key, value in timeofday_counter.most_common():
            file.write(f"  {key}: {value:,}\n")

        file.write("\nWeather distribution:\n")
        for key, value in weather_counter.most_common():
            file.write(f"  {key}: {value:,}\n")

        file.write("\nScene distribution:\n")
        for key, value in scene_counter.most_common():
            file.write(f"  {key}: {value:,}\n")

        file.write("\nEvaluation-class box distribution:\n")
        for key, value in class_counter.most_common():
            file.write(f"  {key}: {value:,}\n")

        file.write("\nAggregate annotation counts:\n")
        file.write(f"  Evaluation-class boxes: {total_eval_boxes:,}\n")
        file.write(f"  Small boxes: {total_small_boxes:,}\n")
        file.write(f"  Occluded boxes: {total_occluded_boxes:,}\n")
        file.write(f"  Truncated boxes: {total_truncated_boxes:,}\n")

        file.write("\nPreliminary scenario candidate counts:\n")
        for key, value in candidate_counts.items():
            file.write(f"  {key}: {value:,}\n")

    print("\nAudit complete.")
    print(f"Matched image files: {matched_images:,}/{total_images:,}")
    print("Preliminary candidate counts:")
    for key, value in candidate_counts.items():
        print(f"  {key}: {value:,}")

    print(f"\nSaved CSV: {OUTPUT_CSV}")
    print(f"Saved summary: {OUTPUT_SUMMARY}")


if __name__ == "__main__":
    main()