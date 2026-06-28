from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

# ============================================================
# Paths
# ============================================================
IMAGE_ROOT = Path(
    r"D:\MFE204_RoadDetection\data\bdd100k_images\bdd100k\images\100k"
)

LABEL_ROOT = Path(
    r"D:\MFE204_RoadDetection\data\bdd100k_labels\bdd100k\labels\100k"
)

SPLIT = "val"

OUTPUT_DIR = Path(r"D:\MFE204_RoadDetection\docs\bdd100k_audit_v2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "bdd100k_val_audit_v2.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "bdd100k_val_audit_v2_summary.txt"

# ============================================================
# Image geometry
# ============================================================
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FRAME_AREA = FRAME_WIDTH * FRAME_HEIGHT

# ============================================================
# BDD100K -> COCO-compatible mapping
# Only keep categories with a clean semantic match.
# ============================================================
BDD_TO_EVAL = {
    "person": "person",
    "bike": "bicycle",
    "car": "car",
    "motor": "motorcycle",
    "bus": "bus",
    "train": "train",
    "truck": "truck",
    "traffic light": "traffic light",
}

# Use dynamic road agents / vehicles to define density and small-object
# scenarios. Traffic lights are valid for mAP, but should not cause
# an image to be labelled "small/distant" by themselves.
DYNAMIC_BDD_CLASSES = {
    "person",
    "bike",
    "car",
    "motor",
    "bus",
    "train",
    "truck",
}

# Small dynamic object = bbox under 0.25% of image area.
# 1280 * 720 * 0.0025 = 2304 pixels.
SMALL_AREA_RATIO = 0.0025
SMALL_AREA_THRESHOLD = FRAME_AREA * SMALL_AREA_RATIO


def to_bool(value) -> bool:
    """Convert BDD-style boolean-like fields safely."""
    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return value != 0

    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}

    return False


def get_objects(record: dict) -> list[dict]:
    """Read object list from the common BDD100K per-image JSON layout."""
    frames = record.get("frames", [])

    if not frames:
        return []

    frame = frames[0]

    if isinstance(frame.get("objects"), list):
        return frame["objects"]

    if isinstance(frame.get("labels"), list):
        return frame["labels"]

    return []


def get_box_area(obj: dict) -> float | None:
    """Return bbox area for a valid box2d annotation."""
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


def percentile(sorted_values: list[float], p: float) -> float:
    """Simple percentile without third-party dependencies."""
    if not sorted_values:
        return 0.0

    index = round((len(sorted_values) - 1) * p)
    return sorted_values[index]


def main() -> None:
    image_dir = IMAGE_ROOT / SPLIT
    label_dir = LABEL_ROOT / SPLIT

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    if not label_dir.exists():
        raise FileNotFoundError(f"Label directory not found: {label_dir}")

    label_files = sorted(label_dir.glob("*.json"))

    if not label_files:
        raise RuntimeError(f"No JSON files found in: {label_dir}")

    print(f"Scanning {len(label_files):,} BDD100K {SPLIT} labels...")

    raw_class_counter = Counter()
    mapped_class_counter = Counter()
    timeofday_counter = Counter()
    weather_counter = Counter()
    scene_counter = Counter()

    rows = []

    total_records = 0
    matched_images = 0
    unreadable_json = 0

    dynamic_counts = []
    eval_counts = []
    occlusion_ratios = []
    small_dynamic_counts = []

    for index, json_path in enumerate(label_files, start=1):
        try:
            with open(json_path, "r", encoding="utf-8") as file:
                record = json.load(file)
        except Exception as exc:
            unreadable_json += 1
            print(f"[WARN] Cannot read {json_path.name}: {exc}")
            continue

        total_records += 1

        image_id = record.get("name", json_path.stem)
        image_path = image_dir / f"{image_id}.jpg"
        image_exists = image_path.exists()
        matched_images += int(image_exists)

        image_attributes = record.get("attributes", {})
        timeofday = str(
            image_attributes.get("timeofday", "undefined")
        ).strip().lower()

        weather = str(
            image_attributes.get("weather", "undefined")
        ).strip().lower()

        scene = str(
            image_attributes.get("scene", "undefined")
        ).strip().lower()

        timeofday_counter[timeofday] += 1
        weather_counter[weather] += 1
        scene_counter[scene] += 1

        mapped_box_count = 0
        dynamic_box_count = 0
        occluded_mapped_count = 0
        occluded_dynamic_count = 0
        truncated_mapped_count = 0
        small_dynamic_count = 0

        mapped_classes_present = Counter()
        raw_classes_present = Counter()

        for obj in get_objects(record):
            raw_category = str(
                obj.get("category", "")
            ).strip().lower()

            if not raw_category:
                continue

            raw_class_counter[raw_category] += 1
            raw_classes_present[raw_category] += 1

            if raw_category not in BDD_TO_EVAL:
                continue

            area = get_box_area(obj)

            if area is None:
                continue

            eval_category = BDD_TO_EVAL[raw_category]

            mapped_box_count += 1
            mapped_class_counter[eval_category] += 1
            mapped_classes_present[eval_category] += 1

            obj_attributes = obj.get("attributes", {})
            is_occluded = to_bool(obj_attributes.get("occluded", False))
            is_truncated = to_bool(obj_attributes.get("truncated", False))

            if is_occluded:
                occluded_mapped_count += 1

            if is_truncated:
                truncated_mapped_count += 1

            if raw_category in DYNAMIC_BDD_CLASSES:
                dynamic_box_count += 1

                if is_occluded:
                    occluded_dynamic_count += 1

                if area < SMALL_AREA_THRESHOLD:
                    small_dynamic_count += 1

        dynamic_occlusion_ratio = (
            occluded_dynamic_count / dynamic_box_count
            if dynamic_box_count > 0
            else 0.0
        )

        # These are intentionally preliminary and stricter than v1.
        # They are used only to inspect candidate availability.
        candidate_daytime_normal = (
            timeofday == "daytime"
            and 3 <= dynamic_box_count <= 8
            and dynamic_occlusion_ratio <= 0.20
        )

        candidate_night_lowlight = (
            timeofday == "night"
            and dynamic_box_count >= 3
        )

        candidate_crowded_occluded = (
            dynamic_box_count >= 12
            and (
                occluded_dynamic_count >= 3
                or dynamic_occlusion_ratio >= 0.30
            )
        )

        candidate_small_distant = (
            small_dynamic_count >= 3
            and dynamic_box_count >= 4
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
                "mapped_box_count": mapped_box_count,
                "dynamic_box_count": dynamic_box_count,
                "occluded_mapped_count": occluded_mapped_count,
                "occluded_dynamic_count": occluded_dynamic_count,
                "dynamic_occlusion_ratio": round(dynamic_occlusion_ratio, 4),
                "truncated_mapped_count": truncated_mapped_count,
                "small_dynamic_count": small_dynamic_count,
                "mapped_classes_present": ";".join(
                    sorted(mapped_classes_present)
                ),
                "raw_classes_present": ";".join(
                    sorted(raw_classes_present)
                ),
                "candidate_daytime_normal": candidate_daytime_normal,
                "candidate_night_lowlight": candidate_night_lowlight,
                "candidate_crowded_occluded": candidate_crowded_occluded,
                "candidate_small_distant": candidate_small_distant,
            }
        )

        dynamic_counts.append(dynamic_box_count)
        eval_counts.append(mapped_box_count)
        occlusion_ratios.append(dynamic_occlusion_ratio)
        small_dynamic_counts.append(small_dynamic_count)

        if index % 1000 == 0:
            print(f"Processed {index:,}/{len(label_files):,} labels...")

    if not rows:
        raise RuntimeError("No valid annotation rows were created.")

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    candidate_counts = {
        "daytime_normal": sum(
            row["candidate_daytime_normal"] for row in rows
        ),
        "night_lowlight": sum(
            row["candidate_night_lowlight"] for row in rows
        ),
        "crowded_occluded": sum(
            row["candidate_crowded_occluded"] for row in rows
        ),
        "small_distant": sum(
            row["candidate_small_distant"] for row in rows
        ),
    }

    dynamic_counts.sort()
    eval_counts.sort()
    occlusion_ratios.sort()
    small_dynamic_counts.sort()

    with open(OUTPUT_SUMMARY, "w", encoding="utf-8") as file:
        file.write("BDD100K Validation Audit v2\n")
        file.write("=" * 70 + "\n\n")

        file.write("Data roots\n")
        file.write(f"  Images: {image_dir}\n")
        file.write(f"  Labels: {label_dir}\n\n")

        file.write("Evaluation mapping\n")
        for raw_name, eval_name in BDD_TO_EVAL.items():
            file.write(f"  {raw_name} -> {eval_name}\n")

        file.write("\nExcluded raw categories\n")
        file.write("  rider: no clean one-to-one COCO mapping\n")
        file.write("  traffic sign: BDD generic sign is not COCO stop sign\n")
        file.write("  lane/* and area/*: polyline/polygon annotations, not bbox objects\n")

        file.write("\nData integrity\n")
        file.write(f"  Readable labels: {total_records:,}\n")
        file.write(f"  Matching images: {matched_images:,}\n")
        file.write(f"  Unreadable JSON: {unreadable_json:,}\n")

        file.write("\nMetadata distributions\n")
        file.write("Time of day:\n")
        for key, value in timeofday_counter.most_common():
            file.write(f"  {key}: {value:,}\n")

        file.write("\nMapped bbox class distribution:\n")
        for key, value in mapped_class_counter.most_common():
            file.write(f"  {key}: {value:,}\n")

        file.write("\nPer-image distribution summary\n")
        file.write(
            "Dynamic box count "
            f"(P25/P50/P75/P90): "
            f"{percentile(dynamic_counts, 0.25):.0f} / "
            f"{percentile(dynamic_counts, 0.50):.0f} / "
            f"{percentile(dynamic_counts, 0.75):.0f} / "
            f"{percentile(dynamic_counts, 0.90):.0f}\n"
        )

        file.write(
            "Mapped box count "
            f"(P25/P50/P75/P90): "
            f"{percentile(eval_counts, 0.25):.0f} / "
            f"{percentile(eval_counts, 0.50):.0f} / "
            f"{percentile(eval_counts, 0.75):.0f} / "
            f"{percentile(eval_counts, 0.90):.0f}\n"
        )

        file.write(
            "Dynamic occlusion ratio "
            f"(P25/P50/P75/P90): "
            f"{percentile(occlusion_ratios, 0.25):.2f} / "
            f"{percentile(occlusion_ratios, 0.50):.2f} / "
            f"{percentile(occlusion_ratios, 0.75):.2f} / "
            f"{percentile(occlusion_ratios, 0.90):.2f}\n"
        )

        file.write(
            "Small dynamic object count "
            f"(P25/P50/P75/P90): "
            f"{percentile(small_dynamic_counts, 0.25):.0f} / "
            f"{percentile(small_dynamic_counts, 0.50):.0f} / "
            f"{percentile(small_dynamic_counts, 0.75):.0f} / "
            f"{percentile(small_dynamic_counts, 0.90):.0f}\n"
        )

        file.write("\nPreliminary candidate counts\n")
        for key, value in candidate_counts.items():
            file.write(f"  {key}: {value:,}\n")

        file.write("\nSmall-object definition\n")
        file.write(
            f"  Dynamic bbox area < {SMALL_AREA_THRESHOLD:.0f} pixels "
            f"({SMALL_AREA_RATIO * 100:.2f}% of a 1280x720 image)\n"
        )

    print("\nAudit v2 complete.")
    print(f"Matched image files: {matched_images:,}/{total_records:,}")
    print("Preliminary candidate counts:")

    for key, value in candidate_counts.items():
        print(f"  {key}: {value:,}")

    print(f"\nSaved CSV: {OUTPUT_CSV}")
    print(f"Saved summary: {OUTPUT_SUMMARY}")


if __name__ == "__main__":
    main()