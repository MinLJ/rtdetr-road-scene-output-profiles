from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

LABEL_DIR = Path(
    r"D:\MFE204_RoadDetection\data\bdd100k_labels\bdd100k\labels\100k\val"
)

OUTPUT_FILE = Path(
    r"D:\MFE204_RoadDetection\docs\bdd100k_audit\bdd100k_raw_category_summary.txt"
)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)


def get_objects(record: dict) -> list[dict]:
    """Support common BDD100K per-image JSON structures."""
    frames = record.get("frames", [])
    if not frames:
        return []

    frame = frames[0]

    if isinstance(frame.get("objects"), list):
        return frame["objects"]

    if isinstance(frame.get("labels"), list):
        return frame["labels"]

    return []


def main() -> None:
    label_files = sorted(LABEL_DIR.glob("*.json"))

    if not label_files:
        raise RuntimeError(f"No JSON files found in: {LABEL_DIR}")

    category_counter = Counter()
    sample_by_category = {}

    print(f"Scanning {len(label_files):,} labels...")

    for index, json_path in enumerate(label_files, start=1):
        with open(json_path, "r", encoding="utf-8") as f:
            record = json.load(f)

        image_id = record.get("name", json_path.stem)

        for obj in get_objects(record):
            category = str(obj.get("category", "")).strip().lower()

            if not category:
                continue

            category_counter[category] += 1

            if category not in sample_by_category:
                sample_by_category[category] = {
                    "image_id": image_id,
                    "object_keys": sorted(obj.keys()),
                    "attributes": obj.get("attributes", {}),
                }

        if index % 1000 == 0:
            print(f"Processed {index:,}/{len(label_files):,}")

    lines = []
    lines.append("BDD100K Raw Category Audit")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Total unique categories: {len(category_counter)}")
    lines.append("")

    lines.append("Raw category counts:")
    for category, count in category_counter.most_common():
        lines.append(f"  {category}: {count:,}")

    lines.append("")
    lines.append("First observed example for each category:")

    for category, _ in category_counter.most_common():
        sample = sample_by_category[category]
        lines.append("")
        lines.append(f"[{category}]")
        lines.append(f"  image_id: {sample['image_id']}")
        lines.append(f"  object_keys: {sample['object_keys']}")
        lines.append(f"  attributes: {sample['attributes']}")

    OUTPUT_FILE.write_text("\n".join(lines), encoding="utf-8")

    print("\nDone.")
    print(f"Unique categories: {len(category_counter)}")
    print(f"Saved: {OUTPUT_FILE}")

    print("\nTop raw categories:")
    for category, count in category_counter.most_common(20):
        print(f"  {category}: {count:,}")


if __name__ == "__main__":
    main()