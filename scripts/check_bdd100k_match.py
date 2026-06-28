from pathlib import Path
import json
import random

IMAGE_ROOT = Path(r"D:\MFE204_RoadDetection\data\bdd100k_images\bdd100k\images\100k")
LABEL_ROOT = Path(r"D:\MFE204_RoadDetection\data\bdd100k_labels\bdd100k\labels\100k")

for split in ["train", "val"]:
    image_dir = IMAGE_ROOT / split
    label_dir = LABEL_ROOT / split

    print(f"\n=== {split.upper()} ===")
    print("Image directory exists:", image_dir.exists())
    print("Label directory exists:", label_dir.exists())

    image_files = list(image_dir.glob("*.jpg"))
    label_files = list(label_dir.glob("*.json"))

    print("Images found:", len(image_files))
    print("JSON labels found:", len(label_files))

    if not image_files or not label_files:
        print("ERROR: Images or labels are missing. Stop here and check the paths.")
        continue

    samples = random.sample(label_files, min(10, len(label_files)))
    matched = 0

    for json_path in samples:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                record = json.load(f)

            image_name = record.get("name", "")
            expected_image = image_dir / f"{image_name}.jpg"

            exists = expected_image.exists()
            matched += int(exists)

            print(
                f"{json_path.name} -> {expected_image.name}: "
                f"{'OK' if exists else 'MISSING'}"
            )

        except Exception as exc:
            print(f"Could not read {json_path.name}: {exc}")

    print(f"Random-match result: {matched}/{len(samples)}")