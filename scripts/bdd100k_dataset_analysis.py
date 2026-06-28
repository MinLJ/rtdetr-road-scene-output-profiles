import json
from pathlib import Path
import pandas as pd
from collections import defaultdict

# =========================
# PATH
# =========================
LABEL_DIR = Path(r"D:\MFE204_RoadDetection\subsets\bdd100k_road200_final\labels")

# =========================
# STATS
# =========================
total_images = 0
total_objects = 0

category_count = defaultdict(int)
objects_per_image = []

crowded_count = 0
small_count = 0
occluded_count = 0

rows = []

# =========================
# LOOP ALL LABELS
# =========================
for json_path in LABEL_DIR.glob("*.json"):

    with open(json_path, "r") as f:
        data = json.load(f)

    img_objects = 0
    is_crowded = False

    # BDD100K format
    if "frames" not in data:
        continue

    for frame in data["frames"]:
        objs = frame.get("objects", [])

        img_objects += len(objs)

        for obj in objs:
            cat = obj.get("category", "unknown")
            category_count[cat] += 1

            attrs = obj.get("attributes", {})

            if attrs.get("occluded", False):
                occluded_count += 1
                is_crowded = True

            # small object heuristic
            if "box2d" in obj:
                b = obj["box2d"]
                w = b["x2"] - b["x1"]
                h = b["y2"] - b["y1"]
                area = w * h

                if area < 2000:
                    small_count += 1

    total_objects += img_objects
    total_images += 1
    objects_per_image.append(img_objects)

    if img_objects > 25:
        crowded_count += 1

    rows.append({
        "image": json_path.name,
        "objects": img_objects
    })

# =========================
# RESULTS
# =========================
avg_objects = total_objects / max(total_images, 1)

df = pd.DataFrame(rows)

# =========================
# PRINT SUMMARY
# =========================
print("\n================ DATASET ANALYSIS ================\n")

print(f"Total images: {total_images}")
print(f"Total objects: {total_objects}")
print(f"Average objects per image: {avg_objects:.2f}")

print("\n---- Top categories ----")
for k, v in sorted(category_count.items(), key=lambda x: -x[1])[:15]:
    print(f"{k}: {v}")

print("\n---- Density stats ----")
print(f"crowded images (>25 objects): {crowded_count}")
print(f"small objects: {small_count}")
print(f"occluded objects: {occluded_count}")

print("\n==================================================\n")

# =========================
# SAVE CSV
# =========================
out_csv = LABEL_DIR.parent / "bdd100k_dataset_stats.csv"
df.to_csv(out_csv, index=False)

print("Saved CSV:", out_csv)