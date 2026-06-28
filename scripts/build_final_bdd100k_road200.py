from pathlib import Path
import csv
import shutil
from collections import defaultdict

# ============================================================
# PATHS
# ============================================================
BASE = Path(r"D:\MFE204_RoadDetection")

REVIEW_DIR = BASE / "subsets" / "bdd100k_road200_v1" / "fullsize_review" / "review_csv"

IMAGE_ROOT = BASE / "data" / "bdd100k_images"
LABEL_ROOT = BASE / "data" / "bdd100k_labels"

OUTPUT_ROOT = BASE / "subsets" / "bdd100k_road200_final"
OUT_IMG = OUTPUT_ROOT / "images"
OUT_LAB = OUTPUT_ROOT / "labels"
LOG_FILE = OUTPUT_ROOT / "missing_files.log"
MANIFEST = OUTPUT_ROOT / "manifest.csv"

SCENARIOS = [
    "daytime_normal",
    "night_lowlight",
    "crowded_occluded",
    "small_distant",
]

OUT_IMG.mkdir(parents=True, exist_ok=True)
OUT_LAB.mkdir(parents=True, exist_ok=True)


# ============================================================
# SAFE IMAGE FINDER (关键修复)
# ============================================================
def find_image(image_id: str):
    exts = [".jpg", ".png", ".jpeg"]

    # 1. direct match in root folders
    for ext in exts:
        p = IMAGE_ROOT / "bdd100k" / "images" / "100k" / f"{image_id}{ext}"
        if p.exists():
            return p

    # 2. fallback: recursive search (slow but safe)
    for ext in exts:
        matches = list(IMAGE_ROOT.rglob(f"{image_id}{ext}"))
        if matches:
            return matches[0]

    return None


# ============================================================
# LOAD CSV
# ============================================================
def load_kept(csv_path):
    rows = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["visual_review_decision"] == "keep":
                rows.append(r)
    return rows


# ============================================================
# COPY SAFE PAIR
# ============================================================
def copy_pair(row, scenario, log):
    image_id = row["image_id"]

    img_src = find_image(image_id)
    if img_src is None:
        log.append(f"[MISSING IMAGE] {image_id}")
        return None

    label_src = LABEL_ROOT.rglob(f"{image_id}.json")
    label_src = next(label_src, None)

    if label_src is None:
        log.append(f"[MISSING LABEL] {image_id}")
        return None

    new_name = f"{scenario}__{image_id}"

    img_dst = OUT_IMG / f"{new_name}.jpg"
    lab_dst = OUT_LAB / f"{new_name}.json"

    shutil.copy2(img_src, img_dst)
    shutil.copy2(label_src, lab_dst)

    return {
        "image_id": new_name,
        "original_id": image_id,
        "scenario": scenario,
        "image_path": str(img_dst),
        "label_path": str(lab_dst),
    }


# ============================================================
# MAIN
# ============================================================
def main():
    all_samples = []
    log = []

    for sc in SCENARIOS:
        csv_path = REVIEW_DIR / f"review_{sc}_70_fullsize.csv"

        if not csv_path.exists():
            raise FileNotFoundError(csv_path)

        kept = load_kept(csv_path)

        print(f"{sc}: kept {len(kept)}")

        for r in kept:
            res = copy_pair(r, sc, log)
            if res:
                all_samples.append(res)

    # ========================================================
    # SAVE MANIFEST
    # ========================================================
    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_samples[0].keys())
        writer.writeheader()
        writer.writerows(all_samples)

    # ========================================================
    # SAVE LOG
    # ========================================================
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(log))

    print("\nDONE")
    print(f"Total samples: {len(all_samples)}")
    print(f"Missing files: {len(log)}")
    print(f"Saved dataset: {OUTPUT_ROOT}")
    print(f"Log: {LOG_FILE}")


if __name__ == "__main__":
    main()