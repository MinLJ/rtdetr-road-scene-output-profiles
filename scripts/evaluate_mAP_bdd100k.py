from ultralytics import YOLO
from pathlib import Path
import json
import numpy as np

# ============================================================
# PATHS
# ============================================================
DATASET = Path(r"D:\MFE204_RoadDetection\subsets\bdd100k_road200_final")

IMAGE_DIR = DATASET / "images"
LABEL_DIR = DATASET / "labels"

# ============================================================
# MODELS
# ============================================================
yolo_model = YOLO("yolov8n.pt")
rtdetr_model = YOLO("rtdetr-l.pt")

MODELS = {
    "yolo": yolo_model,
    "rtdetr": rtdetr_model
}

# ============================================================
# IOU
# ============================================================
def iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (a[2] - a[0]) * (a[3] - a[1])
    area2 = (b[2] - b[0]) * (b[3] - b[1])

    return inter / (area1 + area2 - inter + 1e-6)

# ============================================================
# GT LOADER (FIXED BDD100K FORMAT)
# ============================================================
def load_gt(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    boxes = []
    labels = []

    if "frames" in data:
        for frame in data["frames"]:
            for obj in frame.get("objects", []):
                if "box2d" not in obj:
                    continue

                b = obj["box2d"]

                boxes.append([
                    b["x1"], b["y1"],
                    b["x2"], b["y2"]
                ])
                labels.append(obj["category"])

    return boxes, labels

# ============================================================
# PREDICTION
# ============================================================
def get_pred(model, img_path):
    res = model.predict(str(img_path), conf=0.25, verbose=False)[0]

    if res.boxes is None:
        return [], []

    boxes = res.boxes.xyxy.cpu().numpy().tolist()

    return boxes

# ============================================================
# EVAL CORE
# ============================================================
def evaluate(model, name):

    TP, FP, FN = 0, 0, 0

    for img_path in IMAGE_DIR.glob("*.jpg"):

        image_id = img_path.stem  # 保留完整名字
        json_path = LABEL_DIR / f"{image_id}.json"

        if not json_path.exists():
            print("[GT MISSING]", json_path)
            continue

        gt_boxes, gt_labels = load_gt(json_path)
        pred_boxes = get_pred(model, img_path)

        # DEBUG ON EMPTY
        if len(gt_boxes) == 0:
            print("[GT EMPTY]", json_path)
        if len(pred_boxes) == 0:
            print("[PRED EMPTY]", img_path)

        matched = set()

        # ============================
        # MATCHING
        # ============================
        for pb in pred_boxes:

            best_iou = 0
            best_gt = -1

            for i, gb in enumerate(gt_boxes):
                if i in matched:
                    continue

                score = iou(pb, gb)

                if score > best_iou:
                    best_iou = score
                    best_gt = i

            if best_iou >= 0.5:
                TP += 1
                matched.add(best_gt)
            else:
                FP += 1

        FN += (len(gt_boxes) - len(matched))

    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    print(f"\n{name}")
    print(f"TP={TP}, FP={FP}, FN={FN}")
    print(f"P={precision:.4f} R={recall:.4f} F1={f1:.4f}")

    return precision, recall, f1

# ============================================================
# MAIN
# ============================================================
def main():

    results = {}

    for name, model in MODELS.items():
        print(f"\nRunning {name} ...")
        p, r, f1 = evaluate(model, name)

        results[name] = {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f1)
        }

    out = DATASET / "eval_results_fixed.json"

    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved:", out)


if __name__ == "__main__":
    main()