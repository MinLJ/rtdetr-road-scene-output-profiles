from ultralytics import YOLO
from pathlib import Path
import json
import numpy as np
from collections import defaultdict

# ============================================================
# PATHS
# ============================================================
DATASET = Path(r"D:\MFE204_RoadDetection\subsets\bdd100k_road200_final")

IMAGE_DIR = DATASET / "images"
LABEL_DIR = DATASET / "labels"

# ============================================================
# MODELS
# ============================================================
models = {
    "yolo": YOLO("yolov8n.pt"),
    "rtdetr": YOLO("rtdetr-l.pt")
}

# ============================================================
# IOU
# ============================================================
def iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    inter = max(0, x2-x1) * max(0, y2-y1)
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])

    return inter / (area_a + area_b - inter + 1e-6)

# ============================================================
# GT LOADER (BDD100K correct)
# ============================================================
def load_gt(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    boxes = []
    labels = []

    for frame in data.get("frames", []):
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
# PRED
# ============================================================
def get_pred(model, img_path):
    res = model.predict(str(img_path), conf=0.25, verbose=False)[0]

    if res.boxes is None:
        return []

    return res.boxes.xyxy.cpu().numpy().tolist()

# ============================================================
# AP CALC (simplified COCO-style)
# ============================================================
def compute_ap(tp, fp, fn):
    tp = np.array(tp)
    fp = np.array(fp)

    if len(tp) == 0:
        return 0.0

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)

    # monotonic precision
    for i in range(len(precision)-2, -1, -1):
        precision[i] = max(precision[i], precision[i+1])

    return np.mean(precision)

# ============================================================
# EVAL
# ============================================================
def evaluate(model, name):

    TP, FP, FN = 0, 0, 0

    class_stats = defaultdict(lambda: {"tp":0, "fp":0, "fn":0})
    scenario_stats = defaultdict(lambda: {"tp":0, "fp":0, "fn":0})

    for img_path in IMAGE_DIR.glob("*.jpg"):

        image_id = img_path.stem
        json_path = LABEL_DIR / f"{image_id}.json"

        if not json_path.exists():
            continue

        gt_boxes, gt_labels = load_gt(json_path)
        pred_boxes = get_pred(model, img_path)

        matched = set()

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

                cls = gt_labels[best_gt]
                class_stats[cls]["tp"] += 1
            else:
                FP += 1

        FN += (len(gt_boxes) - len(matched))

        # class FN
        for i, cls in enumerate(gt_labels):
            if i not in matched:
                class_stats[cls]["fn"] += 1

        # scenario parsing
        scenario = img_path.stem.split("__")[0]
        scenario_stats[scenario]["tp"] += len(matched)
        scenario_stats[scenario]["fp"] += len(pred_boxes) - len(matched)
        scenario_stats[scenario]["fn"] += len(gt_boxes) - len(matched)

    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    return {
        "model": name,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "class_stats": dict(class_stats),
        "scenario_stats": dict(scenario_stats)
    }

# ============================================================
# MAIN
# ============================================================
def main():

    results = {}

    for name, model in models.items():
        print(f"\nRunning {name} ...")
        r = evaluate(model, name)

        print(f"{name}: P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f}")

        results[name] = r

    out = DATASET / "eval_results_coco_style.json"

    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved:", out)


if __name__ == "__main__":
    main()