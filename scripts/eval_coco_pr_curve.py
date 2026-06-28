from ultralytics import YOLO
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# ============================================================
# PATHS
# ============================================================
DATASET = Path(r"D:\MFE204_RoadDetection\subsets\bdd100k_road200_final")

IMAGE_DIR = DATASET / "images"
LABEL_DIR = DATASET / "labels"

OUT_DIR = DATASET / "eval_pr_results"
OUT_DIR.mkdir(exist_ok=True)

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
    area1 = (a[2]-a[0])*(a[3]-a[1])
    area2 = (b[2]-b[0])*(b[3]-b[1])

    return inter / (area1 + area2 - inter + 1e-6)

# ============================================================
# GT LOADER (BDD100K)
# ============================================================
def load_gt(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    boxes = []

    for frame in data.get("frames", []):
        for obj in frame.get("objects", []):
            if "box2d" not in obj:
                continue

            b = obj["box2d"]
            boxes.append([b["x1"], b["y1"], b["x2"], b["y2"]])

    return boxes

# ============================================================
# PRED WITH SCORES
# ============================================================
def get_pred(model, img_path):
    res = model.predict(str(img_path), conf=0.01, verbose=False)[0]

    if res.boxes is None:
        return [], []

    boxes = res.boxes.xyxy.cpu().numpy()
    scores = res.boxes.conf.cpu().numpy()

    return boxes, scores

# ============================================================
# COLLECT ALL SCORES
# ============================================================
def evaluate_pr(model):

    y_true = []
    y_score = []

    for img_path in IMAGE_DIR.glob("*.jpg"):

        image_id = img_path.stem
        json_path = LABEL_DIR / f"{image_id}.json"

        if not json_path.exists():
            continue

        gt_boxes = load_gt(json_path)
        pred_boxes, scores = get_pred(model, img_path)

        matched_gt = set()

        for i, pb in enumerate(pred_boxes):

            best_iou = 0
            best_gt = -1

            for j, gb in enumerate(gt_boxes):
                if j in matched_gt:
                    continue

                score = iou(pb, gb)

                if score > best_iou:
                    best_iou = score
                    best_gt = j

            if best_iou >= 0.5:
                y_true.append(1)
                matched_gt.add(best_gt)
            else:
                y_true.append(0)

            y_score.append(scores[i])

        # FN ignored in PR curve accumulation

    y_true = np.array(y_true)
    y_score = np.array(y_score)

    # sort by confidence
    idx = np.argsort(-y_score)
    y_true = y_true[idx]

    precision = []
    recall = []

    tp = 0
    fp = 0
    fn = np.sum(y_true == 1)

    for i in range(len(y_true)):
        if y_true[i] == 1:
            tp += 1
            fn -= 1
        else:
            fp += 1

        precision.append(tp / (tp + fp + 1e-6))
        recall.append(tp / (tp + fn + 1e-6))

    return precision, recall

# ============================================================
# AP (area under PR curve)
# ============================================================
def compute_ap(precision, recall):
    precision = np.array(precision)
    recall = np.array(recall)

    ap = 0
    for i in range(1, len(recall)):
        ap += (recall[i] - recall[i-1]) * precision[i]

    return ap

# ============================================================
# MAIN
# ============================================================
def main():

    results = {}

    for name, model in models.items():

        print(f"\nRunning PR for {name} ...")

        p, r = evaluate_pr(model)
        ap = compute_ap(p, r)

        results[name] = {
            "AP": float(ap),
            "precision_curve": p,
            "recall_curve": r
        }

        # plot
        plt.figure()
        plt.plot(r, p)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"PR Curve - {name}")
        plt.savefig(OUT_DIR / f"{name}_pr_curve.png")

        print(f"{name} AP: {ap:.4f}")

    with open(OUT_DIR / "pr_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved to:", OUT_DIR)


if __name__ == "__main__":
    main()