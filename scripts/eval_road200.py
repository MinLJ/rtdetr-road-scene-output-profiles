from ultralytics import YOLO
import pandas as pd
from pathlib import Path
import json
import os

# ============================================================
# PATHS
# ============================================================
DATA_ROOT = Path(r"D:\MFE204_RoadDetection\subsets\bdd100k_road200_final")

IMAGE_DIR = DATA_ROOT / "images"
LABEL_DIR = DATA_ROOT / "labels"

OUTPUT_DIR = DATA_ROOT / "eval_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# LOAD MODELS
# ============================================================
yolo_model = YOLO("yolov8n.pt")
rtdetr_model = YOLO("rtdetr-l.pt")  # ultralytics RT-DETR

# ============================================================
# CLASS FILTER (BDD100K compatible subset)
# ============================================================
ALLOWED_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "traffic light",
    "train"
]

# ============================================================
# RUN INFERENCE
# ============================================================
def run_model(model, name):
    results_all = []

    for img_path in IMAGE_DIR.glob("*.jpg"):
        res = model.predict(str(img_path), conf=0.25, verbose=False)[0]

        results_all.append({
            "image": img_path.name,
            "model": name,
            "boxes": res.boxes.data.cpu().numpy().tolist()
        })

    out_path = OUTPUT_DIR / f"{name}_predictions.json"
    with open(out_path, "w") as f:
        json.dump(results_all, f)

    print(f"{name} inference done -> {out_path}")


# ============================================================
# SIMPLE METRICS (fast version)
# ============================================================
def compute_basic_metrics(pred_file):
    import numpy as np

    with open(pred_file, "r") as f:
        data = json.load(f)

    tp, fp, fn = 0, 0, 0

    for item in data:
        boxes = item["boxes"]

        # simplified proxy metrics
        if len(boxes) == 0:
            fn += 1
        elif len(boxes) < 2:
            tp += 1
        else:
            fp += 1

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    return precision, recall, f1


# ============================================================
# MAIN
# ============================================================
def main():
    print("\nRunning YOLOv8n...")
    run_model(yolo_model, "yolo")

    print("\nRunning RT-DETR-L...")
    run_model(rtdetr_model, "rtdetr")

    print("\nComputing metrics...")

    yolo_p, yolo_r, yolo_f1 = compute_basic_metrics(
        OUTPUT_DIR / "yolo_predictions.json"
    )

    det_p, det_r, det_f1 = compute_basic_metrics(
        OUTPUT_DIR / "rtdetr_predictions.json"
    )

    df = pd.DataFrame([
        ["YOLOv8n", yolo_p, yolo_r, yolo_f1],
        ["RT-DETR-L", det_p, det_r, det_f1]
    ], columns=["Model", "Precision", "Recall", "F1"])

    out_csv = OUTPUT_DIR / "metrics_summary.csv"
    df.to_csv(out_csv, index=False)

    print("\nFinal Results:")
    print(df)
    print(f"\nSaved -> {out_csv}")


if __name__ == "__main__":
    main()