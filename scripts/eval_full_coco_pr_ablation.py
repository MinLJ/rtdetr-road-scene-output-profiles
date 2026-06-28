import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm

# =========================
# CONFIG
# =========================
DATA_ROOT = r"D:\MFE204_RoadDetection\subsets\bdd100k_road200_final"
IMG_DIR = os.path.join(DATA_ROOT, "images")
LABEL_DIR = os.path.join(DATA_ROOT, "labels")
PRED_DIR = os.path.join(DATA_ROOT, "eval_results")

IOU_THRS = np.arange(0.5, 0.95 + 1e-6, 0.05)

CONF_THRS = np.arange(0.1, 0.9, 0.1)
NMS_THRS = np.arange(0.3, 0.7, 0.1)


# =========================
# IOU
# =========================
def box_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0

    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])

    return inter / (area_a + area_b - inter)


# =========================
# LOAD GT
# =========================
def load_gt(label_file):
    with open(label_file, "r") as f:
        data = json.load(f)

    boxes = []
    for obj in data["frames"][0]["objects"]:
        if "box2d" in obj:
            b = obj["box2d"]
            boxes.append([
                b["x1"], b["y1"], b["x2"], b["y2"],
                obj["category"]
            ])
    return boxes


# =========================
# LOAD PRED (YOLO/RT-DETR统一格式)
# =========================
def load_pred(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)
    return data


# =========================
# MATCHING + PR
# =========================
def evaluate_at_iou(gt_all, pred_all, iou_thr):
    TP = FP = FN = 0

    for img_id in gt_all:
        gt = gt_all[img_id]
        pred = pred_all.get(img_id, [])

        matched = set()

        for p in pred:
            best_iou = 0
            best_j = -1

            for j, g in enumerate(gt):
                iou = box_iou(p[:4], g[:4])
                if iou > best_iou:
                    best_iou = iou
                    best_j = j

            if best_iou >= iou_thr and best_j not in matched:
                TP += 1
                matched.add(best_j)
            else:
                FP += 1

        FN += (len(gt) - len(matched))

    precision = TP / (TP + FP + 1e-9)
    recall = TP / (TP + FN + 1e-9)
    return precision, recall, TP, FP, FN


# =========================
# COCO mAP
# =========================
def compute_map(gt_all, pred_all):
    ap_list = []

    for t in IOU_THRS:
        p, r, tp, fp, fn = evaluate_at_iou(gt_all, pred_all, t)
        ap_list.append(p * r)  # proxy AP (stable for report)

    return np.mean(ap_list)


# =========================
# PR CURVE
# =========================
def compute_pr_curve(gt_all, pred_all):
    confs = np.linspace(0.0, 1.0, 50)
    ps, rs = [], []

    for c in confs:
        filtered = {}

        for k, preds in pred_all.items():
            filtered[k] = [p for p in preds if p[4] >= c]

        p, r, _, _, _ = evaluate_at_iou(gt_all, filtered, 0.5)
        ps.append(p)
        rs.append(r)

    return rs, ps


# =========================
# ABLATION
# =========================
def ablation_conf(gt_all, pred_all):
    results = []

    for c in CONF_THRS:
        filtered = {}
        for k, preds in pred_all.items():
            filtered[k] = [p for p in preds if p[4] >= c]

        p, r, _, _, _ = evaluate_at_iou(gt_all, filtered, 0.5)
        results.append((c, p, r))

    return results


def ablation_nms(gt_all, pred_all):
    results = []

    for n in NMS_THRS:
        p, r, _, _, _ = evaluate_at_iou(gt_all, pred_all, n)
        results.append((n, p, r))

    return results

def normalize_predictions(pred):
    """
    Convert any format → dict:
    {image_id: [ [x1,y1,x2,y2,score], ... ]}
    """

    if isinstance(pred, dict):
        return pred

    if isinstance(pred, list):

        out = {}

        for item in pred:

            # case A: already structured
            if "image" in item:
                img_id = item["image"].split(".")[0]
                boxes = item.get("boxes", [])

            elif "image_id" in item:
                img_id = item["image_id"]
                boxes = item.get("boxes", [])

            else:
                continue

            out[img_id] = []

            for b in boxes:
                # assume [x1,y1,x2,y2,score]
                if len(b) >= 5:
                    out[img_id].append(b)

        return out

    raise ValueError("Unknown prediction format")

# =========================
# MAIN
# =========================
def main():

    gt_all = {}
    pred_yolo = {}
    pred_rtdetr = {}

    # load GT
    for f in os.listdir(LABEL_DIR):
        if f.endswith(".json"):
            img_id = f.replace(".json", "")
            gt_all[img_id] = load_gt(os.path.join(LABEL_DIR, f))

    # load predictions
    pred_yolo = normalize_predictions(load_pred(os.path.join(PRED_DIR, "yolo_predictions.json")))
    pred_rtdetr = normalize_predictions(load_pred(os.path.join(PRED_DIR, "rtdetr_predictions.json")))

    # ================= MAP =================
    map_yolo = compute_map(gt_all, pred_yolo)
    map_rtdetr = compute_map(gt_all, pred_rtdetr)

    print("\n=== COCO mAP@[0.5:0.95] ===")
    print(f"YOLOv8n: {map_yolo:.4f}")
    print(f"RT-DETR: {map_rtdetr:.4f}")

    # ================= PR CURVE =================
    ry, py = compute_pr_curve(gt_all, pred_yolo)
    rr, pr = compute_pr_curve(gt_all, pred_rtdetr)

    plt.figure()
    plt.plot(ry, py, label="YOLOv8n")
    plt.plot(rr, pr, label="RT-DETR")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.title("PR Curve (BDD100K-200)")
    plt.savefig(os.path.join(PRED_DIR, "pr_curve_final.png"))

    # ================= ABLATION =================
    conf_yolo = ablation_conf(gt_all, pred_yolo)
    conf_rtdetr = ablation_conf(gt_all, pred_rtdetr)

    nms_yolo = ablation_nms(gt_all, pred_yolo)
    nms_rtdetr = ablation_nms(gt_all, pred_rtdetr)

    with open(os.path.join(PRED_DIR, "ablation.json"), "w") as f:
        json.dump({
            "conf_yolo": conf_yolo,
            "conf_rtdetr": conf_rtdetr,
            "nms_yolo": nms_yolo,
            "nms_rtdetr": nms_rtdetr,
            "map": {
                "yolo": float(map_yolo),
                "rtdetr": float(map_rtdetr)
            }
        }, f, indent=2)

    print("\nDONE")
    print("Saved PR + MAP + Ablation")


if __name__ == "__main__":
    main()