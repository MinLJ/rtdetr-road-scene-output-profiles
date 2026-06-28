from __future__ import annotations

"""
Road200 class-aware COCO evaluator for BDD100K-derived labels.

What this script does
---------------------
1. Reads the locked Road200 manifest.
2. Converts only the eight BDD100K classes that have a clean COCO-model match:
   person, bike->bicycle, car, motor->motorcycle, bus, train, truck, traffic light.
3. Runs YOLOv8n and/or RT-DETR-L with low-score inference to retain ranked detections.
4. Evaluates predictions with the official pycocotools COCO bbox evaluator:
   AP@[0.50:0.95], AP50, AP75, AR@1/10/100.
5. Computes class-aware Precision / Recall / F1 at a clearly stated operating
   threshold (default: 0.25) using one-to-one IoU matching at IoU=0.50.
6. Writes overall, per-class, per-scenario, IoU-sweep, PR-curve, prediction,
   reproducibility-index, and environment files.

Important:
- This script intentionally ignores BDD categories that cannot be compared
  cleanly with COCO-pretrained models (for example: traffic sign, rider,
  lane/*, and area/*).
- It is a baseline evaluator. Do not change model post-processing parameters
  here while establishing baseline results. The ablation runner should use
  this script's exported COCO GT and prediction schema afterwards.
"""

import argparse
import csv
import json
import platform
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
import ultralytics
from ultralytics import YOLO

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except ImportError as exc:
    raise SystemExit(
        "\npycocotools is required for this evaluator.\n"
        "Install it in the active road_det_bdd environment with:\n"
        "  python -m pip install pycocotools\n"
    ) from exc


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------
BASE = Path(r"D:\MFE204_RoadDetection")
DATASET_ROOT = BASE / "subsets" / "bdd100k_road200_final"
MANIFEST_PATH = DATASET_ROOT / "manifest.csv"
OUTPUT_ROOT = DATASET_ROOT / "eval_coco_standard"

# ---------------------------------------------------------------------
# Evaluation taxonomy
# Only these eight classes are shared cleanly between BDD100K and COCO.
# ---------------------------------------------------------------------
CANONICAL_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "train",
    "truck",
    "traffic light",
]

CATEGORY_ID = {name: index + 1 for index, name in enumerate(CANONICAL_CLASSES)}

BDD_TO_CANONICAL = {
    "person": "person",
    "bike": "bicycle",
    "car": "car",
    "motor": "motorcycle",
    "bus": "bus",
    "train": "train",
    "truck": "truck",
    "traffic light": "traffic light",
}

# Ultralytics' COCO class names are mapped dynamically from each model result,
# then normalized through this dictionary.
MODEL_TO_CANONICAL = {
    "person": "person",
    "bicycle": "bicycle",
    "car": "car",
    "motorcycle": "motorcycle",
    "bus": "bus",
    "train": "train",
    "truck": "truck",
    "traffic light": "traffic light",
}

MODEL_WEIGHTS = {
    "yolo": "yolov8n.pt",
    "rtdetr": "rtdetr-l.pt",
}

EXPECTED_SCENARIOS = [
    "daytime_normal",
    "night_lowlight",
    "crowded_occluded",
    "small_distant",
]


def normalize_name(value: Any) -> str:
    """Normalize category labels from BDD100K and Ultralytics."""
    return str(value).strip().lower().replace("_", " ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Class-aware official COCO evaluation for BDD100K Road200."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["yolo", "rtdetr"],
        choices=sorted(MODEL_WEIGHTS),
        help="Models to run. Default: yolo rtdetr",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Evaluate only 10 manifest images to validate the pipeline.",
    )
    parser.add_argument(
        "--device",
        default="0",
        help="Ultralytics device argument. Default: 0",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size. Default: 640",
    )
    parser.add_argument(
        "--inference-conf",
        type=float,
        default=0.001,
        help=(
            "Low score floor used to retain ranked detections for AP. "
            "Default: 0.001"
        ),
    )
    parser.add_argument(
        "--inference-iou",
        type=float,
        default=0.70,
        help=(
            "Model-native NMS IoU setting where applicable. "
            "This is recorded as the baseline configuration. Default: 0.70"
        ),
    )
    parser.add_argument(
        "--max-det",
        type=int,
        default=300,
        help=(
            "Maximum model detections retained before COCO evaluation. "
            "COCO reports its standard AP at maxDets=100. Default: 300"
        ),
    )
    parser.add_argument(
        "--operating-conf",
        type=float,
        default=0.25,
        help=(
            "Score threshold for reported class-aware P/R/F1 operating point. "
            "Default: 0.25"
        ),
    )
    return parser.parse_args()


def read_manifest(smoke_test: bool) -> list[dict[str, Any]]:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")

    with MANIFEST_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    required_columns = {
        "image_id",
        "original_id",
        "scenario",
        "image_path",
        "label_path",
    }
    missing = required_columns.difference(rows[0].keys() if rows else set())
    if missing:
        raise RuntimeError(
            "manifest.csv is missing required columns: "
            + ", ".join(sorted(missing))
        )

    rows.sort(key=lambda row: (row["scenario"], row["image_id"]))

    if smoke_test:
        rows = rows[:10]

    if not rows:
        raise RuntimeError("No rows found in manifest.csv.")

    if not smoke_test and len(rows) != 200:
        raise RuntimeError(
            f"Expected the locked Road200 manifest to contain 200 images, "
            f"but found {len(rows)}."
        )

    seen_ids = set()
    samples: list[dict[str, Any]] = []

    for coco_image_id, row in enumerate(rows, start=1):
        image_path = Path(row["image_path"])
        label_path = Path(row["label_path"])

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image listed in manifest does not exist: {image_path}"
            )
        if not label_path.exists():
            raise FileNotFoundError(
                f"Label listed in manifest does not exist: {label_path}"
            )
        if row["image_id"] in seen_ids:
            raise RuntimeError(f"Duplicate manifest image_id: {row['image_id']}")
        seen_ids.add(row["image_id"])

        with Image.open(image_path) as image:
            width, height = image.size

        samples.append(
            {
                "coco_image_id": coco_image_id,
                "image_id": row["image_id"],
                "original_id": row["original_id"],
                "scenario": row["scenario"],
                "image_path": str(image_path),
                "label_path": str(label_path),
                "file_name": image_path.name,
                "width": width,
                "height": height,
            }
        )

    return samples


def bdd_objects_from_label(label_path: Path) -> list[dict[str, Any]]:
    """
    Read the BDD100K per-image label format used in this project:
    {name, frames: [{timestamp, objects: [...]}], attributes}.
    Only the first frame belongs to the selected still image.
    """
    with label_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    frames = data.get("frames", [])
    if not frames:
        return []

    objects = frames[0].get("objects", [])
    if not isinstance(objects, list):
        return []

    return objects


def valid_xyxy(box: dict[str, Any]) -> tuple[float, float, float, float] | None:
    try:
        x1 = float(box["x1"])
        y1 = float(box["y1"])
        x2 = float(box["x2"])
        y2 = float(box["y2"])
    except (KeyError, TypeError, ValueError):
        return None

    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)

    if width <= 0.0 or height <= 0.0:
        return None

    return x1, y1, x2, y2


def build_coco_ground_truth(
    samples: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[int, list[dict[str, Any]]], list[dict[str, Any]]]:
    coco_images: list[dict[str, Any]] = []
    coco_annotations: list[dict[str, Any]] = []
    gt_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    audit_rows: list[dict[str, Any]] = []

    annotation_id = 1

    for sample in samples:
        coco_image_id = sample["coco_image_id"]
        coco_images.append(
            {
                "id": coco_image_id,
                "file_name": sample["file_name"],
                "width": sample["width"],
                "height": sample["height"],
                "scenario": sample["scenario"],
                "source_image_id": sample["image_id"],
                "source_original_id": sample["original_id"],
            }
        )

        raw_objects = bdd_objects_from_label(Path(sample["label_path"]))
        raw_bbox_objects = 0
        evaluated_objects = 0
        ignored_objects = 0

        for obj in raw_objects:
            box = valid_xyxy(obj.get("box2d", {}))
            if box is None:
                continue

            raw_bbox_objects += 1
            raw_category = normalize_name(obj.get("category", ""))
            canonical_category = BDD_TO_CANONICAL.get(raw_category)

            if canonical_category is None:
                ignored_objects += 1
                continue

            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            category_id = CATEGORY_ID[canonical_category]

            annotation = {
                "id": annotation_id,
                "image_id": coco_image_id,
                "category_id": category_id,
                "bbox": [x1, y1, width, height],
                "area": width * height,
                "iscrowd": 0,
            }

            coco_annotations.append(annotation)
            gt_by_image[coco_image_id].append(annotation)
            annotation_id += 1
            evaluated_objects += 1

        audit_rows.append(
            {
                "coco_image_id": coco_image_id,
                "image_id": sample["image_id"],
                "scenario": sample["scenario"],
                "raw_bbox_objects": raw_bbox_objects,
                "evaluated_shared_class_objects": evaluated_objects,
                "ignored_unmapped_bbox_objects": ignored_objects,
            }
        )

    coco_ground_truth = {
        "info": {
            "description": (
                "BDD100K Road200 class-aware evaluation ground truth. "
                "Only eight BDD-to-COCO compatible object classes are included."
            ),
            "version": "1.0",
        },
        "licenses": [],
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": [
            {"id": CATEGORY_ID[name], "name": name, "supercategory": "road_object"}
            for name in CANONICAL_CLASSES
        ],
    }

    return coco_ground_truth, gt_by_image, audit_rows


def class_name_from_result(result: Any, class_index: int) -> str:
    names = result.names
    if isinstance(names, dict):
        return normalize_name(names[class_index])
    return normalize_name(names[class_index])


def run_model_predictions(
    model_key: str,
    samples: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model = YOLO(MODEL_WEIGHTS[model_key])
    predictions: list[dict[str, Any]] = []
    source_class_counter: Counter[str] = Counter()
    retained_class_counter: Counter[str] = Counter()
    empty_prediction_images = 0

    start_time = time.perf_counter()

    for index, sample in enumerate(samples, start=1):
        result = model.predict(
            source=sample["image_path"],
            conf=args.inference_conf,
            iou=args.inference_iou,
            max_det=args.max_det,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )[0]

        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            empty_prediction_images += 1
            continue

        xyxy = boxes.xyxy.detach().cpu().numpy()
        confidence = boxes.conf.detach().cpu().numpy()
        classes = boxes.cls.detach().cpu().numpy().astype(int)

        for coordinates, score, class_index in zip(xyxy, confidence, classes):
            source_name = class_name_from_result(result, int(class_index))
            source_class_counter[source_name] += 1

            canonical_category = MODEL_TO_CANONICAL.get(source_name)
            if canonical_category is None:
                continue

            x1, y1, x2, y2 = [float(value) for value in coordinates.tolist()]
            width = max(0.0, x2 - x1)
            height = max(0.0, y2 - y1)

            if width <= 0.0 or height <= 0.0:
                continue

            predictions.append(
                {
                    "image_id": sample["coco_image_id"],
                    "category_id": CATEGORY_ID[canonical_category],
                    "bbox": [x1, y1, width, height],
                    "score": float(score),
                }
            )
            retained_class_counter[canonical_category] += 1

        if index % 25 == 0 or index == len(samples):
            print(f"  {model_key}: inferred {index}/{len(samples)} images")

    elapsed_seconds = time.perf_counter() - start_time
    metadata = {
        "weight_file": MODEL_WEIGHTS[model_key],
        "inference_images": len(samples),
        "elapsed_seconds": elapsed_seconds,
        "mean_seconds_per_image": elapsed_seconds / len(samples),
        "empty_prediction_images": empty_prediction_images,
        "all_model_predicted_class_counts": dict(source_class_counter),
        "retained_shared_class_prediction_counts": dict(retained_class_counter),
    }
    return predictions, metadata


def mean_valid(values: np.ndarray) -> float | None:
    valid = values[values > -1]
    return float(np.mean(valid)) if valid.size else None


def run_coco_eval(
    coco_gt: COCO,
    coco_dt: COCO,
    image_ids: list[int],
) -> COCOeval:
    evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
    evaluator.params.imgIds = sorted(image_ids)
    evaluator.params.catIds = [CATEGORY_ID[name] for name in CANONICAL_CLASSES]
    # Match the official COCO implementation exactly. Using np.arange here can
    # represent 0.75 as 0.7500000000000002, making pycocotools print AP75=-1.
    evaluator.params.iouThrs = np.linspace(0.50, 0.95, 10)
    evaluator.params.maxDets = [1, 10, 100]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return evaluator


def metrics_from_coco_eval(evaluator: COCOeval) -> dict[str, float]:
    return {
        "mAP_50_95": float(evaluator.stats[0]),
        "AP50": float(evaluator.stats[1]),
        "AP75": float(evaluator.stats[2]),
        "AR1": float(evaluator.stats[6]),
        "AR10": float(evaluator.stats[7]),
        "AR100": float(evaluator.stats[8]),
    }


def ap_by_iou(evaluator: COCOeval) -> list[dict[str, float]]:
    # precision dimensions: [IoU, recall, category, area, maxDet]
    precision = evaluator.eval["precision"]
    rows: list[dict[str, float]] = []

    for threshold_index, threshold in enumerate(evaluator.params.iouThrs):
        values = precision[threshold_index, :, :, 0, 2]
        ap_value = mean_valid(values)
        rows.append(
            {
                "iou_threshold": round(float(threshold), 2),
                "macro_ap": float(ap_value) if ap_value is not None else float("nan"),
            }
        )
    return rows


def per_class_ap(
    evaluator: COCOeval,
    coco_gt: COCO,
) -> list[dict[str, Any]]:
    precision = evaluator.eval["precision"]
    category_ids = evaluator.params.catIds

    threshold_index_50 = int(
        np.where(np.isclose(evaluator.params.iouThrs, 0.50))[0][0]
    )
    threshold_index_75 = int(
        np.where(np.isclose(evaluator.params.iouThrs, 0.75))[0][0]
    )

    gt_counts = Counter(annotation["category_id"] for annotation in coco_gt.dataset["annotations"])
    category_name = {
        category["id"]: category["name"]
        for category in coco_gt.dataset["categories"]
    }

    rows: list[dict[str, Any]] = []

    for category_index, category_id in enumerate(category_ids):
        ap = mean_valid(precision[:, :, category_index, 0, 2])
        ap50 = mean_valid(precision[threshold_index_50, :, category_index, 0, 2])
        ap75 = mean_valid(precision[threshold_index_75, :, category_index, 0, 2])

        rows.append(
            {
                "class_name": category_name[category_id],
                "gt_instances": int(gt_counts[category_id]),
                "AP_50_95": ap,
                "AP50": ap50,
                "AP75": ap75,
            }
        )

    return rows


def macro_pr_at_iou50(evaluator: COCOeval) -> list[dict[str, float]]:
    precision = evaluator.eval["precision"]
    threshold_index_50 = int(
        np.where(np.isclose(evaluator.params.iouThrs, 0.50))[0][0]
    )

    # dimensions after indexing: [recall, category]
    precision_at_50 = precision[threshold_index_50, :, :, 0, 2]
    rows: list[dict[str, float]] = []

    for recall_index, recall_value in enumerate(evaluator.params.recThrs):
        macro_precision = mean_valid(precision_at_50[recall_index, :])
        if macro_precision is not None:
            rows.append(
                {
                    "recall": float(recall_value),
                    "precision": float(macro_precision),
                }
            )

    return rows


def xywh_to_xyxy(box: list[float]) -> tuple[float, float, float, float]:
    x, y, width, height = box
    return x, y, x + width, y + height


def box_iou_xywh(first: list[float], second: list[float]) -> float:
    ax1, ay1, ax2, ay2 = xywh_to_xyxy(first)
    bx1, by1, bx2, by2 = xywh_to_xyxy(second)

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if intersection <= 0.0:
        return 0.0

    first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = first_area + second_area - intersection

    return intersection / union if union > 0.0 else 0.0


def operating_point_metrics(
    gt_by_image: dict[int, list[dict[str, Any]]],
    predictions: list[dict[str, Any]],
    image_ids: list[int],
    score_threshold: float,
    iou_threshold: float = 0.50,
) -> dict[str, float | int]:
    """
    Class-aware one-to-one matching used only for a stated operating point.
    This is deliberately separate from COCO AP, which integrates ranked
    predictions across score thresholds.
    """
    predictions_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        if prediction["score"] >= score_threshold:
            predictions_by_image[prediction["image_id"]].append(prediction)

    true_positive = 0
    false_positive = 0
    false_negative = 0

    for image_id in image_ids:
        ground_truths = gt_by_image.get(image_id, [])
        predictions_for_image = predictions_by_image.get(image_id, [])

        gt_by_category: dict[int, list[dict[str, Any]]] = defaultdict(list)
        pred_by_category: dict[int, list[dict[str, Any]]] = defaultdict(list)

        for ground_truth in ground_truths:
            gt_by_category[ground_truth["category_id"]].append(ground_truth)
        for prediction in predictions_for_image:
            pred_by_category[prediction["category_id"]].append(prediction)

        for category_id in CATEGORY_ID.values():
            category_ground_truths = gt_by_category.get(category_id, [])
            category_predictions = sorted(
                pred_by_category.get(category_id, []),
                key=lambda item: item["score"],
                reverse=True,
            )
            matched_gt_indices: set[int] = set()

            for prediction in category_predictions:
                best_iou = 0.0
                best_gt_index = -1

                for gt_index, ground_truth in enumerate(category_ground_truths):
                    if gt_index in matched_gt_indices:
                        continue

                    iou_value = box_iou_xywh(
                        prediction["bbox"],
                        ground_truth["bbox"],
                    )
                    if iou_value > best_iou:
                        best_iou = iou_value
                        best_gt_index = gt_index

                if best_iou >= iou_threshold and best_gt_index >= 0:
                    true_positive += 1
                    matched_gt_indices.add(best_gt_index)
                else:
                    false_positive += 1

            false_negative += len(category_ground_truths) - len(matched_gt_indices)

    precision = (
        true_positive / (true_positive + false_positive)
        if (true_positive + false_positive) > 0
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if (true_positive + false_negative) > 0
        else 0.0
    )
    f1_score = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0.0
        else 0.0
    )

    return {
        "TP": true_positive,
        "FP": false_positive,
        "FN": false_negative,
        "precision": precision,
        "recall": recall,
        "F1": f1_score,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def create_figures(
    per_model_pr: dict[str, list[dict[str, float]]],
    per_model_iou: dict[str, list[dict[str, float]]],
    overall_rows: list[dict[str, Any]],
    figures_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)

    # Macro PR@0.50 over the eight shared classes.
    plt.figure(figsize=(6.8, 5.0))
    for model_name, points in per_model_pr.items():
        if points:
            recall = [point["recall"] for point in points]
            precision = [point["precision"] for point in points]
            plt.plot(recall, precision, label=model_name)
    plt.xlabel("Recall")
    plt.ylabel("Macro precision")
    plt.title("Macro Precision--Recall Curves at IoU = 0.50")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "macro_pr_iou50.png", dpi=300)
    plt.close()

    # AP degradation across the IoU sweep.
    plt.figure(figsize=(6.8, 5.0))
    for model_name, rows in per_model_iou.items():
        thresholds = [row["iou_threshold"] for row in rows]
        average_precision = [row["macro_ap"] for row in rows]
        plt.plot(thresholds, average_precision, marker="o", label=model_name)
    plt.xlabel("IoU threshold")
    plt.ylabel("Macro AP")
    plt.title("Localization Sensitivity Across IoU Thresholds")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "ap_by_iou_threshold.png", dpi=300)
    plt.close()

    # Overall metric comparison.
    model_names = [row["model"] for row in overall_rows]
    metrics = ["mAP_50_95", "AP50", "AP75", "precision", "recall", "F1"]
    positions = np.arange(len(metrics))
    width = 0.35 if len(model_names) == 2 else 0.8 / max(len(model_names), 1)

    plt.figure(figsize=(9.5, 5.2))
    for index, row in enumerate(overall_rows):
        offset = (index - (len(overall_rows) - 1) / 2) * width
        values = [row[metric] for metric in metrics]
        plt.bar(positions + offset, values, width=width, label=row["model"])
    plt.xticks(positions, ["mAP", "AP50", "AP75", "P@0.25", "R@0.25", "F1@0.25"])
    plt.ylabel("Score")
    plt.ylim(0.0, 1.0)
    plt.title("Overall Class-aware Detection Metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "overall_metric_comparison.png", dpi=300)
    plt.close()


def environment_metadata(args: argparse.Namespace) -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else None

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "ultralytics": ultralytics.__version__,
        "arguments": vars(args),
        "taxonomy": {
            "canonical_classes": CANONICAL_CLASSES,
            "bdd_to_canonical": BDD_TO_CANONICAL,
            "model_to_canonical": MODEL_TO_CANONICAL,
        },
    }


def main() -> None:
    args = parse_args()

    run_name = "smoke_test" if args.smoke_test else "full_road200"
    run_dir = OUTPUT_ROOT / run_name
    figures_dir = run_dir / "figures"
    run_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("\n[1/5] Reading manifest and validating source pairs...")
    samples = read_manifest(args.smoke_test)
    print(f"  Evaluation images: {len(samples)}")

    scenarios_present = Counter(sample["scenario"] for sample in samples)
    print(f"  Scenario counts: {dict(scenarios_present)}")

    print("\n[2/5] Building class-aware COCO ground truth...")
    coco_gt_dict, gt_by_image, audit_rows = build_coco_ground_truth(samples)

    gt_json_path = run_dir / "road200_gt_coco.json"
    write_json(gt_json_path, coco_gt_dict)
    write_csv(run_dir / "road200_image_index.csv", samples)
    write_csv(run_dir / "annotation_audit.csv", audit_rows)
    write_json(run_dir / "environment_and_config.json", environment_metadata(args))

    total_evaluated_gt = len(coco_gt_dict["annotations"])
    total_ignored_gt = sum(
        int(row["ignored_unmapped_bbox_objects"]) for row in audit_rows
    )
    print(
        "  Shared-class GT boxes: "
        f"{total_evaluated_gt}; ignored unmapped bbox objects: {total_ignored_gt}"
    )

    coco_gt = COCO(str(gt_json_path))
    all_image_ids = [sample["coco_image_id"] for sample in samples]
    scenario_to_image_ids: dict[str, list[int]] = defaultdict(list)
    for sample in samples:
        scenario_to_image_ids[sample["scenario"]].append(sample["coco_image_id"])

    all_overall_rows: list[dict[str, Any]] = []
    all_scenario_rows: list[dict[str, Any]] = []
    all_class_rows: list[dict[str, Any]] = []
    all_iou_rows: list[dict[str, Any]] = []
    all_pr_rows: list[dict[str, Any]] = []
    model_metadata: dict[str, Any] = {}
    pr_by_model: dict[str, list[dict[str, float]]] = {}
    iou_by_model: dict[str, list[dict[str, float]]] = {}

    for model_key in args.models:
        print(f"\n[3/5] Running fresh inference for {model_key}...")
        predictions, metadata = run_model_predictions(model_key, samples, args)
        model_metadata[model_key] = metadata

        prediction_path = run_dir / f"predictions_{model_key}_coco.json"
        write_json(prediction_path, predictions)
        print(
            f"  Retained shared-class detections: {len(predictions)}; "
            f"empty-output images: {metadata['empty_prediction_images']}"
        )

        if not predictions:
            raise RuntimeError(
                f"{model_key} produced no compatible predictions. "
                "Check model weights, image paths, and class-name mapping."
            )

        coco_dt = coco_gt.loadRes(str(prediction_path))

        print(f"\n[4/5] Official COCO evaluation for {model_key} (all Road200 images)...")
        overall_eval = run_coco_eval(coco_gt, coco_dt, all_image_ids)
        coco_metrics = metrics_from_coco_eval(overall_eval)
        operating_metrics = operating_point_metrics(
            gt_by_image=gt_by_image,
            predictions=predictions,
            image_ids=all_image_ids,
            score_threshold=args.operating_conf,
            iou_threshold=0.50,
        )

        overall_row = {
            "model": model_key,
            "evaluated_images": len(all_image_ids),
            **coco_metrics,
            **operating_metrics,
            "operating_confidence_threshold": args.operating_conf,
            "matching_iou_for_operating_metrics": 0.50,
            "mean_seconds_per_image": metadata["mean_seconds_per_image"],
        }
        all_overall_rows.append(overall_row)

        model_class_rows = per_class_ap(overall_eval, coco_gt)
        for row in model_class_rows:
            row["model"] = model_key
        all_class_rows.extend(model_class_rows)

        model_iou_rows = ap_by_iou(overall_eval)
        for row in model_iou_rows:
            row["model"] = model_key
        all_iou_rows.extend(model_iou_rows)
        iou_by_model[model_key] = ap_by_iou(overall_eval)

        model_pr_rows = macro_pr_at_iou50(overall_eval)
        for row in model_pr_rows:
            row["model"] = model_key
        all_pr_rows.extend(model_pr_rows)
        pr_by_model[model_key] = macro_pr_at_iou50(overall_eval)

        print(f"\n[5/5] Scenario-level COCO evaluation for {model_key}...")
        for scenario_name in EXPECTED_SCENARIOS:
            image_ids = scenario_to_image_ids.get(scenario_name, [])
            if not image_ids:
                continue

            scenario_eval = run_coco_eval(coco_gt, coco_dt, image_ids)
            scenario_coco_metrics = metrics_from_coco_eval(scenario_eval)
            scenario_operating_metrics = operating_point_metrics(
                gt_by_image=gt_by_image,
                predictions=predictions,
                image_ids=image_ids,
                score_threshold=args.operating_conf,
                iou_threshold=0.50,
            )

            all_scenario_rows.append(
                {
                    "model": model_key,
                    "scenario": scenario_name,
                    "evaluated_images": len(image_ids),
                    **scenario_coco_metrics,
                    **scenario_operating_metrics,
                    "operating_confidence_threshold": args.operating_conf,
                    "matching_iou_for_operating_metrics": 0.50,
                }
            )

    write_csv(run_dir / "metrics_overall.csv", all_overall_rows)
    write_csv(run_dir / "metrics_by_scenario.csv", all_scenario_rows)
    write_csv(run_dir / "metrics_by_class.csv", all_class_rows)
    write_csv(run_dir / "metrics_by_iou.csv", all_iou_rows)
    write_csv(run_dir / "macro_pr_iou50.csv", all_pr_rows)
    write_json(run_dir / "model_inference_metadata.json", model_metadata)

    create_figures(
        per_model_pr=pr_by_model,
        per_model_iou=iou_by_model,
        overall_rows=all_overall_rows,
        figures_dir=figures_dir,
    )

    report_lines = [
        "BDD100K Road200 Class-aware COCO Evaluation",
        "=" * 66,
        f"Run type: {run_name}",
        f"Evaluation images: {len(samples)}",
        f"Shared BDD-to-COCO classes: {', '.join(CANONICAL_CLASSES)}",
        f"Shared-class GT boxes: {total_evaluated_gt}",
        f"Ignored unmapped bbox objects: {total_ignored_gt}",
        "",
        "Overall results:",
    ]
    for row in all_overall_rows:
        report_lines.append(
            f"  {row['model']}: "
            f"mAP@[.50:.95]={row['mAP_50_95']:.4f}, "
            f"AP50={row['AP50']:.4f}, "
            f"AP75={row['AP75']:.4f}, "
            f"P@{args.operating_conf:.2f}={row['precision']:.4f}, "
            f"R@{args.operating_conf:.2f}={row['recall']:.4f}, "
            f"F1@{args.operating_conf:.2f}={row['F1']:.4f}"
        )
    report_lines.extend(
        [
            "",
            "Interpretation notes:",
            "- AP/mAP values are official pycocotools COCO bbox metrics.",
            "- P/R/F1 are class-aware one-to-one matching metrics at the stated",
            "  operating confidence threshold and IoU=0.50.",
            "- Unmapped BDD categories are excluded from both ground truth and",
            "  predictions to preserve a fair COCO-pretrained comparison.",
        ]
    )
    (run_dir / "README_results.txt").write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("\nCompleted successfully.")
    print(f"Results directory: {run_dir}")
    print("\nPaper-facing files to inspect first:")
    print(f"  {run_dir / 'metrics_overall.csv'}")
    print(f"  {run_dir / 'metrics_by_scenario.csv'}")
    print(f"  {figures_dir / 'macro_pr_iou50.png'}")
    print(f"  {figures_dir / 'ap_by_iou_threshold.png'}")
    print(f"  {figures_dir / 'overall_metric_comparison.png'}")


if __name__ == "__main__":
    main()
