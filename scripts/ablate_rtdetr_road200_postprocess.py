from __future__ import annotations

"""
Offline post-processing ablation for frozen RT-DETR-L Road200 predictions.

This script NEVER reruns RT-DETR-L inference and NEVER changes Road200.
It applies explicit output-layer strategies to the frozen COCO-format
prediction JSON produced by evaluate_road200_coco_standard.py.

Ablation families
-----------------
1) Confidence floor:
   0.001, 0.05, 0.10, 0.20, 0.25, 0.35, 0.50
2) Per-image max detections:
   top-50, top-100, top-300
3) External class-aware NMS:
   none, IoU=0.45, 0.55, 0.65

Metrics
-------
- Official pycocotools bbox mAP@[.50:.95], AP50, AP75
- Class-aware P/R/F1 at a declared operating score threshold and IoU=0.50
- Number of retained detections per image

Important methodological note
-----------------------------
RT-DETR is evaluated here with an *external class-aware NMS* option. This is
a deliberately added output-layer strategy for study; it must not be described
as a native RT-DETR requirement in the paper.
"""

import argparse
import contextlib
import csv
import hashlib
import io
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except ImportError as exc:
    raise SystemExit(
        "pycocotools is required. Install it in road_det_bdd with:\n"
        "  python -m pip install pycocotools"
    ) from exc


# ---------------------------------------------------------------------
# Frozen Road200 baseline files
# ---------------------------------------------------------------------
BASE = Path(r"D:\MFE204_RoadDetection")
DATASET_ROOT = BASE / "subsets" / "bdd100k_road200_final"
BASELINE_DIR = DATASET_ROOT / "eval_coco_standard" / "full_road200"

GT_JSON = BASELINE_DIR / "road200_gt_coco.json"
PREDICTIONS_JSON = BASELINE_DIR / "predictions_rtdetr_coco.json"
BASELINE_METRICS_CSV = BASELINE_DIR / "metrics_overall.csv"

OUTPUT_DIR_NAME = "ablation_rtdetr_postprocess_v1"
CATEGORY_IDS = list(range(1, 9))  # Fixed Road200 taxonomy from the baseline evaluator.
EXPECTED_IMAGE_COUNT = 200

CONFIDENCE_SETTINGS = [0.001, 0.05, 0.10, 0.20, 0.25, 0.35, 0.50]
MAX_DET_SETTINGS = [50, 100, 300]
NMS_SETTINGS: list[float | None] = [None, 0.45, 0.55, 0.65]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline RT-DETR-L post-processing ablation on frozen Road200 predictions."
    )
    parser.add_argument(
        "--families",
        nargs="+",
        choices=["confidence", "maxdet", "nms"],
        default=["confidence", "maxdet", "nms"],
        help="Ablation families to execute. Default: all three.",
    )
    parser.add_argument(
        "--operating-conf",
        type=float,
        default=0.25,
        help=(
            "Fixed score threshold for P/R/F1 in max-det and NMS sweeps. "
            "Confidence sweep uses its own floor as the operating threshold. "
            "Default: 0.25"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting CSV/JSON/figure outputs in the ablation directory.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def xywh_to_xyxy(box: list[float]) -> tuple[float, float, float, float]:
    x, y, width, height = [float(value) for value in box]
    return x, y, x + width, y + height


def iou_xywh(first: list[float], second: list[float]) -> float:
    ax1, ay1, ax2, ay2 = xywh_to_xyxy(first)
    bx1, by1, bx2, by2 = xywh_to_xyxy(second)

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if intersection <= 0.0:
        return 0.0

    area_first = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_second = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_first + area_second - intersection
    return intersection / union if union > 0.0 else 0.0


def class_aware_nms(
    predictions: list[dict[str, Any]],
    iou_threshold: float,
) -> list[dict[str, Any]]:
    """
    Greedy class-aware NMS.

    Predictions from different classes are intentionally never compared.
    Within each (image_id, category_id) group, a lower-scored prediction is
    suppressed only when its IoU with a previously kept prediction is strictly
    greater than the declared threshold.
    """
    by_group: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        by_group[(int(prediction["image_id"]), int(prediction["category_id"]))].append(
            prediction
        )

    retained: list[dict[str, Any]] = []

    for group_predictions in by_group.values():
        ordered = sorted(group_predictions, key=lambda item: float(item["score"]), reverse=True)
        kept: list[dict[str, Any]] = []

        for candidate in ordered:
            if all(
                iou_xywh(candidate["bbox"], selected["bbox"]) <= iou_threshold
                for selected in kept
            ):
                kept.append(candidate)

        retained.extend(kept)

    return retained


def top_k_per_image(
    predictions: list[dict[str, Any]],
    max_detections: int,
) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        grouped[int(prediction["image_id"])].append(prediction)

    retained: list[dict[str, Any]] = []
    for image_predictions in grouped.values():
        retained.extend(
            sorted(image_predictions, key=lambda item: float(item["score"]), reverse=True)[
                :max_detections
            ]
        )
    return retained


def apply_strategy(
    raw_predictions: list[dict[str, Any]],
    confidence_floor: float,
    max_detections: int,
    nms_iou: float | None,
) -> list[dict[str, Any]]:
    """
    Fixed strategy order:
    1. Remove predictions below the confidence floor.
    2. Optionally apply external class-aware NMS.
    3. Retain at most top-k predictions per image globally by score.
    """
    filtered = [
        prediction
        for prediction in raw_predictions
        if float(prediction["score"]) >= confidence_floor
    ]

    if nms_iou is not None:
        filtered = class_aware_nms(filtered, nms_iou)

    return top_k_per_image(filtered, max_detections)


def coerce_coco_stats(evaluator: COCOeval) -> dict[str, float]:
    return {
        "mAP_50_95": float(evaluator.stats[0]),
        "AP50": float(evaluator.stats[1]),
        "AP75": float(evaluator.stats[2]),
        "AR1": float(evaluator.stats[6]),
        "AR10": float(evaluator.stats[7]),
        "AR100": float(evaluator.stats[8]),
    }


def run_coco_evaluation(
    coco_gt: COCO,
    predictions: list[dict[str, Any]],
    image_ids: list[int],
) -> dict[str, float]:
    # pycocotools expects a result object; all selected strategies in this
    # experiment retain predictions, but fail loudly if a future setting does not.
    if not predictions:
        return {
            "mAP_50_95": 0.0,
            "AP50": 0.0,
            "AP75": 0.0,
            "AR1": 0.0,
            "AR10": 0.0,
            "AR100": 0.0,
        }

    coco_dt = coco_gt.loadRes(predictions)
    evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
    evaluator.params.imgIds = sorted(image_ids)
    evaluator.params.catIds = CATEGORY_IDS
    evaluator.params.iouThrs = np.linspace(0.50, 0.95, 10)
    evaluator.params.maxDets = [1, 10, 100]
    evaluator.evaluate()
    evaluator.accumulate()

    # Need evaluator.summarize() to populate evaluator.stats, but do not flood
    # the terminal with 50+ COCO summary blocks during the ablation.
    with contextlib.redirect_stdout(io.StringIO()):
        evaluator.summarize()

    return coerce_coco_stats(evaluator)


def build_gt_by_image(coco_gt_data: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    gt_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in coco_gt_data["annotations"]:
        gt_by_image[int(annotation["image_id"])].append(annotation)
    return gt_by_image


def operating_metrics(
    gt_by_image: dict[int, list[dict[str, Any]]],
    predictions: list[dict[str, Any]],
    image_ids: list[int],
    operating_confidence: float,
    iou_threshold: float = 0.50,
) -> dict[str, float | int]:
    """
    One-to-one class-aware matching for P/R/F1 at a declared operating score.
    This is intentionally distinct from AP/mAP.
    """
    predictions_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        if float(prediction["score"]) >= operating_confidence:
            predictions_by_image[int(prediction["image_id"])].append(prediction)

    tp = 0
    fp = 0
    fn = 0

    for image_id in image_ids:
        gt_by_category: dict[int, list[dict[str, Any]]] = defaultdict(list)
        pred_by_category: dict[int, list[dict[str, Any]]] = defaultdict(list)

        for ground_truth in gt_by_image.get(image_id, []):
            gt_by_category[int(ground_truth["category_id"])].append(ground_truth)

        for prediction in predictions_by_image.get(image_id, []):
            pred_by_category[int(prediction["category_id"])].append(prediction)

        for category_id in CATEGORY_IDS:
            category_ground_truths = gt_by_category.get(category_id, [])
            category_predictions = sorted(
                pred_by_category.get(category_id, []),
                key=lambda item: float(item["score"]),
                reverse=True,
            )
            matched_indices: set[int] = set()

            for prediction in category_predictions:
                best_iou = 0.0
                best_index = -1

                for gt_index, ground_truth in enumerate(category_ground_truths):
                    if gt_index in matched_indices:
                        continue
                    current_iou = iou_xywh(prediction["bbox"], ground_truth["bbox"])
                    if current_iou > best_iou:
                        best_iou = current_iou
                        best_index = gt_index

                if best_index >= 0 and best_iou >= iou_threshold:
                    tp += 1
                    matched_indices.add(best_index)
                else:
                    fp += 1

            fn += len(category_ground_truths) - len(matched_indices)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "F1": f1,
    }


def scenario_image_ids(coco_gt_data: dict[str, Any]) -> dict[str, list[int]]:
    by_scenario: dict[str, list[int]] = defaultdict(list)
    for image in coco_gt_data["images"]:
        scenario = image.get("scenario", "unknown")
        by_scenario[str(scenario)].append(int(image["id"]))
    return dict(by_scenario)


def baseline_expected_map() -> float | None:
    if not BASELINE_METRICS_CSV.exists():
        return None

    with BASELINE_METRICS_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("model") == "rtdetr":
                try:
                    return float(row["mAP_50_95"])
                except (KeyError, TypeError, ValueError):
                    return None
    return None


def format_nms(nms_iou: float | None) -> str:
    return "none" if nms_iou is None else f"{nms_iou:.2f}"


def evaluate_configuration(
    *,
    family: str,
    setting: str,
    raw_predictions: list[dict[str, Any]],
    coco_gt: COCO,
    gt_by_image: dict[int, list[dict[str, Any]]],
    image_ids: list[int],
    scenario_ids: dict[str, list[int]],
    confidence_floor: float,
    max_detections: int,
    nms_iou: float | None,
    operating_confidence: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    processed = apply_strategy(
        raw_predictions=raw_predictions,
        confidence_floor=confidence_floor,
        max_detections=max_detections,
        nms_iou=nms_iou,
    )

    coco_metrics = run_coco_evaluation(coco_gt, processed, image_ids)
    op_metrics = operating_metrics(
        gt_by_image=gt_by_image,
        predictions=processed,
        image_ids=image_ids,
        operating_confidence=operating_confidence,
        iou_threshold=0.50,
    )

    average_detections = len(processed) / len(image_ids)

    common = {
        "family": family,
        "setting": setting,
        "confidence_floor": confidence_floor,
        "external_class_aware_nms_iou": format_nms(nms_iou),
        "per_image_max_detections": max_detections,
        "operating_confidence_for_PRF1": operating_confidence,
        "matching_iou_for_PRF1": 0.50,
        "retained_predictions": len(processed),
        "avg_retained_detections_per_image": average_detections,
        **coco_metrics,
        **op_metrics,
    }

    scenario_rows: list[dict[str, Any]] = []
    for scenario, ids in sorted(scenario_ids.items()):
        scenario_coco = run_coco_evaluation(coco_gt, processed, ids)
        scenario_operating = operating_metrics(
            gt_by_image=gt_by_image,
            predictions=processed,
            image_ids=ids,
            operating_confidence=operating_confidence,
            iou_threshold=0.50,
        )
        scenario_rows.append(
            {
                **common,
                "scenario": scenario,
                "evaluated_images": len(ids),
                **scenario_coco,
                **scenario_operating,
            }
        )

    return common, scenario_rows


def save_figures(summary_rows: list[dict[str, Any]], figures_dir: Path) -> None:
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)

    def rows_for(family: str) -> list[dict[str, Any]]:
        return [row for row in summary_rows if row["family"] == family]

    # Confidence sweep: performance trade-offs.
    confidence_rows = sorted(
        rows_for("confidence"), key=lambda row: float(row["confidence_floor"])
    )
    if confidence_rows:
        x = [float(row["confidence_floor"]) for row in confidence_rows]
        plt.figure(figsize=(7.2, 5.0))
        for metric, label in [
            ("mAP_50_95", "mAP@[.50:.95]"),
            ("AP50", "AP50"),
            ("precision", "Precision"),
            ("recall", "Recall"),
            ("F1", "F1"),
        ]:
            plt.plot(x, [float(row[metric]) for row in confidence_rows], marker="o", label=label)
        plt.xlabel("Confidence floor")
        plt.ylabel("Score")
        plt.ylim(0.0, 1.0)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.title("RT-DETR-L Confidence-floor Ablation")
        plt.tight_layout()
        plt.savefig(figures_dir / "confidence_ablation_metrics.png", dpi=300)
        plt.close()

        plt.figure(figsize=(7.2, 4.6))
        plt.plot(
            x,
            [float(row["avg_retained_detections_per_image"]) for row in confidence_rows],
            marker="o",
        )
        plt.xlabel("Confidence floor")
        plt.ylabel("Average retained detections per image")
        plt.grid(True, alpha=0.3)
        plt.title("Output Density Under Confidence Filtering")
        plt.tight_layout()
        plt.savefig(figures_dir / "confidence_ablation_output_density.png", dpi=300)
        plt.close()

    # Max-detection sweep.
    maxdet_rows = sorted(
        rows_for("maxdet"), key=lambda row: int(row["per_image_max_detections"])
    )
    if maxdet_rows:
        x = [int(row["per_image_max_detections"]) for row in maxdet_rows]
        plt.figure(figsize=(7.2, 5.0))
        for metric, label in [
            ("mAP_50_95", "mAP@[.50:.95]"),
            ("AP50", "AP50"),
            ("precision", "Precision@0.25"),
            ("recall", "Recall@0.25"),
            ("F1", "F1@0.25"),
        ]:
            plt.plot(x, [float(row[metric]) for row in maxdet_rows], marker="o", label=label)
        plt.xlabel("Per-image maximum detections")
        plt.ylabel("Score")
        plt.ylim(0.0, 1.0)
        plt.xticks(x)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.title("RT-DETR-L Maximum-detection Ablation")
        plt.tight_layout()
        plt.savefig(figures_dir / "maxdet_ablation_metrics.png", dpi=300)
        plt.close()

    # External NMS sweep.
    nms_rows = rows_for("external_nms")
    if nms_rows:
        labels = [str(row["external_class_aware_nms_iou"]) for row in nms_rows]
        x = np.arange(len(nms_rows))
        metrics = [
            ("mAP_50_95", "mAP@[.50:.95]"),
            ("precision", "Precision@0.25"),
            ("recall", "Recall@0.25"),
            ("F1", "F1@0.25"),
        ]
        width = 0.18

        plt.figure(figsize=(8.2, 5.0))
        for index, (metric, label) in enumerate(metrics):
            offset = (index - (len(metrics) - 1) / 2) * width
            plt.bar(
                x + offset,
                [float(row[metric]) for row in nms_rows],
                width=width,
                label=label,
            )
        plt.xticks(x, labels)
        plt.xlabel("External class-aware NMS IoU threshold")
        plt.ylabel("Score")
        plt.ylim(0.0, 1.0)
        plt.legend()
        plt.title("RT-DETR-L External Class-aware NMS Ablation")
        plt.tight_layout()
        plt.savefig(figures_dir / "external_nms_ablation_metrics.png", dpi=300)
        plt.close()


def main() -> None:
    args = parse_args()

    for required in [GT_JSON, PREDICTIONS_JSON]:
        if not required.exists():
            raise FileNotFoundError(
                f"Required frozen baseline file not found:\n  {required}\n"
                "Run evaluate_road200_coco_standard.py on full Road200 first."
            )

    output_dir = BASELINE_DIR / OUTPUT_DIR_NAME
    figures_dir = output_dir / "figures"

    if (output_dir / "ablation_summary.csv").exists() and not args.overwrite:
        raise FileExistsError(
            f"{output_dir / 'ablation_summary.csv'} already exists.\n"
            "Use --overwrite only when intentionally regenerating the same study."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] Loading frozen Road200 GT and RT-DETR-L predictions...")
    coco_gt_data = read_json(GT_JSON)
    raw_predictions = read_json(PREDICTIONS_JSON)
    if not isinstance(raw_predictions, list):
        raise TypeError("predictions_rtdetr_coco.json must contain a JSON list.")

    coco_gt = COCO(str(GT_JSON))
    image_ids = sorted(int(image["id"]) for image in coco_gt_data["images"])
    if len(image_ids) != EXPECTED_IMAGE_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_IMAGE_COUNT} Road200 images but found {len(image_ids)}."
        )

    gt_by_image = build_gt_by_image(coco_gt_data)
    scenario_ids = scenario_image_ids(coco_gt_data)

    print(f"  Images: {len(image_ids)}")
    print(f"  Raw frozen RT-DETR-L predictions: {len(raw_predictions)}")
    print(f"  Scenarios: { {name: len(ids) for name, ids in sorted(scenario_ids.items())} }")

    print("\n[2/5] Reproducing the frozen no-external-NMS baseline...")
    baseline_row, baseline_scenarios = evaluate_configuration(
        family="baseline_reproduction",
        setting="frozen_raw_predictions",
        raw_predictions=raw_predictions,
        coco_gt=coco_gt,
        gt_by_image=gt_by_image,
        image_ids=image_ids,
        scenario_ids=scenario_ids,
        confidence_floor=0.001,
        max_detections=300,
        nms_iou=None,
        operating_confidence=args.operating_conf,
    )

    expected_map = baseline_expected_map()
    actual_map = float(baseline_row["mAP_50_95"])
    if expected_map is not None:
        delta = abs(actual_map - expected_map)
        print(
            f"  Reproduced mAP@[.50:.95]={actual_map:.6f}; "
            f"frozen baseline={expected_map:.6f}; delta={delta:.8f}"
        )
        if delta > 1e-6:
            raise RuntimeError(
                "Baseline reproduction failed. Do not continue with ablation.\n"
                f"Expected mAP={expected_map:.8f}, got {actual_map:.8f}.\n"
                "Check that GT and prediction JSON files came from the same full_road200 run."
            )
    else:
        print(
            f"  Reproduced mAP@[.50:.95]={actual_map:.6f}. "
            "No baseline metrics CSV was found for cross-checking."
        )

    summary_rows: list[dict[str, Any]] = [baseline_row]
    scenario_rows: list[dict[str, Any]] = baseline_scenarios

    print("\n[3/5] Running offline post-processing ablations...")
    if "confidence" in args.families:
        for floor in CONFIDENCE_SETTINGS:
            print(f"  confidence floor = {floor:.3f}")
            row, rows_by_scenario = evaluate_configuration(
                family="confidence",
                setting=f"conf_{floor:.3f}",
                raw_predictions=raw_predictions,
                coco_gt=coco_gt,
                gt_by_image=gt_by_image,
                image_ids=image_ids,
                scenario_ids=scenario_ids,
                confidence_floor=floor,
                max_detections=300,
                nms_iou=None,
                operating_confidence=floor,
            )
            summary_rows.append(row)
            scenario_rows.extend(rows_by_scenario)

    if "maxdet" in args.families:
        for max_detections in MAX_DET_SETTINGS:
            print(f"  max detections per image = {max_detections}")
            row, rows_by_scenario = evaluate_configuration(
                family="maxdet",
                setting=f"top_{max_detections}",
                raw_predictions=raw_predictions,
                coco_gt=coco_gt,
                gt_by_image=gt_by_image,
                image_ids=image_ids,
                scenario_ids=scenario_ids,
                confidence_floor=0.001,
                max_detections=max_detections,
                nms_iou=None,
                operating_confidence=args.operating_conf,
            )
            summary_rows.append(row)
            scenario_rows.extend(rows_by_scenario)

    if "nms" in args.families:
        for nms_iou in NMS_SETTINGS:
            label = format_nms(nms_iou)
            print(f"  external class-aware NMS IoU = {label}")
            row, rows_by_scenario = evaluate_configuration(
                family="external_nms",
                setting=f"nms_{label}",
                raw_predictions=raw_predictions,
                coco_gt=coco_gt,
                gt_by_image=gt_by_image,
                image_ids=image_ids,
                scenario_ids=scenario_ids,
                confidence_floor=0.001,
                max_detections=300,
                nms_iou=nms_iou,
                operating_confidence=args.operating_conf,
            )
            summary_rows.append(row)
            scenario_rows.extend(rows_by_scenario)

    print("\n[4/5] Writing CSV/JSON outputs and figures...")
    write_csv(output_dir / "ablation_summary.csv", summary_rows)
    write_csv(output_dir / "ablation_by_scenario.csv", scenario_rows)

    config = {
        "study_name": "Road200 RT-DETR-L Offline Post-processing Ablation v1",
        "input_ground_truth": str(GT_JSON),
        "input_predictions": str(PREDICTIONS_JSON),
        "input_ground_truth_sha256": sha256_file(GT_JSON),
        "input_predictions_sha256": sha256_file(PREDICTIONS_JSON),
        "postprocess_order": [
            "score_floor",
            "optional_external_class_aware_nms",
            "global_top_k_per_image",
        ],
        "confidence_settings": CONFIDENCE_SETTINGS,
        "max_detections_settings": MAX_DET_SETTINGS,
        "external_class_aware_nms_iou_settings": [
            "none" if value is None else value for value in NMS_SETTINGS
        ],
        "fixed_operating_confidence_for_maxdet_and_nms": args.operating_conf,
        "prf1_matching_iou": 0.50,
        "coco_iou_thresholds": [round(float(value), 2) for value in np.linspace(0.50, 0.95, 10)],
        "coco_max_dets": [1, 10, 100],
        "note": (
            "External class-aware NMS is an added post-processing intervention; "
            "it is not presented as native RT-DETR behavior."
        ),
    }
    write_json(output_dir / "ablation_config.json", config)
    save_figures(summary_rows, figures_dir)

    print("\n[5/5] Reporting best settings by each headline metric...")
    for metric, label in [
        ("mAP_50_95", "mAP@[.50:.95]"),
        ("F1", "F1"),
        ("precision", "Precision"),
        ("recall", "Recall"),
    ]:
        best = max(
            [row for row in summary_rows if row["family"] != "baseline_reproduction"],
            key=lambda row: float(row[metric]),
        )
        print(
            f"  Best {label}: {best['family']} / {best['setting']} "
            f"= {float(best[metric]):.4f}"
        )

    print("\nCompleted successfully.")
    print(f"Output directory: {output_dir}")
    print("Inspect these files first:")
    print(f"  {output_dir / 'ablation_summary.csv'}")
    print(f"  {output_dir / 'ablation_by_scenario.csv'}")
    print(f"  {output_dir / 'ablation_config.json'}")
    print(f"  {figures_dir}")


if __name__ == "__main__":
    main()
