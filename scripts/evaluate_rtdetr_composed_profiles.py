from __future__ import annotations

"""
Confirmation study for the composed RT-DETR-L post-processing pipeline.

Purpose
-------
The component-wise ablation established the isolated effects of confidence
filtering, top-k output capping, and external class-aware NMS. This script tests
the actual composed output profiles used for deployment/visualisation:

  confidence floor in {0.25, 0.35}
  external class-aware NMS IoU in {none, 0.45, 0.55, 0.65}
  fixed per-image maximum detections = 100

The frozen raw baseline (.001 / no NMS / top-300) is included as a reference.
All measurements use the same frozen Road200 COCO ground truth and the same
frozen RT-DETR-L prediction JSON. No model inference is run.
"""

import argparse
import contextlib
import csv
import io
from pathlib import Path
from typing import Any

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# This sibling script is the previously validated single-factor ablation engine.
from ablate_rtdetr_road200_postprocess import (
    BASELINE_DIR,
    CATEGORY_IDS,
    EXPECTED_IMAGE_COUNT,
    GT_JSON,
    OUTPUT_DIR_NAME,
    PREDICTIONS_JSON,
    apply_strategy,
    build_gt_by_image,
    operating_metrics,
    scenario_image_ids,
    sha256_file,
    write_csv,
    write_json,
)

PROFILE_GRID = [
    {"profile": "raw_baseline", "confidence_floor": 0.001, "nms_iou": None, "max_det": 300},
    {"profile": "conf025_no_nms", "confidence_floor": 0.25, "nms_iou": None, "max_det": 100},
    {"profile": "conf025_nms045", "confidence_floor": 0.25, "nms_iou": 0.45, "max_det": 100},
    {"profile": "conf025_nms055", "confidence_floor": 0.25, "nms_iou": 0.55, "max_det": 100},
    {"profile": "conf025_nms065", "confidence_floor": 0.25, "nms_iou": 0.65, "max_det": 100},
    {"profile": "conf035_no_nms", "confidence_floor": 0.35, "nms_iou": None, "max_det": 100},
    {"profile": "conf035_nms045", "confidence_floor": 0.35, "nms_iou": 0.45, "max_det": 100},
    {"profile": "conf035_nms055", "confidence_floor": 0.35, "nms_iou": 0.55, "max_det": 100},
    {"profile": "conf035_nms065", "confidence_floor": 0.35, "nms_iou": 0.65, "max_det": 100},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate composed RT-DETR-L post-processing profiles offline."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting profile-study outputs.",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        import json
        return json.load(handle)


def run_coco(
    coco_gt: COCO,
    predictions: list[dict[str, Any]],
    image_ids: list[int],
) -> dict[str, float]:
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
    with contextlib.redirect_stdout(io.StringIO()):
        evaluator.summarize()

    return {
        "mAP_50_95": float(evaluator.stats[0]),
        "AP50": float(evaluator.stats[1]),
        "AP75": float(evaluator.stats[2]),
        "AR1": float(evaluator.stats[6]),
        "AR10": float(evaluator.stats[7]),
        "AR100": float(evaluator.stats[8]),
    }


def nms_label(value: float | None) -> str:
    return "none" if value is None else f"{value:.2f}"


def evaluate_profile(
    profile: dict[str, Any],
    raw_predictions: list[dict[str, Any]],
    coco_gt: COCO,
    gt_by_image: dict[int, list[dict[str, Any]]],
    image_ids: list[int],
    scenario_ids: dict[str, list[int]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    confidence_floor = float(profile["confidence_floor"])
    nms_iou = profile["nms_iou"]
    max_det = int(profile["max_det"])

    processed = apply_strategy(
        raw_predictions=raw_predictions,
        confidence_floor=confidence_floor,
        max_detections=max_det,
        nms_iou=nms_iou,
    )

    overall = {
        "profile": profile["profile"],
        "confidence_floor": confidence_floor,
        "external_class_aware_nms_iou": nms_label(nms_iou),
        "per_image_max_detections": max_det,
        "retained_predictions": len(processed),
        "avg_retained_detections_per_image": len(processed) / len(image_ids),
        **run_coco(coco_gt, processed, image_ids),
        **operating_metrics(
            gt_by_image,
            processed,
            image_ids,
            operating_confidence=confidence_floor,
            iou_threshold=0.50,
        ),
        "operating_confidence_for_PRF1": confidence_floor,
        "matching_iou_for_PRF1": 0.50,
    }

    scenario_rows: list[dict[str, Any]] = []
    for scenario, ids in sorted(scenario_ids.items()):
        local_processed_count = sum(
            1 for pred in processed if int(pred["image_id"]) in set(ids)
        )
        row = {
            "profile": profile["profile"],
            "scenario": scenario,
            "evaluated_images": len(ids),
            "confidence_floor": confidence_floor,
            "external_class_aware_nms_iou": nms_label(nms_iou),
            "per_image_max_detections": max_det,
            "retained_predictions": local_processed_count,
            "avg_retained_detections_per_image": local_processed_count / len(ids),
            **run_coco(coco_gt, processed, ids),
            **operating_metrics(
                gt_by_image,
                processed,
                ids,
                operating_confidence=confidence_floor,
                iou_threshold=0.50,
            ),
            "operating_confidence_for_PRF1": confidence_floor,
            "matching_iou_for_PRF1": 0.50,
        }
        scenario_rows.append(row)

    return overall, scenario_rows


def make_figures(rows: list[dict[str, Any]], figures_dir: Path) -> None:
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)

    # Each point is annotated: mAP/F1 trade-off, marker size encodes output density.
    plt.figure(figsize=(8.5, 6.2))
    for row in rows:
        size = 40 + 1.2 * min(float(row["avg_retained_detections_per_image"]), 260)
        plt.scatter(
            float(row["mAP_50_95"]),
            float(row["F1"]),
            s=size,
            label=row["profile"],
        )
        plt.annotate(
            row["profile"].replace("_", "\n"),
            (float(row["mAP_50_95"]), float(row["F1"])),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )
    plt.xlabel("COCO mAP@[.50:.95]")
    plt.ylabel("F1 at profile confidence threshold")
    plt.title("Accuracy--Usability Trade-off of RT-DETR-L Output Profiles")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "profile_tradeoff_map_f1.png", dpi=300)
    plt.close()

    # Output density vs F1, making the compactness trade-off visible.
    plt.figure(figsize=(8.5, 5.6))
    for row in rows:
        plt.scatter(
            float(row["avg_retained_detections_per_image"]),
            float(row["F1"]),
            s=70,
        )
        plt.annotate(
            row["profile"].replace("_", "\n"),
            (
                float(row["avg_retained_detections_per_image"]),
                float(row["F1"]),
            ),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )
    plt.xlabel("Average retained detections per image")
    plt.ylabel("F1 at profile confidence threshold")
    plt.title("Output Density versus Detection Balance")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "profile_output_density_f1.png", dpi=300)
    plt.close()


def main() -> None:
    args = parse_args()

    if not GT_JSON.exists() or not PREDICTIONS_JSON.exists():
        raise FileNotFoundError(
            "Frozen full_road200 inputs are missing. "
            "Run the standard evaluator before this profile study."
        )

    output_dir = BASELINE_DIR / "rtdetr_composed_profiles_v1"
    figures_dir = output_dir / "figures"
    summary_path = output_dir / "profile_summary.csv"

    if summary_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{summary_path} already exists. Use --overwrite to regenerate it."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("[1/4] Loading frozen Road200 data...")
    gt_data = read_json(GT_JSON)
    raw_predictions = read_json(PREDICTIONS_JSON)
    coco_gt = COCO(str(GT_JSON))

    image_ids = sorted(int(image["id"]) for image in gt_data["images"])
    if len(image_ids) != EXPECTED_IMAGE_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_IMAGE_COUNT} images, found {len(image_ids)}."
        )

    gt_by_image = build_gt_by_image(gt_data)
    scenarios = scenario_image_ids(gt_data)

    print(f"  Images: {len(image_ids)}")
    print(f"  Frozen raw predictions: {len(raw_predictions)}")
    print(f"  Profiles: {len(PROFILE_GRID)}")

    print("\n[2/4] Evaluating composed post-processing profiles...")
    summary_rows: list[dict[str, Any]] = []
    scenario_rows: list[dict[str, Any]] = []

    for profile in PROFILE_GRID:
        print(
            f"  {profile['profile']}: "
            f"conf={profile['confidence_floor']}, "
            f"nms={nms_label(profile['nms_iou'])}, "
            f"max_det={profile['max_det']}"
        )
        overall, per_scenario = evaluate_profile(
            profile,
            raw_predictions,
            coco_gt,
            gt_by_image,
            image_ids,
            scenarios,
        )
        summary_rows.append(overall)
        scenario_rows.extend(per_scenario)

    print("\n[3/4] Writing results...")
    write_csv(summary_path, summary_rows)
    write_csv(output_dir / "profile_by_scenario.csv", scenario_rows)
    write_json(
        output_dir / "profile_config.json",
        {
            "study_name": "RT-DETR-L Road200 composed post-processing profile study v1",
            "source_ground_truth": str(GT_JSON),
            "source_predictions": str(PREDICTIONS_JSON),
            "ground_truth_sha256": sha256_file(GT_JSON),
            "predictions_sha256": sha256_file(PREDICTIONS_JSON),
            "profiles": PROFILE_GRID,
            "postprocess_order": [
                "confidence_floor",
                "optional_external_class_aware_nms",
                "per_image_global_top_k",
            ],
            "note": (
                "External class-aware NMS is an experimental output-layer "
                "intervention, not a native RT-DETR requirement."
            ),
        },
    )
    make_figures(summary_rows, figures_dir)

    print("\n[4/4] Headline configurations:")
    for metric, label in [
        ("mAP_50_95", "mAP@[.50:.95]"),
        ("F1", "F1"),
        ("precision", "Precision"),
        ("recall", "Recall"),
    ]:
        best = max(summary_rows, key=lambda row: float(row[metric]))
        print(f"  Best {label}: {best['profile']} = {float(best[metric]):.4f}")

    print("\nDone.")
    print(f"Output: {output_dir}")
    print(f"  {summary_path}")
    print(f"  {output_dir / 'profile_by_scenario.csv'}")
    print(f"  {figures_dir}")


if __name__ == "__main__":
    main()
