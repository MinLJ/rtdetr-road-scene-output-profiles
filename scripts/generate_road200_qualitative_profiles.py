from __future__ import annotations

"""
Deterministic qualitative visualisation for frozen Road200 RT-DETR-L profiles.

This script does NOT run inference and does NOT change the dataset. It reads:
  - full_road200/road200_gt_coco.json
  - full_road200/predictions_rtdetr_coco.json
  - bdd100k_road200_final/manifest.csv

It produces a fixed four-case comparison, one case per scenario, for:
  1. Raw metric baseline:
     confidence >= 0.001, no external NMS, top-300 per image
  2. Balanced output profile:
     confidence >= 0.25, external class-aware NMS IoU=0.45, top-100
  3. Compact output profile:
     confidence >= 0.35, external class-aware NMS IoU=0.55, top-100

Selection protocol
------------------
For each scenario, the selected image is the deterministic "median-complexity"
case. The script computes, within that scenario:
  - number of shared-class ground-truth boxes,
  - retained balanced-profile detections,
  - retained compact-profile detections.

It then selects the image with the smallest normalized L1 distance from the
three scenario medians. Ties are broken by source image ID. This avoids
hand-picking visually favourable examples while still producing examples that
are typical of each scenario's object and output density.

Outputs
-------
.../full_road200/qualitative_profiles_v1/
  qualitative_profile_comparison_4x3.png
  qualitative_ground_truth_reference_2x2.png
  case_<scenario>.png                   (four individual 1x3 panels)
  qualitative_selection.csv
  qualitative_selection_readme.txt
"""

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# ---------------------------------------------------------------------
# Frozen Road200 paths
# ---------------------------------------------------------------------
BASE = Path(r"D:\MFE204_RoadDetection")
DATASET_ROOT = BASE / "subsets" / "bdd100k_road200_final"
MANIFEST_PATH = DATASET_ROOT / "manifest.csv"
BASELINE_DIR = DATASET_ROOT / "eval_coco_standard" / "full_road200"
GT_JSON = BASELINE_DIR / "road200_gt_coco.json"
PRED_JSON = BASELINE_DIR / "predictions_rtdetr_coco.json"
OUT_DIR = BASELINE_DIR / "qualitative_profiles_v1"

SCENARIO_ORDER = [
    "daytime_normal",
    "night_lowlight",
    "crowded_occluded",
    "small_distant",
]

PROFILE_CONFIGS = {
    "Raw baseline": {
        "confidence_floor": 0.001,
        "external_nms_iou": None,
        "max_detections": 300,
        "short_name": "raw",
    },
    "Balanced profile": {
        "confidence_floor": 0.25,
        "external_nms_iou": 0.45,
        "max_detections": 100,
        "short_name": "balanced",
    },
    "Compact profile": {
        "confidence_floor": 0.35,
        "external_nms_iou": 0.55,
        "max_detections": 100,
        "short_name": "compact",
    },
}


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_manifest() -> dict[str, dict[str, str]]:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")

    with MANIFEST_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    required = {"image_id", "scenario", "image_path", "label_path"}
    found = set(rows[0].keys()) if rows else set()
    missing = required.difference(found)
    if missing:
        raise RuntimeError(f"manifest.csv lacks columns: {sorted(missing)}")

    return {row["image_id"]: row for row in rows}


def xywh_to_xyxy(box: list[float]) -> tuple[float, float, float, float]:
    x, y, width, height = [float(value) for value in box]
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

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def class_aware_nms(
    predictions: list[dict[str, Any]],
    iou_threshold: float,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        grouped[
            (int(prediction["image_id"]), int(prediction["category_id"]))
        ].append(prediction)

    retained: list[dict[str, Any]] = []
    for group_predictions in grouped.values():
        ordered = sorted(
            group_predictions,
            key=lambda prediction: float(prediction["score"]),
            reverse=True,
        )
        selected: list[dict[str, Any]] = []

        for candidate in ordered:
            if all(
                box_iou_xywh(candidate["bbox"], kept["bbox"]) <= iou_threshold
                for kept in selected
            ):
                selected.append(candidate)

        retained.extend(selected)

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
            sorted(
                image_predictions,
                key=lambda prediction: float(prediction["score"]),
                reverse=True,
            )[:max_detections]
        )
    return retained


def apply_profile(
    raw_predictions: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    score_floor = float(config["confidence_floor"])
    nms_iou = config["external_nms_iou"]
    max_detections = int(config["max_detections"])

    retained = [
        prediction
        for prediction in raw_predictions
        if float(prediction["score"]) >= score_floor
    ]
    if nms_iou is not None:
        retained = class_aware_nms(retained, float(nms_iou))
    return top_k_per_image(retained, max_detections)


def by_image(
    records: list[dict[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[int(record["image_id"])].append(record)
    return dict(grouped)


def robust_scale(values: list[float]) -> float:
    """
    Small helper for deterministic median-complexity selection.
    A zero IQR can occur in very uniform scenario groups; fall back to 1.0.
    """
    if not values:
        return 1.0
    q75, q25 = np.percentile(values, [75, 25])
    iqr = float(q75 - q25)
    return iqr if iqr > 0.0 else 1.0


def select_representative_cases(
    images: list[dict[str, Any]],
    gt_by_image: dict[int, list[dict[str, Any]]],
    balanced_by_image: dict[int, list[dict[str, Any]]],
    compact_by_image: dict[int, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []

    for scenario in SCENARIO_ORDER:
        candidates = [
            image for image in images if str(image.get("scenario")) == scenario
        ]
        if not candidates:
            raise RuntimeError(f"No images found for scenario: {scenario}")

        gt_counts = [len(gt_by_image.get(int(image["id"]), [])) for image in candidates]
        balanced_counts = [
            len(balanced_by_image.get(int(image["id"]), []))
            for image in candidates
        ]
        compact_counts = [
            len(compact_by_image.get(int(image["id"]), []))
            for image in candidates
        ]

        med_gt = float(np.median(gt_counts))
        med_balanced = float(np.median(balanced_counts))
        med_compact = float(np.median(compact_counts))

        scale_gt = robust_scale(gt_counts)
        scale_balanced = robust_scale(balanced_counts)
        scale_compact = robust_scale(compact_counts)

        scored_candidates: list[dict[str, Any]] = []
        for image in candidates:
            image_id = int(image["id"])
            gt_count = len(gt_by_image.get(image_id, []))
            balanced_count = len(balanced_by_image.get(image_id, []))
            compact_count = len(compact_by_image.get(image_id, []))

            distance = (
                abs(gt_count - med_gt) / scale_gt
                + abs(balanced_count - med_balanced) / scale_balanced
                + abs(compact_count - med_compact) / scale_compact
            )

            scored_candidates.append(
                {
                    "image": image,
                    "selection_score": float(distance),
                    "gt_count": gt_count,
                    "balanced_count": balanced_count,
                    "compact_count": compact_count,
                    "median_gt_count": med_gt,
                    "median_balanced_count": med_balanced,
                    "median_compact_count": med_compact,
                }
            )

        # source_image_id is present in the frozen COCO GT JSON; this gives a
        # stable, independent tie-breaker rather than relying on file ordering.
        scored_candidates.sort(
            key=lambda item: (
                item["selection_score"],
                str(item["image"].get("source_image_id", "")),
            )
        )
        selected.append(scored_candidates[0])

    return selected


def category_lookup(gt_data: dict[str, Any]) -> dict[int, str]:
    return {
        int(category["id"]): str(category["name"])
        for category in gt_data["categories"]
    }


def default_class_color(category_id: int) -> str:
    """
    Uses Matplotlib's default color cycle rather than a custom palette.
    """
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return colors[(category_id - 1) % len(colors)]


def draw_box(
    ax: Any,
    bbox: list[float],
    category_id: int,
    label: str | None,
    linewidth: float,
    alpha: float,
    linestyle: str = "-",
) -> None:
    x, y, width, height = [float(value) for value in bbox]
    rectangle = patches.Rectangle(
        (x, y),
        width,
        height,
        fill=False,
        linewidth=linewidth,
        alpha=alpha,
        linestyle=linestyle,
        edgecolor=default_class_color(category_id),
    )
    ax.add_patch(rectangle)

    if label:
        ax.text(
            x,
            max(0.0, y - 2.0),
            label,
            fontsize=5.2,
            va="bottom",
            ha="left",
            color="white",
            bbox={
                "facecolor": default_class_color(category_id),
                "alpha": min(0.9, alpha + 0.1),
                "pad": 0.8,
                "edgecolor": "none",
            },
        )


def display_image(ax: Any, image_path: Path) -> None:
    with Image.open(image_path) as image:
        ax.imshow(image.convert("RGB"))
    ax.set_axis_off()


def draw_prediction_panel(
    ax: Any,
    image_path: Path,
    predictions: list[dict[str, Any]],
    class_names: dict[int, str],
    panel_title: str,
    show_prediction_labels: bool,
) -> None:
    display_image(ax, image_path)

    # A raw panel can contain >200 predictions. It intentionally uses faint,
    # unlabeled boxes to make density visible without obscuring every pixel.
    raw_style = len(predictions) > 50
    for prediction in sorted(
        predictions,
        key=lambda record: float(record["score"]),
    ):
        category_id = int(prediction["category_id"])
        score = float(prediction["score"])
        label = None
        if show_prediction_labels and not raw_style:
            label = f"{class_names[category_id]} {score:.2f}"

        draw_box(
            ax=ax,
            bbox=prediction["bbox"],
            category_id=category_id,
            label=label,
            linewidth=0.35 if raw_style else 1.10,
            alpha=0.16 if raw_style else 0.88,
        )

    ax.set_title(
        f"{panel_title}\n{len(predictions)} retained detections",
        fontsize=10,
        pad=6,
    )


def draw_gt_panel(
    ax: Any,
    image_path: Path,
    ground_truths: list[dict[str, Any]],
    class_names: dict[int, str],
    scenario_label: str,
    source_image_id: str,
) -> None:
    display_image(ax, image_path)
    for annotation in ground_truths:
        category_id = int(annotation["category_id"])
        draw_box(
            ax=ax,
            bbox=annotation["bbox"],
            category_id=category_id,
            label=class_names[category_id],
            linewidth=1.15,
            alpha=0.92,
            linestyle="--",
        )
    ax.set_title(
        f"{scenario_label}\nGT: {len(ground_truths)} | {source_image_id}",
        fontsize=10,
        pad=6,
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_main_comparison(
    selected_cases: list[dict[str, Any]],
    manifest_by_image: dict[str, dict[str, str]],
    gt_by_image: dict[int, list[dict[str, Any]]],
    raw_by_image: dict[int, list[dict[str, Any]]],
    balanced_by_image: dict[int, list[dict[str, Any]]],
    compact_by_image: dict[int, list[dict[str, Any]]],
    class_names: dict[int, str],
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(
        nrows=4,
        ncols=3,
        figsize=(18.0, 18.2),
        constrained_layout=True,
    )

    profile_columns = [
        ("Raw baseline", raw_by_image, False),
        ("Balanced profile", balanced_by_image, True),
        ("Compact profile", compact_by_image, True),
    ]

    for row_index, selection in enumerate(selected_cases):
        image_metadata = selection["image"]
        image_id = int(image_metadata["id"])
        source_image_id = str(image_metadata["source_image_id"])
        scenario = str(image_metadata["scenario"])
        image_path = Path(manifest_by_image[source_image_id]["image_path"])

        for col_index, (name, prediction_map, show_labels) in enumerate(profile_columns):
            draw_prediction_panel(
                ax=axes[row_index, col_index],
                image_path=image_path,
                predictions=prediction_map.get(image_id, []),
                class_names=class_names,
                panel_title=(
                    f"{name} | {scenario.replace('_', ' ')}"
                    if row_index == 0
                    else name
                ),
                show_prediction_labels=show_labels,
            )

        axes[row_index, 0].text(
            -0.05,
            0.5,
            scenario.replace("_", "\n"),
            transform=axes[row_index, 0].transAxes,
            fontsize=11,
            fontweight="bold",
            va="center",
            ha="right",
            rotation=90,
        )

    figure.suptitle(
        "RT-DETR-L Output Profiles on Deterministically Selected Road200 Cases",
        fontsize=16,
    )
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def make_ground_truth_reference(
    selected_cases: list[dict[str, Any]],
    manifest_by_image: dict[str, dict[str, str]],
    gt_by_image: dict[int, list[dict[str, Any]]],
    class_names: dict[int, str],
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(13.0, 8.2),
        constrained_layout=True,
    )

    for axis, selection in zip(axes.flat, selected_cases):
        image_metadata = selection["image"]
        image_id = int(image_metadata["id"])
        source_image_id = str(image_metadata["source_image_id"])
        scenario = str(image_metadata["scenario"])
        image_path = Path(manifest_by_image[source_image_id]["image_path"])

        draw_gt_panel(
            ax=axis,
            image_path=image_path,
            ground_truths=gt_by_image.get(image_id, []),
            class_names=class_names,
            scenario_label=scenario.replace("_", " "),
            source_image_id=source_image_id,
        )

    figure.suptitle(
        "Ground-truth Reference for the Four Fixed Qualitative Cases",
        fontsize=15,
    )
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def make_individual_case_panel(
    selection: dict[str, Any],
    manifest_by_image: dict[str, dict[str, str]],
    raw_by_image: dict[int, list[dict[str, Any]]],
    balanced_by_image: dict[int, list[dict[str, Any]]],
    compact_by_image: dict[int, list[dict[str, Any]]],
    class_names: dict[int, str],
    output_path: Path,
) -> None:
    image_metadata = selection["image"]
    image_id = int(image_metadata["id"])
    source_image_id = str(image_metadata["source_image_id"])
    scenario = str(image_metadata["scenario"])
    image_path = Path(manifest_by_image[source_image_id]["image_path"])

    figure, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(17.0, 4.8),
        constrained_layout=True,
    )

    profile_columns = [
        ("Raw baseline", raw_by_image, False),
        ("Balanced profile", balanced_by_image, True),
        ("Compact profile", compact_by_image, True),
    ]
    for axis, (name, prediction_map, labels) in zip(axes, profile_columns):
        draw_prediction_panel(
            ax=axis,
            image_path=image_path,
            predictions=prediction_map.get(image_id, []),
            class_names=class_names,
            panel_title=f"{name} | {scenario.replace('_', ' ')}",
            show_prediction_labels=labels,
        )

    figure.suptitle(
        f"Road200 qualitative case: {scenario.replace('_', ' ')} | "
        f"{source_image_id}",
        fontsize=14,
    )
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    for required_path in [MANIFEST_PATH, GT_JSON, PRED_JSON]:
        if not required_path.exists():
            raise FileNotFoundError(f"Required file not found: {required_path}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/4] Loading frozen Road200 inputs...")
    gt_data = read_json(GT_JSON)
    raw_predictions = read_json(PRED_JSON)
    manifest_by_image = read_manifest()

    images = gt_data["images"]
    gt_by_image = by_image(gt_data["annotations"])
    class_names = category_lookup(gt_data)

    print(f"  Images: {len(images)}")
    print(f"  Raw RT-DETR-L predictions: {len(raw_predictions)}")

    print("\n[2/4] Applying fixed output profiles offline...")
    processed_profiles: dict[str, dict[int, list[dict[str, Any]]]] = {}
    for profile_name, config in PROFILE_CONFIGS.items():
        retained = apply_profile(raw_predictions, config)
        processed_profiles[profile_name] = by_image(retained)
        print(f"  {profile_name}: {len(retained)} retained predictions")

    print("\n[3/4] Selecting one median-complexity case per scenario...")
    selected_cases = select_representative_cases(
        images=images,
        gt_by_image=gt_by_image,
        balanced_by_image=processed_profiles["Balanced profile"],
        compact_by_image=processed_profiles["Compact profile"],
    )

    selection_rows: list[dict[str, Any]] = []
    for selection in selected_cases:
        image = selection["image"]
        image_id = int(image["id"])
        source_image_id = str(image["source_image_id"])
        row = {
            "scenario": image["scenario"],
            "coco_image_id": image_id,
            "source_image_id": source_image_id,
            "image_path": manifest_by_image[source_image_id]["image_path"],
            "selection_score": selection["selection_score"],
            "shared_class_gt_boxes": selection["gt_count"],
            "raw_retained_detections": len(
                processed_profiles["Raw baseline"].get(image_id, [])
            ),
            "balanced_retained_detections": selection["balanced_count"],
            "compact_retained_detections": selection["compact_count"],
            "scenario_median_gt_boxes": selection["median_gt_count"],
            "scenario_median_balanced_detections": selection["median_balanced_count"],
            "scenario_median_compact_detections": selection["median_compact_count"],
        }
        selection_rows.append(row)
        print(
            f"  {row['scenario']}: {row['source_image_id']} | "
            f"GT={row['shared_class_gt_boxes']} | "
            f"raw/balanced/compact="
            f"{row['raw_retained_detections']}/"
            f"{row['balanced_retained_detections']}/"
            f"{row['compact_retained_detections']}"
        )

    write_csv(OUT_DIR / "qualitative_selection.csv", selection_rows)

    readme = """Road200 RT-DETR-L qualitative profile selection
================================================

This directory was generated from frozen full_road200 prediction JSON.
No model inference was run and no image was manually selected.

Profiles
--------
Raw baseline:
  confidence >= 0.001; no external NMS; max_det=300
Balanced profile:
  confidence >= 0.25; external class-aware NMS IoU=0.45; max_det=100
Compact profile:
  confidence >= 0.35; external class-aware NMS IoU=0.55; max_det=100

Selection rule
--------------
For each scenario, the selected image minimizes the normalized L1 distance to
that scenario's median:
  (a) number of shared-class GT boxes,
  (b) number of balanced-profile retained detections, and
  (c) number of compact-profile retained detections.

Ties are broken by source image ID. See qualitative_selection.csv.
"""
    (OUT_DIR / "qualitative_selection_readme.txt").write_text(
        readme,
        encoding="utf-8",
    )

    print("\n[4/4] Rendering qualitative comparison figures...")
    make_main_comparison(
        selected_cases=selected_cases,
        manifest_by_image=manifest_by_image,
        gt_by_image=gt_by_image,
        raw_by_image=processed_profiles["Raw baseline"],
        balanced_by_image=processed_profiles["Balanced profile"],
        compact_by_image=processed_profiles["Compact profile"],
        class_names=class_names,
        output_path=OUT_DIR / "qualitative_profile_comparison_4x3.png",
    )
    make_ground_truth_reference(
        selected_cases=selected_cases,
        manifest_by_image=manifest_by_image,
        gt_by_image=gt_by_image,
        class_names=class_names,
        output_path=OUT_DIR / "qualitative_ground_truth_reference_2x2.png",
    )

    for selection in selected_cases:
        scenario = str(selection["image"]["scenario"])
        make_individual_case_panel(
            selection=selection,
            manifest_by_image=manifest_by_image,
            raw_by_image=processed_profiles["Raw baseline"],
            balanced_by_image=processed_profiles["Balanced profile"],
            compact_by_image=processed_profiles["Compact profile"],
            class_names=class_names,
            output_path=OUT_DIR / f"case_{scenario}.png",
        )

    print("\nCompleted successfully.")
    print(f"Output directory: {OUT_DIR}")
    print("Inspect these first:")
    print(f"  {OUT_DIR / 'qualitative_selection.csv'}")
    print(f"  {OUT_DIR / 'qualitative_profile_comparison_4x3.png'}")
    print(f"  {OUT_DIR / 'qualitative_ground_truth_reference_2x2.png'}")


if __name__ == "__main__":
    main()
