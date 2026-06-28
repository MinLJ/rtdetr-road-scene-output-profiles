from __future__ import annotations

"""
Create the two figure assets required by Methodology Section 3.

Output files:
  D:\MFE204_RoadDetection\repo\figures\road200_scenario_examples.png
  D:\MFE204_RoadDetection\repo\figures\road200_evaluation_pipeline.png

The scenario grid uses the already fixed, deterministically selected cases
listed in qualitative_selection.csv. It does not draw detection boxes, so it
introduces the four Road200 scenarios cleanly in the Methodology section.
"""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image

BASE = Path(r"D:\MFE204_RoadDetection")
REPO_FIGURES = BASE / "repo" / "figures"
QUAL_DIR = (
    BASE
    / "subsets"
    / "bdd100k_road200_final"
    / "eval_coco_standard"
    / "full_road200"
    / "qualitative_profiles_v1"
)
SELECTION_CSV = QUAL_DIR / "qualitative_selection.csv"

SCENARIO_ORDER = [
    "daytime_normal",
    "night_lowlight",
    "crowded_occluded",
    "small_distant",
]


def load_selected_cases() -> dict[str, dict[str, str]]:
    with SELECTION_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    by_scenario = {row["scenario"]: row for row in rows}
    missing = set(SCENARIO_ORDER).difference(by_scenario)
    if missing:
        raise RuntimeError(
            "Missing selected cases for: " + ", ".join(sorted(missing))
        )
    return by_scenario


def make_scenario_grid(selected: dict[str, dict[str, str]]) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 7.4), constrained_layout=True)

    for axis, scenario in zip(axes.flat, SCENARIO_ORDER):
        row = selected[scenario]
        with Image.open(Path(row["image_path"])) as image:
            axis.imshow(image.convert("RGB"))
        axis.set_axis_off()
        axis.set_title(
            scenario.replace("_", " ").title(),
            fontsize=12,
            pad=6,
        )

    figure.suptitle(
        "Representative Road200 Images from the Four Evaluation Scenarios",
        fontsize=15,
    )
    figure.savefig(
        REPO_FIGURES / "road200_scenario_examples.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def add_box(axis, x, y, width, height, text, fontsize=9):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.3,
    )
    axis.add_patch(patch)
    axis.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        wrap=True,
        fontsize=fontsize,
    )


def arrow(axis, x1, y1, x2, y2):
    axis.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops={"arrowstyle": "->", "lw": 1.3},
    )


def make_pipeline_figure() -> None:
    figure, axis = plt.subplots(figsize=(12.6, 7.6))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_axis_off()

    # Left-hand construction and model-evaluation stream.
    add_box(
        axis, 0.04, 0.76, 0.20, 0.13,
        "Locked Road200\nBDD100K-val, 200 images\n4 scenarios × 50",
    )
    add_box(
        axis, 0.31, 0.76, 0.20, 0.13,
        "Official BDD100K\nbox2d labels\n8 shared COCO classes",
    )
    add_box(
        axis, 0.58, 0.76, 0.16, 0.13,
        "Pretrained\nYOLOv8n and\nRT-DETR-L",
    )
    add_box(
        axis, 0.80, 0.76, 0.16, 0.13,
        "COCO evaluator\nmAP, AP50, AP75\nscenario metrics",
    )

    arrow(axis, 0.24, 0.825, 0.31, 0.825)
    arrow(axis, 0.51, 0.825, 0.58, 0.825)
    arrow(axis, 0.74, 0.825, 0.80, 0.825)

    # Lower RT-DETR-L output profile stream.
    add_box(
        axis, 0.10, 0.45, 0.20, 0.13,
        "Frozen RT-DETR-L\nranked predictions\nscore floor = 0.001",
    )
    add_box(
        axis, 0.40, 0.45, 0.20, 0.13,
        "Controlled output layer\nconfidence floor → optional\nclass-aware NMS → top-k",
    )
    add_box(
        axis, 0.72, 0.51, 0.20, 0.10,
        "Balanced profile\n0.25 / NMS 0.45 /\ntop-100",
    )
    add_box(
        axis, 0.72, 0.33, 0.20, 0.10,
        "Compact profile\n0.35 / NMS 0.55 /\ntop-100",
    )

    arrow(axis, 0.26, 0.76, 0.20, 0.58)
    arrow(axis, 0.30, 0.515, 0.40, 0.515)
    arrow(axis, 0.60, 0.515, 0.72, 0.56)
    arrow(axis, 0.60, 0.485, 0.72, 0.38)

    # Outputs and reporting.
    add_box(
        axis, 0.14, 0.10, 0.23, 0.13,
        "Single-factor ablations\nconfidence / max detections\nexternal class-aware NMS",
    )
    add_box(
        axis, 0.47, 0.10, 0.20, 0.13,
        "Deterministic\nqualitative cases\none per scenario",
    )
    add_box(
        axis, 0.77, 0.10, 0.17, 0.13,
        "Paper evidence\nmetrics, ablations,\nand visual analysis",
    )

    arrow(axis, 0.50, 0.45, 0.255, 0.23)
    arrow(axis, 0.82, 0.33, 0.57, 0.23)
    arrow(axis, 0.37, 0.165, 0.47, 0.165)
    arrow(axis, 0.67, 0.165, 0.77, 0.165)

    axis.set_title(
        "Road200 Evaluation and RT-DETR-L Output-Refinement Workflow",
        fontsize=16,
        pad=15,
    )
    figure.savefig(
        REPO_FIGURES / "road200_evaluation_pipeline.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def main() -> None:
    if not SELECTION_CSV.exists():
        raise FileNotFoundError(
            f"Selection file not found: {SELECTION_CSV}\n"
            "Run generate_road200_qualitative_profiles.py first."
        )

    REPO_FIGURES.mkdir(parents=True, exist_ok=True)
    selected = load_selected_cases()
    make_scenario_grid(selected)
    make_pipeline_figure()

    print("Created:")
    print(REPO_FIGURES / "road200_scenario_examples.png")
    print(REPO_FIGURES / "road200_evaluation_pipeline.png")


if __name__ == "__main__":
    main()
