from __future__ import annotations

"""
Generate paper-facing Section 4 figures from the frozen Road200 result files.

The script reads the validated evaluation outputs already produced in:
  subsets/bdd100k_road200_final/eval_coco_standard/full_road200/

It creates only data-driven figures; it does not run any model inference and
does not alter metrics.

Output:
  D:\MFE204_RoadDetection\repo\figures\
    road200_overall_ap_comparison.png
    road200_scenario_map_comparison.png
    rtdetr_confidence_tradeoff.png
    qualitative_crowded_profiles.png
    qualitative_small_distant_profiles.png
"""

from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE = Path(r"D:\MFE204_RoadDetection")
RESULTS_DIR = (
    BASE
    / "subsets"
    / "bdd100k_road200_final"
    / "eval_coco_standard"
    / "full_road200"
)
FIGURES_DIR = BASE / "repo" / "figures"

OVERALL_CSV = RESULTS_DIR / "metrics_overall.csv"
SCENARIO_CSV = RESULTS_DIR / "metrics_by_scenario.csv"
ABLATION_CSV = (
    RESULTS_DIR
    / "ablation_rtdetr_postprocess_v1"
    / "ablation_summary.csv"
)
QUALITATIVE_DIR = RESULTS_DIR / "qualitative_profiles_v1"


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required input not found:\n{path}")


def make_overall_ap_comparison(overall: pd.DataFrame) -> None:
    rows = overall.set_index("model").loc[["yolo", "rtdetr"]]
    metrics = ["mAP_50_95", "AP50", "AP75"]
    labels = ["mAP@[.50:.95]", "AP50", "AP75"]

    positions = np.arange(len(metrics))
    width = 0.34

    plt.figure(figsize=(7.4, 4.9))
    plt.bar(
        positions - width / 2,
        rows.loc["yolo", metrics].to_numpy(dtype=float),
        width,
        label="YOLOv8n",
    )
    plt.bar(
        positions + width / 2,
        rows.loc["rtdetr", metrics].to_numpy(dtype=float),
        width,
        label="RT-DETR-L",
    )
    plt.xticks(positions, labels)
    plt.ylabel("COCO score")
    plt.ylim(0.0, 0.70)
    plt.title("Overall Class-aware COCO Detection Performance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "road200_overall_ap_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def make_scenario_map_comparison(scenario: pd.DataFrame) -> None:
    ordered_scenarios = [
        "daytime_normal",
        "night_lowlight",
        "crowded_occluded",
        "small_distant",
    ]
    display_labels = [
        "Daytime\nnormal",
        "Night\nlow-light",
        "Crowded /\noccluded",
        "Small /\ndistant",
    ]

    yolo = (
        scenario[scenario["model"] == "yolo"]
        .set_index("scenario")
        .loc[ordered_scenarios, "mAP_50_95"]
        .to_numpy(dtype=float)
    )
    rtdetr = (
        scenario[scenario["model"] == "rtdetr"]
        .set_index("scenario")
        .loc[ordered_scenarios, "mAP_50_95"]
        .to_numpy(dtype=float)
    )

    positions = np.arange(len(ordered_scenarios))
    width = 0.34

    plt.figure(figsize=(8.0, 5.1))
    plt.bar(positions - width / 2, yolo, width, label="YOLOv8n")
    plt.bar(positions + width / 2, rtdetr, width, label="RT-DETR-L")
    plt.xticks(positions, display_labels)
    plt.ylabel("mAP@[.50:.95]")
    plt.ylim(0.0, 0.50)
    plt.title("Scenario-wise Class-aware COCO Performance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "road200_scenario_map_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def make_confidence_tradeoff(ablation: pd.DataFrame) -> None:
    rows = (
        ablation[ablation["family"] == "confidence"]
        .sort_values("confidence_floor")
        .copy()
    )

    if rows.empty:
        raise RuntimeError("No confidence-family rows found in ablation_summary.csv.")

    x = rows["confidence_floor"].to_numpy(dtype=float)

    plt.figure(figsize=(7.6, 5.3))
    plt.plot(x, rows["mAP_50_95"].to_numpy(dtype=float), marker="o", label="mAP@[.50:.95]")
    plt.plot(x, rows["precision"].to_numpy(dtype=float), marker="o", label="Precision")
    plt.plot(x, rows["recall"].to_numpy(dtype=float), marker="o", label="Recall")
    plt.plot(x, rows["F1"].to_numpy(dtype=float), marker="o", label="F1")
    plt.xlabel("Confidence floor")
    plt.ylabel("Score")
    plt.xlim(0.0, 0.52)
    plt.ylim(0.0, 1.0)
    plt.title("RT-DETR-L Confidence-floor Trade-off")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "rtdetr_confidence_tradeoff.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def copy_qualitative_cases() -> None:
    case_mapping = {
        "case_crowded_occluded.png": "qualitative_crowded_profiles.png",
        "case_small_distant.png": "qualitative_small_distant_profiles.png",
    }
    for source_name, target_name in case_mapping.items():
        source = QUALITATIVE_DIR / source_name
        require(source)
        shutil.copy2(source, FIGURES_DIR / target_name)


def main() -> None:
    for path in [OVERALL_CSV, SCENARIO_CSV, ABLATION_CSV]:
        require(path)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    overall = pd.read_csv(OVERALL_CSV)
    scenario = pd.read_csv(SCENARIO_CSV)
    ablation = pd.read_csv(ABLATION_CSV)

    make_overall_ap_comparison(overall)
    make_scenario_map_comparison(scenario)
    make_confidence_tradeoff(ablation)
    copy_qualitative_cases()

    print("Created/updated Section 4 figures:")
    for filename in [
        "road200_overall_ap_comparison.png",
        "road200_scenario_map_comparison.png",
        "rtdetr_confidence_tradeoff.png",
        "qualitative_crowded_profiles.png",
        "qualitative_small_distant_profiles.png",
    ]:
        print(FIGURES_DIR / filename)


if __name__ == "__main__":
    main()
