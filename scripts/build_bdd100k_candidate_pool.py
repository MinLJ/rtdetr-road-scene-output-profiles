from __future__ import annotations

import csv
import hashlib
import random
from pathlib import Path

# ============================================================
# Input / output paths
# ============================================================
AUDIT_CSV = Path(
    r"D:\MFE204_RoadDetection\docs\bdd100k_audit_v2\bdd100k_val_audit_v2.csv"
)

OUTPUT_ROOT = Path(
    r"D:\MFE204_RoadDetection\subsets\bdd100k_road200_v1"
)

MANIFEST_DIR = OUTPUT_ROOT / "manifests"
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

MASTER_MANIFEST = MANIFEST_DIR / "bdd100k_candidate_pool_280.csv"
SUMMARY_FILE = MANIFEST_DIR / "bdd100k_candidate_pool_summary.txt"

# We deliberately over-sample before visual review.
CANDIDATES_PER_SCENARIO = 70

# Fixed seed: makes the candidate selection reproducible.
RANDOM_SEED = 20260626


def to_int(value: str) -> int:
    return int(float(value))


def to_float(value: str) -> float:
    return float(value)


def stable_tie_breaker(image_id: str) -> str:
    """Stable ordering independent of OS file order."""
    return hashlib.sha1(image_id.encode("utf-8")).hexdigest()


def load_rows() -> list[dict]:
    if not AUDIT_CSV.exists():
        raise FileNotFoundError(f"Audit CSV not found:\n{AUDIT_CSV}")

    rows = []

    with open(AUDIT_CSV, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            row["dynamic_box_count"] = to_int(row["dynamic_box_count"])
            row["small_dynamic_count"] = to_int(row["small_dynamic_count"])
            row["occluded_dynamic_count"] = to_int(
                row["occluded_dynamic_count"]
            )
            row["dynamic_occlusion_ratio"] = to_float(
                row["dynamic_occlusion_ratio"]
            )

            row["timeofday"] = row["timeofday"].strip().lower()
            row["weather"] = row["weather"].strip().lower()
            row["scene"] = row["scene"].strip().lower()

            rows.append(row)

    return rows


def is_daytime_normal(row: dict) -> bool:
    return (
        row["timeofday"] == "daytime"
        and 3 <= row["dynamic_box_count"] <= 8
        and row["dynamic_occlusion_ratio"] <= 0.20
    )


def is_night_lowlight(row: dict) -> bool:
    return (
        row["timeofday"] == "night"
        and 5 <= row["dynamic_box_count"] <= 12
    )


def is_crowded_occluded(row: dict) -> bool:
    return (
        row["timeofday"] in {"daytime", "dawn/dusk"}
        and row["dynamic_box_count"] >= 17
        and row["dynamic_occlusion_ratio"] >= 0.75
    )


def is_small_distant(row: dict) -> bool:
    return (
        row["timeofday"] in {"daytime", "dawn/dusk"}
        and 9 <= row["dynamic_box_count"] <= 16
        and row["small_dynamic_count"] >= 10
    )


def choose_candidates(
    rows: list[dict],
    predicate,
    scenario_name: str,
    used_image_ids: set[str],
    seed_offset: int,
) -> tuple[list[dict], int]:
    """
    Find eligible rows, exclude previously selected images,
    then choose a deterministic random sample.
    """
    eligible = [
        row
        for row in rows
        if predicate(row) and row["image_id"] not in used_image_ids
    ]

    # Stable sort before sampling so result remains reproducible.
    eligible.sort(key=lambda row: stable_tie_breaker(row["image_id"]))

    if len(eligible) < CANDIDATES_PER_SCENARIO:
        raise RuntimeError(
            f"{scenario_name}: only {len(eligible)} eligible images found, "
            f"but {CANDIDATES_PER_SCENARIO} are required."
        )

    rng = random.Random(RANDOM_SEED + seed_offset)
    selected = rng.sample(eligible, CANDIDATES_PER_SCENARIO)

    # Make output visually easier to review.
    selected.sort(
        key=lambda row: (
            row["timeofday"],
            row["scene"],
            row["weather"],
            row["dynamic_box_count"],
            stable_tie_breaker(row["image_id"]),
        )
    )

    for row in selected:
        row["scenario_candidate"] = scenario_name
        row["candidate_status"] = "pending_visual_review"
        row["visual_review_decision"] = ""
        row["visual_review_reason"] = ""

    return selected, len(eligible)


def main() -> None:
    rows = load_rows()

    used_image_ids: set[str] = set()

    scenario_definitions = [
        (
            "daytime_normal",
            is_daytime_normal,
            "daytime; 3-8 dynamic objects; dynamic occlusion ratio <= 0.20",
        ),
        (
            "night_lowlight",
            is_night_lowlight,
            "night; 5-12 dynamic objects",
        ),
        (
            "crowded_occluded",
            is_crowded_occluded,
            "daytime/dawn-dusk; >=17 dynamic objects; "
            "dynamic occlusion ratio >= 0.75",
        ),
        (
            "small_distant",
            is_small_distant,
            "daytime/dawn-dusk; 9-16 dynamic objects; "
            ">=10 small dynamic objects",
        ),
    ]

    all_selected = []
    scenario_results = []

    for seed_offset, (scenario_name, predicate, rule_text) in enumerate(
        scenario_definitions,
        start=1,
    ):
        selected, eligible_count = choose_candidates(
            rows=rows,
            predicate=predicate,
            scenario_name=scenario_name,
            used_image_ids=used_image_ids,
            seed_offset=seed_offset,
        )

        used_image_ids.update(row["image_id"] for row in selected)
        all_selected.extend(selected)

        scenario_results.append(
            {
                "scenario": scenario_name,
                "eligible_before_sampling": eligible_count,
                "selected_candidates": len(selected),
                "rule": rule_text,
            }
        )

    if len(all_selected) != 4 * CANDIDATES_PER_SCENARIO:
        raise RuntimeError(
            f"Expected {4 * CANDIDATES_PER_SCENARIO} candidates, "
            f"but got {len(all_selected)}."
        )

    output_columns = [
        "image_id",
        "split",
        "image_path",
        "scenario_candidate",
        "candidate_status",
        "visual_review_decision",
        "visual_review_reason",
        "timeofday",
        "weather",
        "scene",
        "mapped_box_count",
        "dynamic_box_count",
        "occluded_dynamic_count",
        "dynamic_occlusion_ratio",
        "small_dynamic_count",
        "mapped_classes_present",
        "raw_classes_present",
    ]

    with open(
        MASTER_MANIFEST,
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=output_columns)
        writer.writeheader()

        for row in all_selected:
            writer.writerow(
                {column: row.get(column, "") for column in output_columns}
            )

    lines = [
        "BDD100K Road-Scene Candidate Pool Summary",
        "=" * 64,
        "",
        f"Input audit CSV: {AUDIT_CSV}",
        f"Random seed: {RANDOM_SEED}",
        f"Candidates per scenario: {CANDIDATES_PER_SCENARIO}",
        f"Total candidate images: {len(all_selected)}",
        "",
        "Scenario definitions and candidate availability:",
    ]

    for result in scenario_results:
        lines.extend(
            [
                "",
                f"[{result['scenario']}]",
                f"  Rule: {result['rule']}",
                f"  Eligible images before sampling: "
                f"{result['eligible_before_sampling']}",
                f"  Selected for visual review: "
                f"{result['selected_candidates']}",
            ]
        )

    lines.extend(
        [
            "",
            "Next step:",
            "Perform visual review for all candidate images.",
            "For each scenario, retain exactly 50 images and mark the remaining",
            "20 as excluded with a specific reason in the manifest.",
        ]
    )

    SUMMARY_FILE.write_text("\n".join(lines), encoding="utf-8")

    print("Candidate pool creation complete.")
    print(f"Total selected candidates: {len(all_selected)}")
    print(f"Manifest: {MASTER_MANIFEST}")
    print(f"Summary: {SUMMARY_FILE}")

    print("\nEligible candidate counts:")
    for result in scenario_results:
        print(
            f"  {result['scenario']}: "
            f"{result['eligible_before_sampling']} eligible -> "
            f"{result['selected_candidates']} selected"
        )


if __name__ == "__main__":
    main()