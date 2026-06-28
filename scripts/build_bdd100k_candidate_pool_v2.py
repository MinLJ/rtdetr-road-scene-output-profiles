from pathlib import Path
import csv
from collections import defaultdict

# ============================================================
# PATHS
# ============================================================
MANIFEST = Path(
    r"D:\MFE204_RoadDetection\subsets\bdd100k_road200_v1\manifests"
    r"\bdd100k_candidate_pool_280.csv"
)

OUTPUT_DIR = MANIFEST.parent
OUTPUT_CSV = OUTPUT_DIR / "bdd100k_candidate_pool_280_v2.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "bdd100k_candidate_pool_summary_v2.txt"


# ============================================================
# CONFIG
# ============================================================
SCENARIOS = [
    "daytime_normal",
    "night_lowlight",
    "crowded_occluded",
    "small_distant",
]

PER_SCENARIO = 70


# ============================================================
# LOAD
# ============================================================
def load_manifest():
    with open(MANIFEST, "r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


# ============================================================
# CROWDED FIX (核心修复点)
# ============================================================
def is_static_density(row):
    """
    排除“路边停车密集但非交通拥堵”的假 crowded
    """
    return (
        row["scene"] == "city street"
        and int(row["dynamic_box_count"]) < 18
        and int(row["occluded_dynamic_count"]) < 8
    )


def crowded_score(row):
    """
    crowded/occluded 强度评分（只用于排序，不影响其他组）
    """
    return (
        int(row["dynamic_box_count"]) * 1.0
        + int(row["occluded_dynamic_count"]) * 2.0
        + float(row["dynamic_occlusion_ratio"]) * 10.0
        + int(row["small_dynamic_count"]) * 0.5
    )


# ============================================================
# MAIN LOGIC
# ============================================================
def build():
    rows = load_manifest()

    grouped = defaultdict(list)
    for r in rows:
        grouped[r["scenario_candidate"]].append(r)

    selected = []

    summary = []

    for sc in SCENARIOS:
        pool = grouped[sc]

        if sc != "crowded_occluded":
            # ====================================================
            # 其他三组：保持原逻辑（不动）
            # ====================================================
            chosen = pool[:PER_SCENARIO]

        else:
            # ====================================================
            # crowded_occluded V2 FIX
            # ====================================================

            filtered = []

            for r in pool:
                if is_static_density(r):
                    continue
                filtered.append(r)

            # 按“真实拥挤/遮挡强度排序”
            filtered.sort(key=crowded_score, reverse=True)

            chosen = filtered[:PER_SCENARIO]

        selected.extend(chosen)

        summary.append(
            f"{sc}: {len(pool)} -> {len(chosen)} selected (filtered={sc=='crowded_occluded'})"
        )

    # ============================================================
    # SAVE CSV
    # ============================================================
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(selected)

    # ============================================================
    # SAVE SUMMARY
    # ============================================================
    with open(OUTPUT_SUMMARY, "w", encoding="utf-8") as f:
        f.write("\n".join(summary))

    print("DONE")
    print("\n".join(summary))
    print(f"\nSaved: {OUTPUT_CSV}")
    print(f"Saved: {OUTPUT_SUMMARY}")


if __name__ == "__main__":
    build()