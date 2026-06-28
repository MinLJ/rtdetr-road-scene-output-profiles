# RT-DETR-L Road-Scene Output-Profile Analysis

This repository accompanies the project **“RT-DETR-L for Road-Scene Object Detection: Ground-Truth Evaluation and Output-Profile Analysis.”**

The project evaluates pretrained **RT-DETR-L** and **YOLOv8n** on Road200, a locked 200-image road-scene case-study subset constructed from the BDD100K validation split. In addition to detector-level comparison, the project examines confidence filtering, top-$K$ capping, and external class-aware NMS as output-profile choices for RT-DETR-L.

## Key result

On the locked Road200 benchmark:

| Model     | mAP@[0.50:0.95] |  AP50 |  AP75 | Precision | Recall |    F1 |
| --------- | --------------: | ----: | ----: | --------: | -----: | ----: |
| YOLOv8n   |           0.127 | 0.221 | 0.120 |     0.718 |  0.315 | 0.438 |
| RT-DETR-L |           0.247 | 0.456 | 0.231 |     0.531 |  0.608 | 0.567 |

RT-DETR-L achieved higher overall AP, recall, and F1 on Road200, while YOLOv8n achieved higher precision at the selected operating point.

## Road200 benchmark

Road200 is a locked 200-image case-study subset drawn from the BDD100K validation split.

| Scenario group     |  Images |
| ------------------ | ------: |
| Daytime normal     |      50 |
| Night low-light    |      50 |
| Crowded / occluded |      50 |
| Small / distant    |      50 |
| **Total**          | **200** |

The evaluation uses eight shared categories:

````text
time normal | 50 |
| Night low-light | 50 |
| Crowded / occluded | 50 |
| Small / distant | 50 |
| **Total** | **200** |

The evaluation uses eight shared categories:

```text
person, bicycle, car, motorcycle,
bus, train, truck, traffic light
````

The subset manifest and BDD100K-to-COCO category mapping are provided under [`data/`](data/).

## Repository structure

```text
paper/      Final paper PDF, LaTeX source, bibliography, and publication figures
data/       Road200 manifest, category mapping, and data-access notes
results/    Reported evaluation summaries, ablations, profiles, and qualitative artifacts
scripts/    Evaluation, post-processing, qualitative-analysis, and figure-generation scripts
docs/       Release notes and project documentation
```

## Reported artifacts

The repository includes:

* overall, scenario-level, class-level, and IoU-level COCO evaluation summaries;
* RT-DETR-L confidence-threshold, NMS, and top-$K$ ablation summaries;
* balanced and compact output-profile summaries;
* deterministic qualitative-case selection records;
* a supplementary small/distant qualitative output-profile comparison;
* the final paper PDF and corresponding LaTeX source.

## Data and usage notes

This repository does **not** redistribute:

* raw BDD100K images;
* original BDD100K annotations;
* model weights;
* local Conda environments, caches, or experiment logs.

The Road200 manifest uses repository-independent relative paths. Users who wish to run the released scripts must obtain the relevant BDD100K material through its official distribution channel and comply with the applicable dataset terms.

The scripts are released in their original research form and may require local path adjustments before execution on a different machine.

## Citation

A formal citation will be added after the paper is submitted or accepted.

## Acknowledgement

This work was developed as part of an academic road-scene object-detection project
