# RT-DETR-L for Road-Scene Object Detection

This repository accompanies the study **“RT-DETR-L for Road-Scene Object Detection: Ground-Truth Evaluation and Output-Profile Analysis.”**

It provides the Road200 evaluation manifest, category mapping, evaluation and output-profile scripts, final paper source, and the reported result summaries. The study compares pretrained YOLOv8n and RT-DETR-L on a locked 200-image BDD100K validation subset spanning four road-scene conditions:

- daytime normal;
- night low-light;
- crowded/occluded;
- small/distant.

## What is included

```text
paper/      LaTeX source, bibliography, compiled paper PDF, and final figures
scripts/    Evaluation, ablation, output-profile, and figure-generation scripts
data/       Road200 manifest, category mapping, and data-access instructions
results/    Reported COCO metrics, ablation summaries, and qualitative-case record
docs/       Reproducibility notes and pre-release checklist
```

## What is not included

This repository does **not** redistribute raw BDD100K images, original BDD100K annotations, model weights, local caches, or machine-specific environments. Obtain the source dataset from its official distribution channel and review the applicable dataset terms before use.

## Reproducibility status

The committed result summaries reproduce the values reported in the paper. Before a public release is final, the scripts must be checked for any machine-specific absolute paths and converted to repository-relative command-line arguments or a configuration file. See [`docs/RELEASE_CHECKLIST.md`](docs/RELEASE_CHECKLIST.md).

## Reported headline result

On the locked Road200 benchmark, RT-DETR-L obtained mAP@[0.50:0.95] of 0.247, compared with 0.127 for YOLOv8n. RT-DETR-L achieved higher recall and F1 at the selected operating point, while YOLOv8n achieved higher precision. The post-processing study separates standard raw AP-ranking output from balanced and compact output profiles.

## Citation

Add the final publication citation here after the paper has been submitted or accepted.

## License

A repository license has not yet been selected. Add a license only after all authors agree on the intended reuse terms for the code, paper source, and derived artifacts.
