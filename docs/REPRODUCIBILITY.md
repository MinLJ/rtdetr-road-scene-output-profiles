# Reproducibility Notes

## Scope of the released artifact

The release contains the Road200 manifest, category mapping, paper source, figure-generation code, automatic evaluation scripts, post-processing scripts, and reported CSV summaries.

The release intentionally excludes:

- raw BDD100K images;
- original BDD100K annotation files;
- model weights;
- local Conda environments and caches;
- unpublished personal or machine-specific files.

## Required external resources

1. Obtain BDD100K validation images and `box2d` annotations through the official source.
2. Install a compatible Python environment with PyTorch, Ultralytics, `pycocotools`, and the packages imported by the scripts.
3. Configure the dataset root and output root using repository-relative paths.

## Reported protocol

- Benchmark: Road200, 200 images, four groups of 50.
- Shared categories: person, bicycle, car, motorcycle, bus, train, truck, traffic light.
- AP evaluation: class-aware COCO bbox evaluation, IoU 0.50:0.95.
- Operating point: confidence 0.25 and IoU 0.50 for precision, recall, and F1.
- RT-DETR-L profile study: confidence filtering, top-K capping, and external class-aware NMS on frozen predictions.

## Important limitation

The current research scripts were developed on a local Windows environment. Before public release, remove all hard-coded local paths and expose required paths through CLI options or a configuration file. The clean-clone test in `RELEASE_CHECKLIST.md` is mandatory before publishing.
