# Road200 data metadata

This directory contains metadata required to identify and reconstruct the locked Road200 subset without redistributing raw BDD100K content.

## Files

- `road200_manifest.csv` — Road200 image identifier, original BDD100K identifier, scenario label, and relative expected paths.
- `road200_category_mapping.csv` — BDD100K-to-COCO category mapping used for class-aware evaluation.

## Manifest path convention

The public manifest uses repository-independent relative paths:

- `image_relpath` is relative to a reconstructed Road200 data root, for example `images/<filename>.jpg`.
- `label_relpath` is relative to the same root, for example `labels/<filename>.json`.

The raw image and annotation files are intentionally not stored in this repository.

## Data policy

Obtain BDD100K validation images and source annotations from the official BDD100K distribution channel. Before publishing any raw image, annotation, crop, or derived dataset artifact, verify that redistribution is permitted under the applicable source terms.

## Expected Road200 scenarios

| Scenario | Images |
|---|---:|
| daytime_normal | 50 |
| night_lowlight | 50 |
| crowded_occluded | 50 |
| small_distant | 50 |
| **Total** | **200** |
