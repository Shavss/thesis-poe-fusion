# Project map (thesis pipeline)

## Components
- Environmental expert: XGBoost trained on GBIF-derived occurrence + environmental covariates
- Visual expert: ResNet-18 trained on image dataset
- Fusion: Product of Experts combining contextual and visual predictions

## Intended notebook/script flow (curation pending)
1) Context model (XGBoost): GBIF ingestion -> feature engineering -> training -> calibration
2) Vision model (ResNet-18): dataset splits -> training (multiple runs) -> calibration
3) Fusion: combine experts (PoE) -> evaluation -> figures/tables

## Data layout (local, not tracked)
- data/: GBIF exports, image folders, WorldCover tiles, weather caches
- models/: trained checkpoints
- outputs/: generated figures/tables and intermediate artefacts

Paths are configured in `configs/paths.yaml` (relative to repo root), with optional env-var overrides.
