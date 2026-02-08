# Project map (thesis pipeline)

## Components
- **Environmental expert:** XGBoost HSI trained on GBIF-derived occurrence + environmental covariates
- **Visual expert:** ResNet-18 trained on image dataset (multiple seeds/runs)
- **Fusion:** Product of Experts (PoE) combining contextual and visual predictions

## Intended notebook/script flow
Single source of truth for paths: `configs/paths.yaml` (repo-relative), loaded via `digital_naturalist.paths.load_paths()`.

1) **Context model (XGBoost)**
   - Notebook: `notebooks/01_env_xgboost.ipynb`
   - Analysis/plots: `notebooks/02_env_xgboost_analysis.ipynb`

2) **Vision model (ResNet-18)**
   - Notebook: `notebooks/03_vision_resnet.ipynb`

3) **Fusion (PoE)**
   - Notebook: `notebooks/04_fusion_poe.ipynb`

## CLI wrappers (optional)
Scripts in `scripts/` resolve and print paths (`--dry-run`) and can be extended to execute notebooks headlessly
(e.g., via `jupyter nbconvert`) or to call moved logic from `src/` (future improvement).

## Data layout (local, not tracked)
- `data/`: GBIF exports, image folders, WorldCover tiles, weather caches
- `models/`: trained checkpoints (ResNet + temperatures, XGBoost artifacts)
- `outputs/`: generated figures/tables and intermediate artefacts
