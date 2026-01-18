# Bayesian Product of Experts for Context-Aware Urban Biodiversity Monitoring

This repository contains the curated code + notebooks used in my master's thesis.

## Main notebooks (recommended order)
1. `notebooks/01_env_xgboost.ipynb` — Environmental expert (XGBoost) trained on GBIF-derived occurrence + environmental covariates.
2. `notebooks/02_visual_resnet.ipynb` — Visual expert (ResNet-18) trained on image dataset.
3. `notebooks/03_fusion_poe.ipynb` — Calibration + Product-of-Experts fusion + evaluation.

## Data & models
Large datasets (GBIF exports, image folders, WorldCover tiles, weather caches) and trained checkpoints are not committed to Git
due to size/licensing. Paths are configured in `configs/paths.yaml`.

## Quick setup
- Create env (example): `conda env create -f environment.yml`
- Activate: `conda activate thesis-py312`
- Run notebooks in order.

