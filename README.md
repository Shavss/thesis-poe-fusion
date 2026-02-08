# Digital Naturalist — Context-Aware Urban Biodiversity Monitoring (PoE Fusion)

This repository contains the code for a Master’s thesis on **context-aware insect species classification** using a **Bayesian Product of Experts (PoE)** late-fusion approach:

- **Vision expert:** ResNet-18 (10 independent runs with different seeds)
- **Context expert:** XGBoost Habitat Suitability Index (HSI)
- **Fusion rule:**  p(y | x, c) ∝ p_vis(y | x) · p_ctx(y | c)


The project is designed to work in two modes:
- **Notebook mode (primary):** notebooks in `notebooks/` reproduce the thesis pipeline.
- **CLI mode (optional):** scripts in `scripts/` run the same pipeline from the terminal (recommended for end users).

> **Important:** This repository does **not** include the training data or pretrained models.  
> End users must provide the datasets locally (see **Data setup**).

---

## Repository layout

- `configs/paths.yaml` — central configuration for data/models/outputs
- `src/digital_naturalist/` — shared utilities (paths loader, feature logic, etc.)
- `notebooks/` — primary research notebooks (training + evaluation)
- `scripts/` — CLI entry points
- `models/` — models saved locally (not committed)
- `outputs/` — generated figures/metrics/predictions (not committed)
- `data/` — datasets (not committed)

---

## Installation

### 1) Create environment (Conda recommended)

```bash
conda env create -f environment.yml
conda activate <your-env-name>
```

### 2) Install the project (so imports work from notebooks and scripts)

```bash
pip install -e .
```

### 3) Quick sanity check

```bash
python -c "from digital_naturalist.paths import load_paths; print(load_paths('configs/paths.yaml'))"
```

If that prints a dictionary of resolved paths, you’re good.

---

## Data setup (required)

This repo expects data to be available locally. Two sources are required:

### A) Image dataset (PyTorch `ImageFolder` structure)

Expected folder structure:

```
data/amsterdam/images_no_vespula/
  train/
    Apis_mellifera/
    ...
  val/
    Apis_mellifera/
    ...
  test/
    ...
  test2/   # temporal holdout split
    ...
```

Notes:
- Folder names must match class labels used during training.
- The notebooks/scripts assume `test2` is the primary temporal holdout.

### B) Context/GBIF tables (parquet/csv)

Expected structure (example):

```
data/amsterdam/train/
  observations_filtered_50m_accuracy.parquet
  ...
data/amsterdam/val/
  observations_filtered_50m_accuracy.parquet
  ...
```

Your exact filenames may differ — update `configs/paths.yaml` and/or notebook constants accordingly.

---

## Configuration (`configs/paths.yaml`)

All paths are treated as **repo-relative** and resolved via `src/digital_naturalist/paths.py`.

Example structure:

```yaml
data:
  gbif_train_dir: "data/amsterdam/train"
  gbif_val_dir: "data/amsterdam/val"
  image_root: "data/amsterdam/images_no_vespula"
  image_train: "train"
  image_val: "val"
  image_test: "test"
  image_test2: "test2"

models:
  root: "models"
  context_dir: "models/context"
  vision_dir: "models/vision"
  vision_temps_dir: "models/vision/temperatures"

outputs:
  root: "outputs"
  context_xgb: "outputs/context_xgb"
  vision_resnet: "outputs/vision_resnet"
  fusion_poe: "outputs/fusion_poe"

artifacts:
  root: "artifacts"
```

---

## Running the full pipeline (CLI mode)

### Overview
A typical end-to-end run consists of:

1) Train/evaluate context expert (XGBoost HSI)  
2) Train/evaluate vision expert (ResNet-18 across 10 seeds + temperature scaling)  
3) Run fusion evaluation (PoE), generate:
   - confusion matrices
   - calibration metrics (ECE, Brier, log-loss)
   - McNemar tests
   - rescued/hurt analysis
   - thesis-ready tables (CSV)

### Commands

From repo root:

```bash
python scripts/run_context_xgboost.py --config configs/paths.yaml
python scripts/train_visual_resnet.py --config configs/paths.yaml
python scripts/run_fusion_poe.py --config configs/paths.yaml --split test2
```

Outputs will appear in:
- `outputs/context_xgb/{figures,metrics,preds}`
- `outputs/vision_resnet/{figures,metrics,preds}`
- `outputs/fusion_poe/{figures,metrics,preds}`

Models will be saved locally to:
- `models/context/` (XGBoost + feature/species mapping files)
- `models/vision/` (10 ResNet `.pth` checkpoints)
- `models/vision/temperatures/` (10 temperature scaling `.npy` files)

> Training can take time and compute (especially 10 ResNet runs).  
> On macOS, the vision notebook typically uses `mps` if available.

---

## Running in notebook mode (recommended for thesis work)

Open notebooks in order:

1. `notebooks/01_env_xgboost.ipynb`  
2. `notebooks/03_vision_resnet.ipynb`  
3. `notebooks/04_fusion_poe.ipynb`

Each notebook should start with:

```python
from digital_naturalist.paths import load_paths
P = load_paths("configs/paths.yaml")
```

So paths are stable even when the notebook CWD is `.../notebooks`.

---

## Results (reference run)

This repository does not ship data or pretrained models. The following tables report the **reference results** from the author’s run on the thesis dataset setup (8 species, `test2` temporal holdout), using:

- Vision expert: **ResNet-18**, 10 independent runs (different seeds)
- Context expert: **XGBoost HSI**
- Fusion: **Product of Experts (PoE)**

> Results may vary if you change dataset versions, splits, preprocessing, or random seeds.

### Per-species performance (Top-1 accuracy)

Mean ± std across 10 ResNet runs. Δ is the improvement in **percentage points** (pp).

| Species | N | CNN (%) | Fused (%) | Δ (pp) | Rel. improv. (%) | Range (pp) |
|---|---:|---:|---:|---:|---:|---:|
| Eristalis tenax | 345 | 70.6 ± 8.3 | 75.4 ± 7.6 | +4.78 ± 1.49 | +6.8 | +2.32 to +6.96 |
| Bombus terrestris | 550 | 48.1 ± 4.7 | 52.8 ± 3.9 | +4.67 ± 1.22 | +9.7 | +3.45 to +6.73 |
| Apis mellifera | 439 | 80.8 ± 4.8 | 83.7 ± 3.4 | +2.87 ± 1.64 | +3.6 | +0.91 to +5.47 |
| Coccinella septempunctata | 326 | 71.4 ± 6.6 | 73.3 ± 6.5 | +1.93 ± 1.13 | +2.7 | -0.31 to +3.68 |
| Bombus lapidarius | 176 | 47.3 ± 7.1 | 48.9 ± 6.2 | +1.65 ± 2.53 | +3.5 | -1.70 to +6.25 |
| Episyrphus balteatus | 323 | 74.4 ± 7.2 | 75.9 ± 6.3 | +1.46 ± 1.59 | +2.0 | -1.24 to +3.41 |
| Eupeodes corollae | 84 | 60.6 ± 10.3 | 61.2 ± 8.0 | +0.60 ± 4.46 | +1.0 | -8.33 to +7.14 |
| Aglais urticae | 297 | 91.6 ± 2.3 | 90.1 ± 2.4 | -1.52 ± 0.86 | -1.7 | -2.69 to +0.34 |
| **Mean (Overall)** | **2540** | **68.6 ± 1.7** | **71.1 ± 1.5** | **+2.55 ± 0.44** | **+3.7** | **+1.73 to +3.23** |

### Fusion interpretability (Rescued vs Hurt)

“Rescued” = ResNet wrong but fused correct.  
“Hurt” = ResNet correct but fused wrong.

| Species | N | Rescued Count (%) | Hurt Count (%) | Net Impact |
|---|---:|---:|---:|---:|
| Apis mellifera | 439 | 25.3 (5.8%) | 12.7 (2.9%) | +2.87 pp |
| Eristalis tenax | 345 | 30.1 (8.7%) | 13.6 (3.9%) | +4.78 pp |
| Bombus terrestris | 550 | 38.5 (7.0%) | 12.8 (2.3%) | +4.67 pp |
| Coccinella septempunctata | 326 | 15.5 (4.8%) | 9.2 (2.8%) | +1.93 pp |
| Bombus lapidarius | 176 | 11.7 (6.6%) | 8.8 (5.0%) | +1.65 pp |
| Episyrphus balteatus | 323 | 21.0 (6.5%) | 16.3 (5.0%) | +1.46 pp |
| Aglais urticae | 297 | 2.6 (0.9%) | 7.1 (2.4%) | -1.52 pp |
| Eupeodes corollae | 84 | 5.6 (6.7%) | 5.1 (6.1%) | +0.60 pp |
| **Overall Mean** | **2540** | **150.3 (5.9%)** | **85.6 (3.4%)** | **+2.55 pp** |

### Where these tables come from

The fusion notebook exports these files (not committed by default):

- `outputs/fusion_poe/metrics/table_per_species_top1.csv`
- `outputs/fusion_poe/metrics/table_rescued_hurt.csv`

---

## Reproducibility notes

Because the repository does not include data/models:
- You **can** reproduce results if you provide the same datasets locally.
- If you use different data versions or splits, numerical results will change.

To improve comparability:
- keep seed settings (`base_seed=42`, 10 runs)
- keep dataset splits consistent (`test2` = temporal holdout)
- keep feature engineering aligned between training and evaluation notebooks

---

## What is not committed to GitHub

You should **not commit**:
- `data/`
- `outputs/`
- `models/` (unless you deliberately publish checkpoints via Git LFS)

Add these to `.gitignore`.

---

---

## Running notebooks from the CLI (nbconvert)

If you want a **single command** to execute a notebook end‑to‑end (useful for a “pipeline-like” CLI), you can run:

```bash
# Make sure the output directory exists first
mkdir -p outputs/fusion_poe

# Execute the notebook and write an executed copy into outputs/
jupyter nbconvert --to notebook --execute notebooks/04_fusion_poe.ipynb \
  --output 04_fusion_poe_executed.ipynb \
  --output-dir outputs/fusion_poe
```

### Why you saw `notebooks/outputs/...` and a `FileNotFoundError`

`nbconvert` can treat `--output` as a *path relative to the notebook* unless you use `--output-dir`.  
So this command:

```bash
jupyter nbconvert --to notebook --execute notebooks/04_fusion_poe.ipynb \
  --output outputs/fusion_poe/04_fusion_poe_executed.ipynb
```

may try to write under `notebooks/outputs/...` (and fail if that folder doesn't exist).  
Using `--output-dir outputs/fusion_poe` avoids that ambiguity and is the most robust pattern.

### Suggested “end user” flow for this repo

Because data/models are not committed, a typical user workflow is:

1. Create env + install editable package (`pip install -e .`)
2. Configure paths in `configs/paths.yaml`
3. Run notebooks interactively **or** run `nbconvert` for a reproducible executed copy
4. Read outputs from:
   - `outputs/context_xgb/`
   - `outputs/vision_resnet/`
   - `outputs/fusion_poe/`

## Licence
See `LICENSE`.

## Context feature engineering (`digital_naturalist.context_features`)

This repository includes a small feature-engineering module for the **context expert (XGBoost / HSI)**:

- `digital_naturalist.context_features.engineer_context_features(...)`
- `digital_naturalist.context_features.prepare_X(...)`

Why it exists:
- It lets you **switch between different context feature sets** (e.g. `"status_quo"` vs `"experimental"`) and measure how those choices affect **HSI performance** and downstream **PoE fusion performance**.
- It keeps feature transforms **pure and reproducible** (DataFrame in → DataFrame out), with no hidden I/O.

How to use it in notebooks (typical pattern):

```python
from digital_naturalist.context_features import engineer_context_features, prepare_X
import pandas as pd

FEATURE_SET = "status_quo"  # or "experimental"

train_df_feat = engineer_context_features(train_df, feature_set=FEATURE_SET)
val_df_feat   = engineer_context_features(val_df,   feature_set=FEATURE_SET)
test_df_feat  = engineer_context_features(test_df,  feature_set=FEATURE_SET)

feature_names = pd.read_csv(FEATURES_PATH)["feature"].tolist()
X_train = prepare_X(train_df_feat, feature_names)
X_val   = prepare_X(val_df_feat,   feature_names)
X_test  = prepare_X(test_df_feat,  feature_names)
```

Note:
- For paper/thesis-grade reproducibility, prefer using a **fixed** `feature_names_*.csv` checked into `models/context/` and do not regenerate features silently.
