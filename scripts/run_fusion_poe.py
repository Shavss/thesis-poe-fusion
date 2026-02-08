#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from digital_naturalist.paths import load_paths


def _require_dir(p: Path, label: str) -> None:
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"{label} not found or not a directory: {p}")


def _require_file(p: Path, label: str) -> None:
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"{label} not found or not a file: {p}")


def _mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Product-of-Experts fusion (ResNet × HSI) evaluation.")
    ap.add_argument("--config", default="configs/paths.yaml", help="Path to paths.yaml (repo-relative or absolute).")
    ap.add_argument("--dry-run", action="store_true", help="Print resolved paths and exit.")
    ap.add_argument("--split", default="test2", choices=["test", "test2"], help="Which image split to evaluate on.")
    args = ap.parse_args()

    P = load_paths(args.config)

    image_split_dir: Path = P["IMAGE_TEST2_DIR"] if args.split == "test2" else P["IMAGE_TEST_DIR"]

    # Models
    vision_models_dir: Path = P["VISION_MODEL_DIR"]
    vision_temps_dir: Path = P["VISION_TEMPS_DIR"]
    context_model_dir: Path = P["CONTEXT_MODEL_DIR"]

    # Outputs
    out_fusion: Path = P["OUT_FUSION_POE"]
    fig_dir: Path = out_fusion / "figures"
    met_dir: Path = out_fusion / "metrics"
    prd_dir: Path = out_fusion / "preds"
    log_dir: Path = (P.get("OUTPUTS_ROOT") or out_fusion.parent) / "logs"

    # Expected context files (based on your repo structure)
    hsi_model_path = context_model_dir / "xgboost_hsi_model_FINAL_no_vespula.json"
    feature_names_path = context_model_dir / "feature_names_FINAL_no_vespula.csv"
    species_mapping_path = context_model_dir / "species_mapping_FINAL_no_vespula.csv"

    print("\n=== Fusion PoE — resolved paths ===")
    print("Config:             ", Path(args.config))
    print("Image split dir:     ", image_split_dir)
    print("Vision models dir:   ", vision_models_dir)
    print("Vision temps dir:    ", vision_temps_dir)
    print("Context model dir:   ", context_model_dir)
    print("HSI model:           ", hsi_model_path)
    print("Feature names:       ", feature_names_path)
    print("Species mapping:     ", species_mapping_path)
    print("Outputs/figures:     ", fig_dir)
    print("Outputs/metrics:     ", met_dir)
    print("Outputs/preds:       ", prd_dir)
    print("Logs:                ", log_dir)
    print("==================================\n")

    if args.dry_run:
        return

    _require_dir(image_split_dir, "Image split folder")
    _require_dir(vision_models_dir, "Vision models folder")
    _require_dir(context_model_dir, "Context model folder")
    _require_file(hsi_model_path, "HSI model JSON")
    _require_file(feature_names_path, "Feature names CSV")
    _require_file(species_mapping_path, "Species mapping CSV")

    for d in (fig_dir, met_dir, prd_dir, log_dir):
        _mkdir(d)

    # ------------------------------------------------------------------
    # Hook for the real fusion code
    # ------------------------------------------------------------------
    try:
        from digital_naturalist.fusion.run import run_fusion_poe  # type: ignore
    except Exception:
        print(
            "This CLI wrapper is ready, but the fusion implementation is currently in notebooks.\n"
            "Next step (recommended): move the logic from notebooks/04_fusion_poe.ipynb into\n"
            "  src/digital_naturalist/fusion/run.py (function: run_fusion_poe)\n\n"
            "For now, run the notebook: notebooks/04_fusion_poe.ipynb\n"
        )
        sys.exit(2)

    run_fusion_poe(
        image_split_dir=image_split_dir,
        vision_models_dir=vision_models_dir,
        vision_temps_dir=vision_temps_dir,
        hsi_model_path=hsi_model_path,
        feature_names_path=feature_names_path,
        species_mapping_path=species_mapping_path,
        out_fig_dir=fig_dir,
        out_met_dir=met_dir,
        out_pred_dir=prd_dir,
        split=args.split,
    )


if __name__ == "__main__":
    main()
