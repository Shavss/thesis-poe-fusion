#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from digital_naturalist.paths import load_paths


def _require_dir(p: Path, label: str) -> None:
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"{label} not found or not a directory: {p}")


def _mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Train ResNet-18 visual expert (10 seeds) and save artifacts + temperatures."
    )
    ap.add_argument("--config", default="configs/paths.yaml", help="Path to paths.yaml (repo-relative or absolute).")
    ap.add_argument("--dry-run", action="store_true", help="Print resolved paths and exit.")
    ap.add_argument("--runs", type=int, default=10, help="Number of runs (default: 10).")
    ap.add_argument("--base-seed", type=int, default=42, help="Base seed (default: 42).")
    ap.add_argument("--epochs", type=int, default=20, help="Epochs (default: 20).")
    ap.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32).")
    ap.add_argument("--test-split", default="test2", choices=["test", "test2"], help="Temporal holdout split.")
    args = ap.parse_args()

    P = load_paths(args.config)

    image_root: Path = P["IMAGE_ROOT"]
    image_train: Path = P["IMAGE_TRAIN_DIR"]
    image_val: Path = P["IMAGE_VAL_DIR"]
    image_test: Path = P["IMAGE_TEST2_DIR"] if args.test_split == "test2" else P["IMAGE_TEST_DIR"]

    vision_models_dir: Path = P["VISION_MODEL_DIR"]
    vision_temps_dir: Path = P["VISION_TEMPS_DIR"]

    out_vision: Path = P["OUT_VISION_RESNET"]
    fig_dir: Path = out_vision / "figures"
    met_dir: Path = out_vision / "metrics"
    prd_dir: Path = out_vision / "preds"
    log_dir: Path = (P.get("OUTPUTS_ROOT") or out_vision.parent) / "logs"

    print("\n=== Visual expert (ResNet) — resolved paths ===")
    print("Config:           ", Path(args.config))
    print("Image root:       ", image_root)
    print("Train images:     ", image_train)
    print("Val images:       ", image_val)
    print("Test images:      ", image_test)
    print("Vision models:    ", vision_models_dir)
    print("Temperatures:     ", vision_temps_dir)
    print("Outputs/figures:  ", fig_dir)
    print("Outputs/metrics:  ", met_dir)
    print("Outputs/preds:    ", prd_dir)
    print("Logs:             ", log_dir)
    print("Runs / seed:      ", args.runs, "/", args.base_seed)
    print("Epochs / batch:   ", args.epochs, "/", args.batch_size)
    print("Holdout split:    ", args.test_split)
    print("============================================\n")

    if args.dry_run:
        return

    # Validate inputs
    _require_dir(image_train, "Train image folder")
    _require_dir(image_val, "Val image folder")
    _require_dir(image_test, "Test image folder")

    # Create outputs
    for d in (vision_models_dir, vision_temps_dir, fig_dir, met_dir, prd_dir, log_dir):
        _mkdir(d)

    # ------------------------------------------------------------------
    # Hook for the real training code
    # ------------------------------------------------------------------
    # Best practice is to move the training loop from notebooks/03_vision_resnet.ipynb
    # into a function in src/, e.g.:
    #   from digital_naturalist.vision.train import run_resnet_experiments
    # Then call it here.

    try:
        from digital_naturalist.vision.train import run_resnet_experiments  # type: ignore
    except Exception:
        print(
            "This CLI wrapper is ready, but the training implementation is still in the notebook.\n"
            "Next step (recommended): move the training logic from notebooks/03_vision_resnet.ipynb into\n"
            "  src/digital_naturalist/vision/train.py  (function: run_resnet_experiments)\n\n"
            "For now, run the notebook: notebooks/03_vision_resnet.ipynb\n"
        )
        sys.exit(2)

    run_resnet_experiments(
        image_root=image_root,
        out_fig_dir=fig_dir,
        out_met_dir=met_dir,
        out_pred_dir=prd_dir,
        vision_models_dir=vision_models_dir,
        vision_temps_dir=vision_temps_dir,
        num_runs=args.runs,
        base_seed=args.base_seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        test_split=args.test_split,
    )


if __name__ == "__main__":
    main()
