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
    ap = argparse.ArgumentParser(description="Train/evaluate the context expert (XGBoost HSI).")
    ap.add_argument("--config", default="configs/paths.yaml", help="Path to paths.yaml (repo-relative or absolute).")
    ap.add_argument("--dry-run", action="store_true", help="Print resolved paths and exit.")
    args = ap.parse_args()

    P = load_paths(args.config)

    gbif_train: Path = P["GBIF_TRAIN_DIR"]
    gbif_val: Path = P["GBIF_VAL_DIR"]

    context_model_dir: Path = P["CONTEXT_MODEL_DIR"]
    out_ctx: Path = P["OUT_CONTEXT_XGB"]
    fig_dir: Path = out_ctx / "figures"
    met_dir: Path = out_ctx / "metrics"
    prd_dir: Path = out_ctx / "preds"
    log_dir: Path = (P.get("OUTPUTS_ROOT") or out_ctx.parent) / "logs"

    print("\n=== Context expert (XGBoost) — resolved paths ===")
    print("Config:           ", Path(args.config))
    print("GBIF train:       ", gbif_train)
    print("GBIF val:         ", gbif_val)
    print("Context models:   ", context_model_dir)
    print("Outputs/figures:  ", fig_dir)
    print("Outputs/metrics:  ", met_dir)
    print("Outputs/preds:    ", prd_dir)
    print("Logs:             ", log_dir)
    print("===============================================\n")

    if args.dry_run:
        return

    _require_dir(gbif_train, "GBIF train folder")
    _require_dir(gbif_val, "GBIF val folder")

    for d in (context_model_dir, fig_dir, met_dir, prd_dir, log_dir):
        _mkdir(d)

    # ------------------------------------------------------------------
    # Hook for the real context training code
    # ------------------------------------------------------------------
    try:
        from digital_naturalist.context.train import run_context_xgboost  # type: ignore
    except Exception:
        print(
            "This CLI wrapper is ready, but the context training implementation is still in notebooks.\n"
            "Next step (recommended): move the logic from notebooks/01_env_xgboost.ipynb into\n"
            "  src/digital_naturalist/context/train.py (function: run_context_xgboost)\n\n"
            "For now, run the notebooks:\n"
            "  - notebooks/01_env_xgboost.ipynb\n"
            "  - notebooks/02_env_xgboost_analysis.ipynb\n"
        )
        sys.exit(2)

    run_context_xgboost(
        gbif_train_dir=gbif_train,
        gbif_val_dir=gbif_val,
        out_fig_dir=fig_dir,
        out_met_dir=met_dir,
        out_pred_dir=prd_dir,
        context_model_dir=context_model_dir,
    )


if __name__ == "__main__":
    main()
