from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import os
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def _read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML in {path} (expected a mapping).")
    return data


def _abs_from_repo(p: str | Path) -> Path:
    p = Path(p).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (REPO_ROOT / p).resolve()


def _env_or_cfg(env: str | None, cfg_value: str | None) -> Path:
    if env:
        v = os.getenv(env)
        if v:
            return _abs_from_repo(v)
    if cfg_value is None:
        raise KeyError("Missing config value and no env override was provided.")
    return _abs_from_repo(cfg_value)


def load_paths(config_path: str | Path = "configs/paths.yaml") -> Dict[str, Path]:
    """
    Load canonical project paths (absolute Paths), anchored to repo root.

    Supports environment variable overrides for portability:
      PROJECT_ROOT, DATA_ROOT, IMAGE_ROOT, MODELS_ROOT, OUTPUTS_ROOT, ARTIFACTS_ROOT
    """
    cfg_path = Path(config_path).expanduser()
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path
    cfg = _read_yaml(cfg_path.resolve())

    project = cfg.get("project", {}) or {}
    data = cfg.get("data", {}) or {}
    models = cfg.get("models", {}) or {}
    outputs = cfg.get("outputs", {}) or {}
    artifacts = cfg.get("artifacts", {}) or {}

    # Roots (with env overrides)
    PROJECT_ROOT = _env_or_cfg("PROJECT_ROOT", project.get("project_root", "."))
    DATA_ROOT = _env_or_cfg("DATA_ROOT", "data")  # optional generic root
    IMAGE_ROOT = _env_or_cfg("IMAGE_ROOT", data.get("image_root"))
    MODELS_ROOT = _env_or_cfg("MODELS_ROOT", models.get("root"))
    OUTPUTS_ROOT = _env_or_cfg("OUTPUTS_ROOT", outputs.get("root"))
    ARTIFACTS_ROOT = _env_or_cfg("ARTIFACTS_ROOT", artifacts.get("root"))

    # Data splits
    GBIF_TRAIN_DIR = _env_or_cfg("GBIF_TRAIN_DIR", data.get("gbif_train_dir"))
    GBIF_VAL_DIR = _env_or_cfg("GBIF_VAL_DIR", data.get("gbif_val_dir"))

    IMAGE_TRAIN_DIR = IMAGE_ROOT / data.get("image_train", "train")
    IMAGE_VAL_DIR = IMAGE_ROOT / data.get("image_val", "val")
    IMAGE_TEST_DIR = IMAGE_ROOT / data.get("image_test", "test")
    IMAGE_TEST2_DIR = IMAGE_ROOT / data.get("image_test2", "test2")

    # Model dirs
    CONTEXT_MODEL_DIR = _env_or_cfg("CONTEXT_MODEL_DIR", models.get("context_dir"))
    VISION_MODEL_DIR = _env_or_cfg("VISION_MODEL_DIR", models.get("vision_dir"))
    VISION_TEMPS_DIR = _env_or_cfg("VISION_TEMPS_DIR", models.get("vision_temps_dir"))

    # Output dirs
    OUT_CONTEXT_XGB = _env_or_cfg("OUT_CONTEXT_XGB", outputs.get("context_xgb"))
    OUT_VISION_RESNET = _env_or_cfg("OUT_VISION_RESNET", outputs.get("vision_resnet"))
    OUT_FUSION_POE = _env_or_cfg("OUT_FUSION_POE", outputs.get("fusion_poe"))

    # Backwards-compatible aliases (so older scripts don't break)
    # These map to the most sensible modern equivalents:
    GBIF_DIR = GBIF_TRAIN_DIR
    IMAGE_DIR = IMAGE_ROOT
    MODEL_DIR = MODELS_ROOT

    return {
        # roots
        "REPO_ROOT": PROJECT_ROOT,
        "DATA_ROOT": DATA_ROOT,
        "IMAGE_ROOT": IMAGE_ROOT,
        "MODELS_ROOT": MODELS_ROOT,
        "OUTPUTS_ROOT": OUTPUTS_ROOT,
        "ARTIFACTS_ROOT": ARTIFACTS_ROOT,

        # data
        "GBIF_TRAIN_DIR": GBIF_TRAIN_DIR,
        "GBIF_VAL_DIR": GBIF_VAL_DIR,
        "IMAGE_TRAIN_DIR": IMAGE_TRAIN_DIR,
        "IMAGE_VAL_DIR": IMAGE_VAL_DIR,
        "IMAGE_TEST_DIR": IMAGE_TEST_DIR,
        "IMAGE_TEST2_DIR": IMAGE_TEST2_DIR,

        # models
        "CONTEXT_MODEL_DIR": CONTEXT_MODEL_DIR,
        "VISION_MODEL_DIR": VISION_MODEL_DIR,
        "VISION_TEMPS_DIR": VISION_TEMPS_DIR,

        # outputs
        "OUT_CONTEXT_XGB": OUT_CONTEXT_XGB,
        "OUT_VISION_RESNET": OUT_VISION_RESNET,
        "OUT_FUSION_POE": OUT_FUSION_POE,

        # legacy aliases
        "GBIF_DIR": GBIF_DIR,
        "IMAGE_DIR": IMAGE_DIR,
        "MODEL_DIR": MODEL_DIR,
    }
