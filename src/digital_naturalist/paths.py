from __future__ import annotations
from pathlib import Path
import os
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

def load_paths(config_path: str | Path = "configs/paths.yaml") -> dict[str, Path]:
    cfg_file = (REPO_ROOT / config_path).resolve()
    with open(cfg_file, "r") as f:
        cfg = yaml.safe_load(f)

    p = cfg.get("project", {})

    def resolve(key: str, env: str | None = None) -> Path:
        # Allow override via environment variables (useful on other machines)
        if env and os.getenv(env):
            return Path(os.getenv(env)).expanduser().resolve()
        return (REPO_ROOT / p[key]).resolve()

    return {
        "GBIF_DIR": resolve("gbif_dir", "GBIF_DIR"),
        "IMAGE_DIR": resolve("image_dir", "IMAGE_DIR"),
        "MODEL_DIR": resolve("model_dir", "MODEL_DIR"),
    }
