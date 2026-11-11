"""Shared helpers: config loading, paths, seeding."""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "config" / "config.yaml"
    with open(path) as fh:
        return yaml.safe_load(fh)


def p(rel: str) -> Path:
    """Resolve a repo-relative path and make sure its parent exists."""
    out = ROOT / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
