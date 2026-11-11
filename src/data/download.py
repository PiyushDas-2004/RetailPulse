"""Download the M5 (Walmart) dataset without Kaggle credentials.

Usage:  python -m src.data.download

WHY this source: the `datasetsforecast` package (Nixtla) mirrors the official M5
competition files in a public GitHub repo, so no Kaggle login is required.
We download the same zip that `datasetsforecast.m5.M5.download` uses, verify its
SHA-256, and extract it to data/raw/m5 (git-ignored; the raw CSVs are ~490 MB).

Two routes are tried in order:
  1. plain HTTPS download of the raw file URL (what works on a normal machine);
  2. a sparse, shallow `git clone` of only the `datasets/` folder (works behind
     proxies that block raw-file URLs but allow git, as in the build environment).
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import requests

from src.utils import ROOT, load_config

EXPECTED_FILES = [
    "calendar.csv",
    "sales_train_evaluation.csv",
    "sell_prices.csv",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _via_http(url: str, dest: Path) -> bool:
    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(dest, "wb") as fh:
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk)
        return True
    except Exception as exc:  # noqa: BLE001 - we fall back below
        print(f"[download] HTTP route failed: {exc}")
        return False


def _via_git(repo: str, dest: Path) -> bool:
    tmp = Path(tempfile.mkdtemp())
    try:
        env = {"GIT_LFS_SKIP_SMUDGE": "1"}
        import os

        env = {**os.environ, **env}
        subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", repo, str(tmp / "r")],
                       check=True, env=env)
        subprocess.run(["git", "-C", str(tmp / "r"), "sparse-checkout", "set", "datasets"], check=True, env=env)
        shutil.copy(tmp / "r" / "datasets" / "m5.zip", dest)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[download] git route failed: {exc}")
        return False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    cfg = load_config()
    raw_dir = ROOT / cfg["paths"]["raw_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    if all((raw_dir / f).exists() for f in EXPECTED_FILES):
        print(f"[download] data already present in {raw_dir}")
        return
    zpath = raw_dir / "m5.zip"
    if not zpath.exists():
        ok = _via_http(cfg["data"]["source_url"], zpath) or _via_git(cfg["data"]["git_fallback_repo"], zpath)
        if not ok:
            sys.exit("[download] could not obtain m5.zip by HTTP or git; check network access.")
    digest = sha256(zpath)
    if digest != cfg["data"]["sha256"]:
        sys.exit(f"[download] checksum mismatch: {digest}")
    print(f"[download] sha256 OK ({digest[:12]}...)")
    with zipfile.ZipFile(zpath) as zf:
        zf.extractall(raw_dir)
    zpath.unlink()
    print(f"[download] extracted to {raw_dir}: {sorted(x.name for x in raw_dir.iterdir())}")


if __name__ == "__main__":
    main()
