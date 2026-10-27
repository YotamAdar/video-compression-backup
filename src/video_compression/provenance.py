"""Lightweight provenance capture without importing ML dependencies."""

import hashlib
from importlib.metadata import PackageNotFoundError, version
import os
from pathlib import Path
import platform
import subprocess


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "--no-optional-locks", "-C", str(root), *args],
        capture_output=True, text=True, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def source_identity(root: Path) -> dict:
    paths = sorted(root.glob("*.py")) + sorted((root / "turbo_ddcm").rglob("*.py"))
    return {
        "root": str(root.resolve()),
        "commit": git(root, "rev-parse", "HEAD"),
        "status": git(root, "status", "--short"),
        "diff": git(root, "diff", "HEAD", "--ignore-space-at-eol"),
        "source_sha256": {str(p.relative_to(root)): sha256(p) for p in paths},
    }


def environment() -> dict:
    packages = {}
    for name in ("torch", "torchvision", "diffusers", "transformers", "accelerate", "numpy", "Pillow", "huggingface-hub"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "packages": packages,
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        "slurm": {k: os.environ[k] for k in ("SLURM_JOB_ID", "SLURM_JOB_NODELIST", "CUDA_VISIBLE_DEVICES") if k in os.environ},
    }
