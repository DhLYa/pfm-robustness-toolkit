"""Where code, data, checkpoints and results live.

Two roots:

* the **project root** – this repository (contains ``pyproject.toml`` and ``third_party/``);
* the **storage root** – the folder holding ``Data/``, ``Models/`` and ``Stats/``.

By default they are the same folder. On a cluster where data lives elsewhere, set
``PFM_STORAGE=/path/to/storage`` (and optionally ``PFM_ROOT`` for the code).
"""

import os
import sys
from pathlib import Path

import torch


def project_root() -> Path:
    env = os.environ.get("PFM_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve().parent
    for start in (Path.cwd(), here):
        for p in (start, *start.parents):
            if (p / "pyproject.toml").is_file() and (p / "pfm_robustness").is_dir():
                return p
    return here.parent


def storage_root() -> Path:
    env = os.environ.get("PFM_STORAGE")
    return Path(env).expanduser().resolve() if env else project_root()


def _existing(root: Path, *names: str) -> Path:
    """First of `names` that exists under `root` (handles Data/ vs data/), else the first name."""
    for n in names:
        if (root / n).exists():
            return root / n
    return root / names[0]


def data_dir(*parts) -> Path:
    return _existing(storage_root(), "Data", "data").joinpath(*parts)


def models_dir(*parts) -> Path:
    return _existing(storage_root(), "Models", "models").joinpath(*parts)


def stats_dir(*parts) -> Path:
    return _existing(storage_root(), "Stats", "stats").joinpath(*parts)


def ensure_dir(path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_exaonepath():
    """Make EXAONEPath's `vision_transformer` module importable (it is not on PyPI)."""
    for root in (project_root(), storage_root()):
        candidate = root / "third_party" / "EXAONEPath"
        if (candidate / "vision_transformer.py").is_file():
            if str(candidate) not in sys.path:
                sys.path.append(str(candidate))
            return candidate
    raise FileNotFoundError(
        "third_party/EXAONEPath not found. Clone https://github.com/LG-AI-EXAONE/EXAONEPath into it."
    )


def default_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
