"""Robustness benchmarking of pathology foundation models, built on REET.

Registries (edit these to add models / datasets):
    pfm_robustness.models.MODELS
    pfm_robustness.datasets.DATASETS
    pfm_robustness.perturbations.FIXED_EVAL, SWEEPS
    pfm_robustness.embedders.KFOLD_EMBEDDERS, SIMILARITY_BACKBONES
"""

from .datasets import DATASETS
from .models import MODELS
from .paths import default_device, project_root, stats_dir, storage_root
from .perturbations import FIXED_EVAL, SWEEPS

__all__ = ["DATASETS", "MODELS", "FIXED_EVAL", "SWEEPS",
           "default_device", "project_root", "stats_dir", "storage_root"]
