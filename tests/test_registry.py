"""Cheap consistency checks for the registries. Run with `pytest tests/` (no GPU or downloads needed)."""

import pytest

from pfm_robustness.datasets import DATASETS
from pfm_robustness.embedders import KFOLD_EMBEDDERS, SIMILARITY_BACKBONES, _PIL_TRANSFORMS
from pfm_robustness.models import MODELS
from pfm_robustness.perturbations import FIXED_EVAL, SWEEPS

STAGE_KEYS = {
    "train": {"trainval_multiplier", "trainval_size", "batch_size"},
    "eval": {"test_multiplier", "batch_size"},
    "sweep": {"test_multiplier", "batch_size"},
    "embed": {"trainval_multiplier", "trainval_size", "test_multiplier", "batch_size"},
    "kfold": {"trainval_multiplier", "trainval_size", "test_multiplier", "batch_size"},
}


@pytest.mark.parametrize("name", list(MODELS))
def test_model_spec_complete(name):
    spec = MODELS[name]
    assert spec.name == name
    assert callable(spec.build) and callable(spec.normalisation)
    assert spec.params_m > 0
    assert spec.weight_key and " " not in spec.weight_key


def test_weight_keys_unique():
    keys = [s.weight_key for s in MODELS.values()]
    assert len(keys) == len(set(keys))


@pytest.mark.parametrize("name", list(DATASETS))
def test_dataset_budgets(name):
    spec = DATASETS[name]
    for stage, keys in STAGE_KEYS.items():
        assert stage in spec.budgets, f"{name} has no '{stage}' budget"
        assert keys <= set(spec.budgets[stage]), f"{name}/{stage} missing {keys - set(spec.budgets[stage])}"
        for k, v in spec.budgets[stage].items():
            assert v > 0, f"{name}/{stage}/{k} must be positive"


@pytest.mark.parametrize("name", list(DATASETS))
def test_dataset_folders_exist(name):
    spec = DATASETS[name]
    from pathlib import Path
    train, test = Path(spec.train_path()), Path(spec.test_path())
    if not train.parent.exists():
        pytest.skip("Data/ not present on this machine")
    assert train.is_dir(), f"missing {train}"
    assert test.is_dir(), f"missing {test}"


def test_sweeps_use_known_perturbations():
    for name, sw in SWEEPS.items():
        assert sw.perturbation in FIXED_EVAL.values(), name
        assert sw.steps >= 2
        assert sw.range[0] != sw.range[1]


def test_embedding_registries_cover_models():
    for name in MODELS:
        assert name in KFOLD_EMBEDDERS, f"{name} missing from KFOLD_EMBEDDERS"
        assert name in SIMILARITY_BACKBONES, f"{name} missing from SIMILARITY_BACKBONES"
        assert name in _PIL_TRANSFORMS, f"{name} missing from embedders._PIL_TRANSFORMS"
