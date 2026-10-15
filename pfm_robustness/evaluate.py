"""Fixed-parameter robustness evaluation (dissertation section 3.1).

For each model, every perturbation in FIXED_EVAL is run `runs` times on a stratified test
subset. Writes ``Stats/<dataset>/<model>/<model>_stats.xlsx`` and saves example
misclassifications under ``Stats/<dataset>/<model>/<perturbation>/``.
"""

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from .datasets import DATASETS, dataset_labels, stratified_subset
from .models import get_transform, load_for_eval
from .paths import default_device, stats_dir
from .perturbations import FIXED_EVAL
from .reet.evaluator import Evaluator
from .reet.metrics import get_metrics

METRIC_NAMES = [
    "Accuracy", "ROC AUC", "PR AUC",
    "Robustness Accuracy", "Fooling Ratio", "Input Sensitivity",
    "Adversarial ROC AUC", "Adversarial PR AUC",
]  # same order as reet.metrics.get_metrics
MEAN_ONLY_METRICS = ["Accuracy", "ROC AUC", "PR AUC"]
METRIC_RENAME = {
    "Accuracy": "Baseline Accuracy",
    "ROC AUC": "Baseline ROC",
    "PR AUC": "Baseline PR",
    "Robustness Accuracy": "Accuracy After Perturbation",
}


def evaluate_perturbation(model, model_name, perturbation_name, dataset, dataloader, save_root,
                          post_transform=None, device="cuda:0"):
    """One run of one perturbation -> tuple of METRIC_NAMES values."""
    p = FIXED_EVAL[perturbation_name]
    evaluator = Evaluator(
        model, model_name,
        dataset, dataloader,
        p.optimiser, p.transform,
        p.optimiser_params, p.transform_params,
        device=device,
        save_dir=str(save_root / model_name / perturbation_name),
        post_transform=post_transform,
    )
    return get_metrics(evaluator.predict(adversarial=True))


def collect_model_stats(model, model_name, test_data, test_loader, save_root, runs=3,
                        perturbations=None, post_transform=None, device="cuda:0") -> pd.DataFrame:
    """Rows = perturbations + "Mean"; columns = (model, metric, Mean/STD)."""
    perturbations = list(perturbations or FIXED_EVAL)
    std_metrics = [m for m in METRIC_NAMES if m not in MEAN_ONLY_METRICS]

    col_tuples = []
    for metric in METRIC_NAMES:
        renamed = METRIC_RENAME.get(metric, metric)
        col_tuples.append((model_name, renamed, "Mean"))
        if metric in std_metrics:
            col_tuples.append((model_name, renamed, "STD"))
    columns = pd.MultiIndex.from_tuples(col_tuples, names=["Model", "Metric", "Statistic"])
    stats_df = pd.DataFrame(index=perturbations, columns=columns, dtype=float)

    for i, pert in enumerate(perturbations, start=1):
        print(f"  {i}/{len(perturbations)}  •  '{pert}'")
        raw = {m: [] for m in METRIC_NAMES}
        for _ in range(runs):
            vals = evaluate_perturbation(model, model_name, pert, test_data, test_loader, save_root,
                                         post_transform=post_transform, device=device)
            for m, v in zip(METRIC_NAMES, vals):
                raw[m].append(v)

        for metric in METRIC_NAMES:
            renamed = METRIC_RENAME.get(metric, metric)
            stats_df.loc[pert, (model_name, renamed, "Mean")] = np.mean(raw[metric])
            if metric in std_metrics:
                stats_df.loc[pert, (model_name, renamed, "STD")] = np.std(raw[metric])

    # Row of averages across perturbations
    mean_row = {}
    for metric in METRIC_NAMES:
        renamed = METRIC_RENAME.get(metric, metric)
        mean_row[(model_name, renamed, "Mean")] = stats_df.loc[:, (model_name, renamed, "Mean")].mean()
        if metric in std_metrics:
            mean_row[(model_name, renamed, "STD")] = stats_df.loc[:, (model_name, renamed, "STD")].mean()
    stats_df.loc["Mean"] = pd.Series(mean_row)
    return stats_df.reindex(perturbations + ["Mean"])


def run_fixed_eval(dataset: str, model_names, runs: int = 3, perturbations=None, device=None,
                   test_multiplier=None, batch_size=None):
    """Evaluate each model (with its head trained on `dataset`) on every perturbation."""
    device = device or default_device()
    spec = DATASETS[dataset]
    budget = spec.budgets["eval"]
    test_multiplier = budget["test_multiplier"] if test_multiplier is None else test_multiplier
    batch_size = budget["batch_size"] if batch_size is None else batch_size
    root = spec.test_path()
    save_root = stats_dir(dataset)

    # Fix the stratified test subset once, so every model sees the same images
    label_ds = spec.cls(root, transform=transforms.ToTensor(), **spec.test_kwargs)
    test_ix = stratified_subset(list(range(len(label_ds))), dataset_labels(label_ds), test_multiplier)

    # Attack in raw 0-1 pixel space; model normalisation is applied after the attack
    base_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    test_full = spec.cls(root, transform=base_transform, **spec.test_kwargs)
    test_subset = Subset(test_full, test_ix)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    for name in model_names:
        print(f"\n────────  {name}  ─ collecting statistics ────────")
        model = load_for_eval(name, dataset, device)
        stats_df = collect_model_stats(
            model=model,
            model_name=name,
            test_data=test_subset,
            test_loader=test_loader,
            save_root=save_root,
            runs=runs,
            perturbations=perturbations,
            post_transform=get_transform(name, model),
            device=device,
        )
        xlsx = save_root / name / f"{name}_stats.xlsx"
        xlsx.parent.mkdir(parents=True, exist_ok=True)
        stats_df.to_excel(xlsx)
        print(f"✔ Metrics saved →  {xlsx}")
