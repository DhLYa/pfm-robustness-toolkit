"""Constraint-sweep evaluation (dissertation section 3.2).

Each perturbation's strength is swept from "no perturbation" to its maximum, recording
adversarial ROC-AUC per step and run. Writes, per model,
``Stats/<dataset>/<model>/constraint_sweep/<model>_<perturbation>_sweep.xlsx`` (+ plot), and
per dataset ``Stats/<dataset>/constraint_sweep/<perturbation>_aggregate.png``.
"""

import copy
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .datasets import DATASETS, build_dataset
from .models import MODELS, get_transform, load_for_eval, model_colour
from .paths import default_device, stats_dir
from .perturbations import SWEEPS
from .reet.evaluator import Evaluator
from .reet.metrics import adversarial_roc_auc

# Perturbations whose strength *decreases* along the sweep range.
DECREASING = ["crop", "jpeg", "zoom_out"]

X_LABELS = {
    "pixel": "L2 Constraint",
    "stain": "L2 Constraint",
    "random_stain": "Weight Range",
    "mean": "Brightness Shift",
    "hed": "HED α",
    "jpeg": "JPEG Quality",
    "blur": "Gaussian σ",
    "rotate": "Rotation Angle (°)",
    "zoom_in": "Zoom In Scale",
    "zoom_out": "Zoom Out Scale",
    "crop": "Crop Size (px)",
}


def apply_sweep_value(hyp: dict, param: str, v):
    """Set one sweep value `v` of `param` in a copy of the optimiser hyperparameters."""
    if param == "sigma" and "weight_ranges" in hyp and "sigma" in hyp["weight_ranges"]:
        # Blur: fixed sigma, kernel ~6σ+1 (odd), applied to the whole 224x224 image
        sigma_val = float(v)
        hyp["weight_ranges"]["sigma"] = (sigma_val, sigma_val)
        kernel = int(math.ceil(6 * sigma_val + 1))
        if kernel % 2 == 0:
            kernel += 1
        hyp["weight_ranges"]["kernel_size"] = (kernel, kernel)
        hyp["weight_ranges"]["corner_x"] = (0, 0)
        hyp["weight_ranges"]["corner_y"] = (0, 0)
        hyp["weight_ranges"]["height"] = (224, 224)
        hyp["weight_ranges"]["width"] = (224, 224)
    elif param == "angle" and "weight_ranges" in hyp and "angle" in hyp["weight_ranges"]:
        hyp["weight_ranges"]["angle"] = (v, v)
    elif param == "scale" and "weight_ranges" in hyp and "scale" in hyp["weight_ranges"]:
        hyp["weight_ranges"]["scale"] = (v, v)
    elif param == "height" and "weight_ranges" in hyp and "height" in hyp["weight_ranges"]:
        # Crop: square crop of side v
        hyp["weight_ranges"]["height"] = (v, v)
        if "width" in hyp["weight_ranges"]:
            hyp["weight_ranges"]["width"] = (v, v)
    elif param in hyp:
        hyp[param] = v
    elif "weight_ranges" in hyp and param in hyp["weight_ranges"]:
        if param == "quality":
            hyp["weight_ranges"][param] = (v, v)
        else:
            hyp["weight_ranges"][param] = (-v, v)
    else:
        raise KeyError(f"Parameter '{param}' not found in attack.hyperparameters nor in weight_ranges.")
    return hyp


def sweep_constraint(evaluator, param: str, param_range: tuple, step_size: float, metric,
                     adversarial: bool = True, runs: int = 3) -> pd.DataFrame:
    """Columns: param, <metric name>, std (ddof=1), run_1..run_n."""
    start, end = param_range
    param_values = np.arange(start, end + step_size, step_size)
    all_runs = []
    original_hyp = copy.deepcopy(evaluator.attack.hyperparameters)

    for run_idx in range(runs):
        print(f"Run {run_idx+1}/{runs} sweeping '{param}'...")
        scores = []
        for i, v in enumerate(param_values):
            print(f"  value {i+1}/{len(param_values)} == {v}")
            evaluator.attack.hyperparameters = apply_sweep_value(copy.deepcopy(original_hyp), param, v)
            scores.append(evaluator.compute_metric(metric, adversarial=adversarial))
        all_runs.append(scores)

    evaluator.attack.hyperparameters = original_hyp
    arr = np.vstack(all_runs)
    df = pd.DataFrame({
        'param': param_values,
        metric.__name__: arr.mean(axis=0),
        'std': arr.std(axis=0, ddof=1),
    })
    for i in range(runs):
        df[f'run_{i+1}'] = all_runs[i]
    return df


def plot_with_error_band(df, model_name: str, perturbation: str, param_name: str,
                         y_col: str, std_col: str = "std"):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=df, x="param", y=y_col, ax=ax, marker="o")
    ax.fill_between(df["param"], df[y_col] - df[std_col], df[y_col] + df[std_col],
                    alpha=0.2, color=sns.color_palette()[0])
    ax.set_xlabel(f"Constraint ({param_name.capitalize()})")
    if perturbation in DECREASING:
        ax.invert_xaxis()
    title = "brightness" if perturbation == "mean" else perturbation
    ax.set_title(f"{model_name} - {title} sweep")
    ax.grid(True)
    fig.tight_layout()
    return fig


def batch_sweep(model, model_name, dataset, dataloader, sweeps: dict, metric, runs, device,
                output_dir, post_transform=None):
    """Run every sweep in `sweeps` for one model; write xlsx + png per perturbation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, (name, sw) in enumerate(sweeps.items()):
        print(f"\n=== Sweeping {i+1} / {len(sweeps)} {name} on model {model_name} ===")
        p = sw.perturbation
        evaluator = Evaluator(
            model, model_name, dataset, dataloader,
            p.optimiser, p.transform, p.optimiser_params, p.transform_params,
            device=device, post_transform=post_transform,
        )
        df = sweep_constraint(evaluator, sw.param, sw.range, sw.step_size, metric,
                              adversarial=True, runs=runs)
        df.to_excel(output_dir / f"{model_name}_{name}_sweep.xlsx", index=False)

        fig = plot_with_error_band(df, model_name, name, sw.param, y_col=metric.__name__)
        out_path = output_dir / f"{model_name}_{name}.png"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot -> {out_path}")


def run_sweeps(dataset: str, model_names, sweeps=None, runs: int = 3, metric=adversarial_roc_auc,
               device=None, test_multiplier=None, batch_size=None):
    """Constraint sweeps for each model (with its head trained on `dataset`)."""
    device = device or default_device()
    sweeps = sweeps or SWEEPS
    spec = DATASETS[dataset]
    budget = spec.budgets["sweep"]
    test_data, test_loader = build_dataset(
        spec.cls,
        root_dir=spec.test_path(),
        batch_size=budget["batch_size"] if batch_size is None else batch_size,
        test_multiplier=budget["test_multiplier"] if test_multiplier is None else test_multiplier,
        transform=spec.tensor_transform(),
        shuffle=False,
        **spec.test_kwargs,
    )
    print(f"Test batches: {len(test_loader)}")

    for name in model_names:
        print(f"----------------------Beginning {name} sweep----------------------")
        model = load_for_eval(name, dataset, device)
        batch_sweep(
            model=model,
            model_name=name,
            dataset=test_data,
            dataloader=test_loader,
            sweeps=sweeps,
            metric=metric,
            runs=runs,
            device=device,
            output_dir=stats_dir(dataset, name, "constraint_sweep"),
            post_transform=get_transform(name, model),
        )


def aggregate_constraint_sweeps(dataset: str, model_names=None, perturbations=None, normalize_x=True):
    """One plot per perturbation with every model's sweep curve, x normalised to 0-1."""
    model_names = list(model_names or MODELS)
    perturbations = list(perturbations or SWEEPS)
    base_dir = stats_dir(dataset)
    output_dir = stats_dir(dataset, "constraint_sweep")
    output_dir.mkdir(parents=True, exist_ok=True)
    palette = {m: model_colour(m) for m in model_names}

    for perturb in perturbations:
        df_list = []
        y_col = None
        for model in model_names:
            path = base_dir / model / "constraint_sweep" / f"{model}_{perturb}_sweep.xlsx"
            if not path.is_file():
                print(f"Missing file for {model}/{perturb}: {path}")
                continue
            df = pd.read_excel(path)
            df['model'] = model
            df_list.append(df)
            if y_col is None:
                y_col = df.columns[1]

        if not df_list or y_col is None:
            print(f"No data found for perturb '{perturb}', skipping.")
            continue

        agg = pd.concat(df_list, ignore_index=True)
        x_col = agg.columns[0]

        if normalize_x:
            x_vals = agg[x_col].values
            min_x, max_x = x_vals.min(), x_vals.max()
            if max_x > min_x:
                if perturb in DECREASING:
                    agg[x_col] = (max_x - x_vals) / (max_x - min_x)
                else:
                    agg[x_col] = (x_vals - min_x) / (max_x - min_x)
            else:
                agg[x_col] = 0.0

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(data=agg, x=x_col, y=y_col, hue='model', linewidth=3, palette=palette, ax=ax)
        ax.set_xlabel("Normalized Constraint" if normalize_x else X_LABELS.get(perturb, "Constraint"))
        ax.set_ylabel(y_col.replace("_", " ").capitalize())
        title = "Brightness" if perturb == "mean" else perturb.capitalize()
        ax.set_title(f"{dataset} {title} sweep")
        ax.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        ax.grid(True)
        fig.tight_layout()

        out_path = output_dir / f"{perturb}_aggregate.png"
        fig.savefig(out_path)
        plt.close(fig)
        print(f"Saved aggregate plot: {out_path}")
