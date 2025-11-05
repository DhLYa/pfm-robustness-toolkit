import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import copy
from reetoolbox.image_evaluator import Evaluator
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ProcessorTransform:
    def __init__(self, processor):
        self.processor = processor
    def __call__(self, img):
        import torch
        # Disable spatial ops; skip extra 1/255 when we already have float tensors
        extra = dict(do_resize=False, do_center_crop=False)
        if isinstance(img, torch.Tensor) and img.dtype.is_floating_point:
            extra["do_rescale"] = False
        out = self.processor(img, return_tensors="pt", **extra)["pixel_values"]
        # Keep batch dim if present
        return out if out.ndim == 4 else out.squeeze(0)


from torchvision import transforms
from timm.data import resolve_data_config, create_transform

import torch
from transformers import AutoImageProcessor

# For torch/timm models
from torchvision import transforms

# Tensor-only normalisation (no ToTensor, no resize/crop)
uni_transform      = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
giga_transform     = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
exaone_transform   = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
virchow_transform  = transforms.Normalize((0.4850, 0.4560, 0.4060), (0.2290, 0.2240, 0.2250))
hoptimus_transform = transforms.Normalize((0.707223, 0.578729, 0.703617), (0.211883, 0.230117, 0.177517))

# ResNets already good:
resnet_transform   = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

from timm.data import resolve_data_config

def timm_transform(model):
    # Find the timm module that has pretrained_cfg
    if hasattr(model, "pretrained_cfg"):
        timm_model = model
    elif hasattr(model, "backbone") and hasattr(model.backbone, "pretrained_cfg"):
        timm_model = model.backbone
    else:
        raise AttributeError("Cannot find .pretrained_cfg on the model or its backbone.")
    cfg = resolve_data_config(timm_model.pretrained_cfg, model=timm_model)
    mean, std = cfg["mean"], cfg["std"]
    return transforms.Normalize(mean=mean, std=std)

# For HuggingFace processor models

hibou_b_processor   = AutoImageProcessor.from_pretrained("histai/hibou-b", trust_remote_code=True)
hibou_L_processor   = AutoImageProcessor.from_pretrained("histai/hibou-L", trust_remote_code=True)
phikon_processor    = AutoImageProcessor.from_pretrained("owkin/phikon", trust_remote_code=True)
phikonv2_processor  = AutoImageProcessor.from_pretrained("owkin/phikon-v2", trust_remote_code=True)

# Wrap processor as callable
hibou_b_transform  = ProcessorTransform(hibou_b_processor)
hibou_L_transform  = ProcessorTransform(hibou_L_processor)
# phikon with upfront up-scaling
phikon_transform   = ProcessorTransform(phikon_processor)
phikonv2_transform = ProcessorTransform(phikonv2_processor)


model_transforms = {
    "ResNet18":     resnet_transform, 
    "ResNet50":     resnet_transform, 
    "UNI":        uni_transform,
    "UNI2":        uni_transform,
    "GigaPath":   giga_transform,
    "Virchow":    virchow_transform,   # special
    "Virchow2":   virchow_transform,   # special
    "EXAONEPath": exaone_transform,
    "Hibou-B":    hibou_b_transform,
    "Hibou-L":    hibou_L_transform,
    "H-Optimus-0": hoptimus_transform,
    "H-Optimus-1": hoptimus_transform,
    "H0-mini":    hoptimus_transform,
    "Phikon v1":  phikon_transform,
    "Phikon v2":  phikonv2_transform,
}

# Cell 7: get_transform that handles timm-based special cases
def get_transform(model_name, model=None):
    if model_name in {"Virchow", "Virchow2", "H0-mini"} and model is not None:
        return timm_transform(model)
    return model_transforms[model_name]

def aggregate_constraint_sweeps(models, dataset_name, perturbs,
                                base_dir=os.path.join('Stats', 'NCT-HE'),
                                output_dir=os.path.join('Stats', 'NCT-HE', 'constraint_sweep'),
                                normalize_x=True):
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    os.makedirs(output_dir, exist_ok=True)
    color_palette = sns.color_palette('tab20', n_colors=len(models))
    palette = dict(zip(models, color_palette))

    for perturb in perturbs:
        df_list = []
        y_col = None

        for model in models:
            fn = f"{model}_{perturb}_sweep.xlsx"
            path = os.path.join(base_dir, model, 'constraint_sweep', fn)
            if not os.path.isfile(path):
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
            if perturb in ["crop", "jpeg", "zoom_out"]:
                if max_x > min_x:
                    agg[x_col] = (max_x - x_vals) / (max_x - min_x)
                else:
                    agg[x_col] = 0.0
            else:
                if max_x > min_x:
                    agg[x_col] = (x_vals - min_x) / (max_x - min_x)
                else:
                    agg[x_col] = 0.0

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(
            data=agg,
            x=x_col,
            y=y_col,
            hue='model',
            linewidth=3,
            palette=palette,
            ax=ax
        )

        x_labels = {
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
            "crop": "Crop Size (px)"
        }
        if normalize_x:
            # Prepend "Normalized" for clarity
            label = x_labels.get(perturb, "Constraint")
            if not label.lower().startswith("normalized"):
                label = f"Normalized Constraint"
            ax.set_xlabel(label)
        else:
            ax.set_xlabel(x_labels.get(perturb, "Constraint"))

        ax.set_ylabel(y_col.replace("_", " ").capitalize())
        if perturb == "mean":
            ax.set_title(f"{dataset_name} Brightness sweep")
        else:
            ax.set_title(f"{dataset_name} {perturb.capitalize()} sweep")
        # ax.legend(title="Model")
        ax.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        ax.grid(True)
        fig.tight_layout()

        out_path = os.path.join(output_dir, f"{perturb}_aggregate.png")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path)
        plt.close(fig)
        print(f"Saved aggregate plot: {out_path}")

def plot_with_error_band(
    df,
    model_name: str,
    perturbation: str,
    param_name: str,
    y_col: str = "Adversarial PR_AUC",
    std_col: str = "std",
):
    x_label = f"Constraint ({param_name.capitalize()})"
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=df, x="param", y=y_col, ax=ax, marker="o")
    ax.fill_between(
        df["param"],
        df[y_col] - df[std_col],
        df[y_col] + df[std_col],
        alpha=0.2,
        color=sns.color_palette()[0]
    )

    ax.set_xlabel(x_label)
    if perturbation in ["crop", "zoom_out", "jpeg"]:
        ax.invert_xaxis()
    if perturbation == "mean":
        ax.set_title(f"{model_name} - brightness sweep")
    else:
        ax.set_title(f"{model_name} - {perturbation} sweep")
    ax.grid(True)
    fig.tight_layout()
    return fig

def sweep_constraint(
    evaluator,
    param: str,
    param_range: tuple,
    step_size: float,
    metric,
    adversarial: bool = True,
    runs: int = 3
) -> pd.DataFrame:
    import numpy as np
    import pandas as pd
    import copy
    import math

    start, end = param_range
    param_values = np.arange(start, end + step_size, step_size)
    all_runs = []
    original_hyp = copy.deepcopy(evaluator.attack.hyperparameters)

    for run_idx in range(runs):
        print(f"Run {run_idx+1}/{runs} sweeping '{param}'...")
        scores = []
        for i, v in enumerate(param_values):
            print(f"  value {i+1}/{len(param_values)} == {v}")
            hyp = copy.deepcopy(original_hyp)

            # Custom logic per perturb type
            if param == "sigma" and "weight_ranges" in hyp and "sigma" in hyp["weight_ranges"]:
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
                # ROTATE: set angle to (v, v)
                hyp["weight_ranges"]["angle"] = (v, v)
            elif param == "scale" and "weight_ranges" in hyp and "scale" in hyp["weight_ranges"]:
                # ZOOM: set scale to (v, v)
                hyp["weight_ranges"]["scale"] = (v, v)
            elif param == "height" and "weight_ranges" in hyp and "height" in hyp["weight_ranges"]:
                # CROP: set height and width to (v, v) for square crop
                hyp["weight_ranges"]["height"] = (v, v)
                if "width" in hyp["weight_ranges"]:
                    hyp["weight_ranges"]["width"] = (v, v)
            elif param in hyp:
                hyp[param] = v
            elif "weight_ranges" in hyp and param in hyp["weight_ranges"]:
                # Special-case JPEG quality (fixed, one-sided)
                if param == "quality":
                    hyp["weight_ranges"][param] = (v, v)
                else:
                    hyp["weight_ranges"][param] = (-v, v)
            else:
                raise KeyError(
                    f"Parameter '{param}' not found in attack.hyperparameters nor in weight_ranges."
                )

            evaluator.attack.hyperparameters = hyp
            score = evaluator.compute_metric(
                metric,
                adversarial=adversarial
            )
            # print(score)
            scores.append(score)
        all_runs.append(scores)

    evaluator.attack.hyperparameters = original_hyp
    arr = np.vstack(all_runs)
    mean_scores = arr.mean(axis=0)
    std_scores = arr.std(axis=0, ddof=1)
    df = pd.DataFrame({
        'param':        param_values,
        metric.__name__:   mean_scores,
        'std':    std_scores,
    })
    for i in range(runs):
        df[f'run_{i+1}'] = all_runs[i]
    return df

def batch_sweep(
    model,
    model_name,
    dataset,
    dataloader,
    perturbations: dict,
    sweep_params: dict,
    sweep_ranges: dict,
    step_sizes: dict,
    metric,
    runs: int = 3,
    device: str = "cuda:0",
    output_dir: str = "sweep_results"
):
    evaluators = {}
    # Get model-specific transform
    post_transform = get_transform(model_name, model=model)  # Uses your model_transforms dict

    for i, (name, cfg) in enumerate(perturbations.items()):
        print(f"\n=== Sweeping {i+1} / {len(perturbations.keys())} {name} on model {model_name} ===")
        evaluator = Evaluator(
            model,
            model_name,
            dataset,
            dataloader,
            cfg["TransformOptimiser"],
            cfg["Transform"],
            cfg["optimiser_params"],
            cfg["transform_params"],
            device=device,
            post_transform=post_transform   # <-- pass here!
        )
        evaluators[name] = evaluator

        param       = sweep_params[name]
        param_range = sweep_ranges[name]
        step_size   = step_sizes[name]

        df = sweep_constraint(
            evaluator     = evaluator,
            param         = param,
            param_range   = param_range,
            step_size     = step_size,
            metric        = metric,
            adversarial   = True,
            runs          = runs
        )

        if name == "brightness":
            xls_path = os.path.join(output_dir, f"{model_name}_brightness_sweep.xlsx")
        else:
            xls_path = os.path.join(output_dir, f"{model_name}_{name}_sweep.xlsx")
        os.makedirs(os.path.dirname(xls_path), exist_ok=True)
        df.to_excel(xls_path, index=False)

        fig = plot_with_error_band(
            df,
            model_name=model_name,
            perturbation=name,
            param_name=param,
            y_col=metric.__name__,
            std_col="std"
        )
        if name == "brightness":
            out_path = os.path.join(output_dir, f"{model_name}_brightness.png")
        else:
            out_path = os.path.join(output_dir, f"{model_name}_{name}.png")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot -> {out_path}")

# ------------ Evaluation Functions ------------
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

def calculate_ppi_df(model_list, perturb_list, dataset_name, PATH):
    ppi_mean_df = pd.DataFrame(index=perturb_list, columns=model_list, dtype=float)
    ppi_std_df = pd.DataFrame(index=perturb_list, columns=model_list, dtype=float)

    for model in model_list:
        for perturb in perturb_list:
            stats_path = os.path.join(
                PATH, "Stats", dataset_name, model, "constraint_sweep", f"{model}_{perturb}_sweep.xlsx"
            )
            if not os.path.isfile(stats_path):
                print(f"Missing: {stats_path}")
                continue
            df = pd.read_excel(stats_path)
            x = df.iloc[:, 0].values  # param
            run_ppis = []
            run_cols = [c for c in df.columns if c.startswith("run_")]
            for run_col in run_cols:
                y = df[run_col].values
                if len(np.unique(x)) < 2:
                    print(f"Not enough points for {model}/{perturb}/{run_col}")
                    continue
                x_norm = (x - x.min()) / (x.max() - x.min())
                sort_idx = np.argsort(x_norm)
                x_norm_sorted = x_norm[sort_idx]
                y_sorted = y[sort_idx]
                ppi = np.trapezoid(y_sorted, x_norm_sorted)
                run_ppis.append(ppi)
            if len(run_ppis) == 0:
                continue
            ppi_mean_df.loc[perturb, model] = np.mean(run_ppis)
            ppi_std_df.loc[perturb, model] = np.std(run_ppis, ddof=1)

    if "mean" in ppi_mean_df.index:
        ppi_mean_df.rename(index={"mean": "brightness"}, inplace=True)
        ppi_std_df.rename(index={"mean": "brightness"}, inplace=True)

    # Add real mean row
    ppi_mean_df.loc["Mean"] = ppi_mean_df.mean(axis=0)
    ppi_std_df.loc["Mean"] = ppi_std_df.mean(axis=0)

    # Sort columns by mean
    mean_ppi = ppi_mean_df.loc["Mean"].sort_values(ascending=True)
    ppi_mean_df = ppi_mean_df[mean_ppi.index]
    ppi_std_df = ppi_std_df[mean_ppi.index]

    out_path = os.path.join(PATH, "Stats", dataset_name, "constraint_sweep", "ppi_aggregate.xlsx")
    with pd.ExcelWriter(out_path) as writer:
        ppi_mean_df.to_excel(writer, sheet_name="ppi_mean")
        ppi_std_df.to_excel(writer, sheet_name="ppi_std")
    print(f"Saved PPI mean/std tables to {out_path}")
    return ppi_mean_df, ppi_std_df

def plot_mean_bar_chart(ppi_mean_df, ppi_std_df, dataset_name, PATH):
    mean_ppi = ppi_mean_df.loc["Mean"]
    std_ppi = ppi_std_df.loc["Mean"]
    plt.figure(figsize=(10, 6))
    ax = mean_ppi.plot(kind="bar", yerr=std_ppi, color="darkorange", capsize=7, error_kw=dict(lw=2,elinewidth=2))
    plt.ylabel("Mean PPI")
    plt.xlabel("Model")
    plt.title(f"{dataset_name} Mean PPI per Model")
    # plt.ylim(min(mean_ppi - std_ppi) - 0.1, 1)
    plt.ylim(0, 1)
    plt.tight_layout()

    # Annotate values
    for i, (v, std) in enumerate(zip(mean_ppi, std_ppi)):
        ax.text(i, v + std + 0.01, f"{v:.3f}", ha='center', va='bottom', fontsize=11, fontweight='medium')

    plot_path = os.path.join(PATH, "Stats", dataset_name, "constraint_sweep", "ppi_mean_bar.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved bar chart to {plot_path}")

def get_spread_x_y(x, y, min_x_dist=15, min_y_dist=0.04):
    x = np.array(x)
    y = np.array(y)
    unique_x, counts_x = np.unique(x, return_counts=True)
    x_new = x.copy().astype(float)
    y_new = y.copy().astype(float)
    for u in unique_x[counts_x > 1]:
        idx = np.where(x == u)[0]
        n = len(idx)
        if n > 1:
            x_offsets = np.linspace(-min_x_dist, min_x_dist, n)
            y_offsets = np.linspace(-min_y_dist, min_y_dist, n)
            x_new[idx] = x[idx] + x_offsets
            y_new[idx] = y[idx] + y_offsets
    return x_new, y_new

def plot_ppi_vs_param_scatter(ppi_mean_df, dataset_name, PATH, model_param_counts, log_x=True):
    outdir = os.path.join(PATH, "Stats", dataset_name, "constraint_sweep", "ppi")
    os.makedirs(outdir, exist_ok=True)
    available_models = [m for m in ppi_mean_df.columns if m in model_param_counts]
    param_counts = np.array([model_param_counts[m] for m in available_models])
    # available_models = model_param_counts.keys()
    # param_counts = model_param_counts.values()

    def plot_scatter(x, y, models, perturb_name, color, fit_color, fname, log_x):
        x_spread, y_spread = get_spread_x_y(x, y)
        plt.figure(figsize=(8, 5))
        plt.scatter(x_spread, y_spread, c=color, s=120, zorder=2)
        for xi, yi, m in zip(x_spread, y_spread, models):
            plt.annotate(m, (xi, yi+0.01), fontsize=10, ha="center", va="bottom", zorder=3)
        if len(x) >= 2 and not np.all(x == x[0]):
            coeffs = np.polyfit(x, y, 1)
            fit_fn = np.poly1d(coeffs)
            x_line = np.linspace(min(x), max(x), 100)
            plt.plot(x_line, fit_fn(x_line), '--', color=fit_color, label="Best Fit", zorder=1)
            r, p = pearsonr(x, y)
            plt.text(0.99, 0.02, f"Pearson r = {r:.2f}", fontsize=12, ha='right', va='bottom',
                     transform=plt.gca().transAxes)

        if log_x:
            plt.xscale("log")
            plt.xlabel("Parameters (millions, log scale)")
        else:
            plt.xlabel("Model Parameter Count (Millions)")

        plt.ylabel(f"PPI ({perturb_name})")
        plt.title(f"{dataset_name} PPI vs. Parameter Count: {perturb_name}")
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")

    for perturb in ppi_mean_df.index:
        if perturb == "Mean":
            continue
        y_spread = ppi_mean_df.loc[perturb, available_models].values.astype(float)
        plot_path = os.path.join(outdir, f"PPI_vs_param_{perturb}.png")
        plot_scatter(param_counts, y_spread, available_models, perturb, "dodgerblue", "gray", plot_path, log_x)

    y_spread = ppi_mean_df.loc["Mean", available_models].values.astype(float)
    plot_path = os.path.join(outdir, "PPI_vs_param_Mean.png")
    plot_scatter(param_counts, y_spread, available_models, "Mean", "darkorange", "gray", plot_path, log_x)


def plot_ppi_heatmap(ppi_mean_df, dataset_name, PATH):
    outdir = os.path.join(PATH, "Stats", dataset_name, "constraint_sweep", "ppi")
    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, "ppi_heatmap.png")
    plt.figure(figsize=(12, 5))
    ppi_mean_df = ppi_mean_df.reindex(["blur", "crop", "jpeg", "brightness", "pixel", "random_stain", "rotate", "zoom_in", "zoom_out","Mean"])
    sns.heatmap(ppi_mean_df, annot=True, fmt=".3f", cmap="viridis", cbar_kws={'label': 'PPI'})
    plt.title(f"{dataset_name} PPI Heatmap")
    plt.ylabel("Perturbation")
    plt.xlabel("Model")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved heatmap to {out_path}")

def calculate_PPI_aggregate(model_list, perturb_list, dataset_name, PATH, model_param_counts):
    ppi_mean_df, ppi_std_df = calculate_ppi_df(model_list, perturb_list, dataset_name, PATH)
    plot_mean_bar_chart(ppi_mean_df, ppi_std_df, dataset_name, PATH)
    plot_ppi_vs_param_scatter(ppi_mean_df, dataset_name, PATH, model_param_counts)
    plot_ppi_heatmap(ppi_mean_df, dataset_name, PATH)


# ─── collect_model_stats with save_dir_base argument ─────────────────────────
def collect_model_stats(
    model,
    model_name: str,
    test_data,
    test_loader,
    save_dir_base: str,
    runs: int = 5,
    perturbations: list = None,
    metric_names: list = None
) -> pd.DataFrame:
    import numpy as np
    import pandas as pd
    import copy

    # Define metric renaming
    metric_rename = {
        "Accuracy": "Baseline Accuracy",
        "ROC AUC": "Baseline ROC",
        "PR AUC": "Baseline PR",
        "Robustness Accuracy": "Accuracy After Perturbation"
    }

    metric_names_full = [
        "Accuracy", "ROC AUC", "PR AUC",
        "Robustness Accuracy", "Fooling Ratio", "Input Sensitivity",
        "Adversarial ROC AUC", "Adversarial PR AUC"
    ]
    mean_only_metrics = ["Accuracy", "ROC AUC", "PR AUC"]
    std_metrics = [m for m in metric_names_full if m not in mean_only_metrics]

    if perturbations is None:
        perturbations = [
            "pixel", "stain", "brightness", "rotate", "crop",
            "blur", "zoomin", "zoomout", "hed", "random_stain", "jpeg"
        ]
    if metric_names is None:
        metric_names = metric_names_full

    # Build MultiIndex columns with renamed metrics
    col_tuples = []
    for m in metric_names:
        m_renamed = metric_rename.get(m, m)
        col_tuples.append((model_name, m_renamed, "Mean"))
        if m in std_metrics:
            col_tuples.append((model_name, m_renamed, "STD"))
    columns = pd.MultiIndex.from_tuples(
        col_tuples, names=["Model", "Metric", "Statistic"]
    )

    # DataFrame indexed by perturbation name
    stats_df = pd.DataFrame(
        index=perturbations,
        columns=columns,
        dtype=float
    )

    for i, p in enumerate(perturbations):
        print(f"-------- {i+1}/{len(perturbations)}: Evaluating '{p}' --------")
        # {metric: [run_1, run_2, ...]}
        raw = {m: [] for m in metric_names_full}

        eval_fn = globals()[f"evaluate_{p}"]
        for run_idx in range(runs):
            print(f"  Run {run_idx+1}/{runs}...", end="")
            vals = eval_fn(
                model,
                model_name,
                test_data,
                test_loader,
                save_dir_base
            )
            print(" done")
            # Now vals = (acc, roc, pr, robust_acc, fool_ratio, input_sens, ad_roc, ad_pr)
            for m, v in zip(metric_names_full, vals):
                raw[m].append(v)

        # fill means & STDs
        for m in metric_names:
            m_renamed = metric_rename.get(m, m)
            stats_df.loc[p, (model_name, m_renamed, "Mean")] = np.mean(raw[m])
            if m in std_metrics:
                stats_df.loc[p, (model_name, m_renamed, "STD")] = np.std(raw[m])

    # Compute and append the "Mean" row (averaged across perturbations)
    mean_row = {}
    for m in metric_names:
        m_renamed = metric_rename.get(m, m)
        mean_row[(model_name, m_renamed, "Mean")] = stats_df.loc[:, (model_name, m_renamed, "Mean")].mean()
        if m in std_metrics:
            mean_row[(model_name, m_renamed, "STD")] = stats_df.loc[:, (model_name, m_renamed, "STD")].mean()

    stats_df.loc["Mean"] = pd.Series(mean_row)
    stats_df = stats_df.reindex(perturbations + ["Mean"])

    return stats_df

from reetoolbox.optimisers import PGD, StochasticSearch
from reetoolbox.image_evaluator import Evaluator
from reetoolbox.metrics import get_metrics

from reetoolbox.transforms import (
    PixelTransform,          # → eval_pixel_optimiser_params, eval_pixel_transform_params
    StainTransform,          # → eval_stain_optimiser_params, eval_stain_transform_params
    MeanTransform,           # → eval_mean_optimiser_params, eval_mean_transform_params
    RotateTransform,         # → eval_rotate_optimiser_params, eval_rotate_transform_params
    CropTransform,           # → eval_crop_optimiser_params, eval_crop_transform_params
    BlurTransform,           # → eval_blur_optimiser_params, eval_blur_transform_params
    ZoomInTransform,         # → eval_zoom_in_optimiser_params, eval_zoom_in_transform_params
    ZoomOutTransform,        # → eval_zoom_out_optimiser_params, eval_zoom_out_transform_params
    HEDTransform,            # → eval_hed_optimiser_params, eval_hed_transform_params
    RandomStainTransform,    # → eval_random_stain_optimiser_params, eval_random_stain_transform_params
    JPEGTransform,           # → eval_jpeg_optimiser_params, eval_jpeg_transform_params
)

from reetoolbox.constants import (
    eval_pixel_optimiser_params,        eval_pixel_transform_params,
    eval_stain_optimiser_params,        eval_stain_transform_params,
    eval_mean_optimiser_params,         eval_mean_transform_params,
    eval_crop_optimiser_params,         eval_crop_transform_params,
    eval_rotate_optimiser_params,       eval_rotate_transform_params,
    eval_crop_optimiser_params,         eval_crop_transform_params,
    eval_blur_optimiser_params,         eval_blur_transform_params,
    eval_zoom_in_optimiser_params,      eval_zoom_in_transform_params,
    eval_zoom_out_optimiser_params,     eval_zoom_out_transform_params,
    eval_hed_optimiser_params,          eval_hed_transform_params,
    eval_random_stain_optimiser_params, eval_random_stain_transform_params,
    eval_jpeg_optimiser_params,         eval_jpeg_transform_params,
)

def evaluate_pixel(model, model_name, test_dataset, test_dataloader, save_dir_base):
    perturb = "pixel"
    save_dir = os.path.join(save_dir_base, model_name, perturb)
    evaluator = Evaluator(
        model, model_name, 
        test_dataset, test_dataloader,
        PGD, PixelTransform,
        eval_pixel_optimiser_params, eval_pixel_transform_params,
        device=device, save_dir=save_dir
    )
    return get_metrics(evaluator.predict(adversarial=True))

def evaluate_stain(model, model_name, test_dataset, test_dataloader, save_dir_base):
    perturb = "stain"
    save_dir = os.path.join(save_dir_base, model_name, perturb)
    evaluator = Evaluator(
        model, model_name, 
        test_dataset, test_dataloader,
        PGD, StainTransform,
        eval_stain_optimiser_params, eval_stain_transform_params,
        device=device, save_dir=save_dir
    )
    return get_metrics(evaluator.predict(adversarial=True))

def evaluate_brightness(model, model_name, test_dataset, test_dataloader, save_dir_base):
    perturb = "brightness"
    save_dir = os.path.join(save_dir_base, model_name, perturb)
    evaluator = Evaluator(
        model, model_name, 
        test_dataset, test_dataloader,
        PGD, MeanTransform,
        eval_mean_optimiser_params, eval_mean_transform_params,
        device=device, save_dir=save_dir
    )
    return get_metrics(evaluator.predict(adversarial=True))

def evaluate_rotate(model, model_name, test_dataset, test_dataloader, save_dir_base):
    perturb = "rotate"
    save_dir = os.path.join(save_dir_base, model_name, perturb)
    evaluator = Evaluator(
        model, model_name, 
        test_dataset, test_dataloader,
        StochasticSearch, RotateTransform,
        eval_rotate_optimiser_params, eval_rotate_transform_params,
        device=device, save_dir=save_dir
    )
    return get_metrics(evaluator.predict(adversarial=True))

def evaluate_crop(model, model_name, test_dataset, test_dataloader, save_dir_base):
    perturb = "crop"
    save_dir = os.path.join(save_dir_base, model_name, perturb)
    evaluator = Evaluator(
        model, model_name, 
        test_dataset, test_dataloader,
        StochasticSearch, CropTransform,
        eval_crop_optimiser_params, eval_crop_transform_params,
        device=device, save_dir=save_dir
    )
    return get_metrics(evaluator.predict(adversarial=True))

def evaluate_blur(model, model_name, test_dataset, test_dataloader, save_dir_base):
    perturb = "blur"
    save_dir = os.path.join(save_dir_base, model_name, perturb)
    evaluator = Evaluator(
        model, model_name, 
        test_dataset, test_dataloader,
        StochasticSearch, BlurTransform,
        eval_blur_optimiser_params, eval_blur_transform_params,
        device=device, save_dir=save_dir
    )
    return get_metrics(evaluator.predict(adversarial=True))

def evaluate_zoomin(model, model_name, test_dataset, test_dataloader, save_dir_base):
    perturb = "zoomin"
    save_dir = os.path.join(save_dir_base, model_name, perturb)
    evaluator = Evaluator(
        model, model_name, 
        test_dataset, test_dataloader,
        StochasticSearch, ZoomInTransform,
        eval_zoom_in_optimiser_params, eval_zoom_in_transform_params,
        device=device, save_dir=save_dir
    )
    return get_metrics(evaluator.predict(adversarial=True))

def evaluate_zoomout(model, model_name, test_dataset, test_dataloader, save_dir_base):
    perturb = "zoomout"
    save_dir = os.path.join(save_dir_base, model_name, perturb)
    evaluator = Evaluator(
        model, model_name, 
        test_dataset, test_dataloader,
        StochasticSearch, ZoomOutTransform,
        eval_zoom_out_optimiser_params, eval_zoom_out_transform_params,
        device=device, save_dir=save_dir
    )
    return get_metrics(evaluator.predict(adversarial=True))

def evaluate_hed(model, model_name, test_dataset, test_dataloader, save_dir_base):
    perturb = "hed"
    save_dir = os.path.join(save_dir_base, model_name, perturb)
    evaluator = Evaluator(
        model, model_name, 
        test_dataset, test_dataloader,
        StochasticSearch, HEDTransform,
        eval_hed_optimiser_params, eval_hed_transform_params,
        device=device, save_dir=save_dir
    )
    return get_metrics(evaluator.predict(adversarial=True))

def evaluate_random_stain(model, model_name, test_dataset, test_dataloader, save_dir_base):
    perturb = "random_stain"
    save_dir = os.path.join(save_dir_base, model_name, perturb)
    evaluator = Evaluator(
        model, model_name, 
        test_dataset, test_dataloader,
        StochasticSearch, RandomStainTransform,
        eval_random_stain_optimiser_params, eval_random_stain_transform_params,
        device=device, save_dir=save_dir
    )
    return get_metrics(evaluator.predict(adversarial=True))

def evaluate_jpeg(model, model_name, test_dataset, test_dataloader, save_dir_base):
    perturb = "jpeg"
    save_dir = os.path.join(save_dir_base, model_name, perturb)
    evaluator = Evaluator(
        model, model_name, 
        test_dataset, test_dataloader,
        StochasticSearch, JPEGTransform,
        eval_jpeg_optimiser_params, eval_jpeg_transform_params,
        device=device, save_dir=save_dir
    )
    return get_metrics(evaluator.predict(adversarial=True))
