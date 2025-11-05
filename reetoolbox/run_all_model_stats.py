# ──────────────────────────────────────────────────────────────────────────────
#  run_all_model_stats.py
#      • Drives evaluation of every model × perturbation combination
#      • Applies the model-specific transform *after* the adversarial attack
#      • Collects metrics and writes an Excel file per model
# ──────────────────────────────────────────────────────────────────────────────
import os
import torch
import pandas as pd
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from torchvision import transforms

# ─── your own modules ────────────────────────────────────────────────────────
from reetoolbox.eval_funcs import get_transform
from reetoolbox.image_evaluator import Evaluator
from reetoolbox.metrics import get_metrics
from reetoolbox.optimisers import PGD, StochasticSearch
from reetoolbox.transforms import (
    PixelTransform, StainTransform, MeanTransform, RotateTransform,
    CropTransform, BlurTransform, ZoomInTransform, ZoomOutTransform,
    HEDTransform, RandomStainTransform, JPEGTransform,
)
from reetoolbox.constants import (
    eval_pixel_optimiser_params,        eval_pixel_transform_params,
    eval_stain_optimiser_params,        eval_stain_transform_params,
    eval_mean_optimiser_params,         eval_mean_transform_params,
    eval_crop_optimiser_params,         eval_crop_transform_params,
    eval_rotate_optimiser_params,       eval_rotate_transform_params,
    eval_blur_optimiser_params,         eval_blur_transform_params,
    eval_zoom_in_optimiser_params,      eval_zoom_in_transform_params,
    eval_zoom_out_optimiser_params,     eval_zoom_out_transform_params,
    eval_hed_optimiser_params,          eval_hed_transform_params,
    eval_random_stain_optimiser_params, eval_random_stain_transform_params,
    eval_jpeg_optimiser_params,         eval_jpeg_transform_params,
)

# If you prefer to choose a device outside this file, just comment this line
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ──────────────────────────────────────────────────────────────────────────────
#  TOP-LEVEL DRIVER
# ──────────────────────────────────────────────────────────────────────────────
def run_all_model_stats(
    model_evals,
    root_dir,
    batch_size,
    test_multiplier,
    runs,
    device,
    PATH,
    save_dir_base,
    dataset_class,
    dataset_kwargs=None,
):
    """
    Evaluate every entry in `model_evals` on every perturbation listed in
    `collect_model_stats()`.  Results are stored as Excel files under
    `save_dir_base/<model_eval['output_subdir']>/…`.
    """
    if dataset_kwargs is None:
        dataset_kwargs = {"folds": (3,), "min_positive": 5}

    # 1. ── Build a *raw* dataset for stratified sampling ──────────────────
    dummy_ds = dataset_class(
        root_dir,
        transform=transforms.ToTensor(),   # No model-specific preprocessing
        **dataset_kwargs,
    )
    ix_test  = list(range(len(dummy_ds)))
    if hasattr(dummy_ds, "labels"):
        labels_test = [dummy_ds.labels[i] for i in ix_test]
    elif hasattr(dummy_ds, "samples"):
        labels_test = [dummy_ds.samples[i]["label"] for i in ix_test]
    else:
        raise AttributeError(
            "Dataset must expose .labels or .samples for stratification."
        )

    # 2. ── Train/test split with identical class distribution ─────────────
    test_ix, _ = train_test_split(
        ix_test,
        train_size=test_multiplier,
        stratify=labels_test,
        random_state=42,
    )

    # 3. ── Loop over every model definition ───────────────────────────────
    for model_eval in model_evals:
        name       = model_eval["model_name"]
        weight_pth = model_eval["weight_path"]
        print(f"\n────────  {name}  ─ collecting statistics ────────")

        # a. ── Load weights ───────────────────────────────────────────────
        model = model_eval["load_func"]()
        if weight_pth is not None:
            model.head.load_state_dict(torch.load(weight_pth))
        model = model.to(device).eval()

        # b. ── Split preprocessing:   raw→tensor (before attack)  vs
        #                               post_transform (after attack) ─────
        base_transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
        post_transform = get_transform(name, model=model)

        # c. ── Build dataset/loader restricted to `test_ix`  ──────────────
        test_full   = dataset_class(root_dir, transform=base_transform,
                                    **dataset_kwargs)
        test_subset = Subset(test_full, test_ix)
        test_loader = DataLoader(test_subset,
                                 batch_size=batch_size,
                                 shuffle=False)

        # d. ── Collect statistics ────────────────────────────────────────
        stats_df = collect_model_stats(
            model=model,
            model_name=name,
            test_data=test_subset,
            test_loader=test_loader,
            save_dir_base=save_dir_base,
            runs=runs,
            post_transform=post_transform,      # ← important
        )

        # e. ── Write Excel file ───────────────────────────────────────────
        xlsx = os.path.join(
            save_dir_base,
            model_eval["output_subdir"],
            f"{name}_stats.xlsx",
        )
        os.makedirs(os.path.dirname(xlsx), exist_ok=True)
        stats_df.to_excel(xlsx)
        print(f"✔ Metrics saved →  {xlsx}")


# ──────────────────────────────────────────────────────────────────────────────
#  STATS HELPER (unchanged logic, now accepts `post_transform`)
# ──────────────────────────────────────────────────────────────────────────────
def collect_model_stats(
    model,
    model_name: str,
    test_data,
    test_loader,
    save_dir_base: str,
    runs: int = 5,
    perturbations: list | None = None,
    metric_names: list | None = None,
    post_transform=None,            # ← NEW
) -> pd.DataFrame:
    """
    Execute every perturbation `runs` times and aggregate metrics.
    """
    import numpy as np

    # ── Metric bookkeeping ───────────────────────────────────────────────
    metric_rename = {
        "Accuracy":              "Baseline Accuracy",
        "ROC AUC":               "Baseline ROC",
        "PR AUC":                "Baseline PR",
        "Robustness Accuracy":   "Accuracy After Perturbation",
    }

    metric_names_full = [
        "Accuracy", "ROC AUC", "PR AUC",
        "Robustness Accuracy", "Fooling Ratio", "Input Sensitivity",
        "Adversarial ROC AUC", "Adversarial PR AUC",
    ]
    mean_only_metrics = ["Accuracy", "ROC AUC", "PR AUC"]
    std_metrics       = [m for m in metric_names_full if m not in mean_only_metrics]

    if perturbations is None:
        perturbations = [
            "pixel", "stain", "brightness", "rotate", "crop",
            "blur", "zoomin", "zoomout", "hed",
            "random_stain", "jpeg",
        ]
    if metric_names is None:
        metric_names = metric_names_full

    # ── Prepare Multi-index DataFrame for results ────────────────────────
    col_tuples = []
    for metric in metric_names:
        renamed = metric_rename.get(metric, metric)
        col_tuples.append((model_name, renamed, "Mean"))
        if metric in std_metrics:
            col_tuples.append((model_name, renamed, "STD"))
    columns = pd.MultiIndex.from_tuples(
        col_tuples, names=["Model", "Metric", "Statistic"]
    )
    stats_df = pd.DataFrame(index=perturbations, columns=columns, dtype=float)

    # ── For each perturbation, call the dedicated wrapper ────────────────
    for i, pert in enumerate(perturbations, start=1):
        print(f"  {i}/{len(perturbations)}  •  '{pert}'")
        raw = {m: [] for m in metric_names_full}

        eval_fn = globals()[f"evaluate_{pert}"]
        for run_idx in range(runs):
            vals = eval_fn(
                model,
                model_name,
                test_data,
                test_loader,
                save_dir_base,
                post_transform=post_transform,   # ← NEW
            )
            for m, v in zip(metric_names_full, vals):
                raw[m].append(v)

        # ─ aggregate to dataframe ────────────────────────────────────────
        for metric in metric_names:
            renamed = metric_rename.get(metric, metric)
            stats_df.loc[pert, (model_name, renamed, "Mean")] = np.mean(raw[metric])
            if metric in std_metrics:
                stats_df.loc[pert, (model_name, renamed, "STD")] = np.std(raw[metric])

    # ── Append row of averages across perturbations ───────────────────────
    mean_row = {}
    for metric in metric_names:
        renamed = metric_rename.get(metric, metric)
        mean_row[(model_name, renamed, "Mean")] = \
            stats_df.loc[:, (model_name, renamed, "Mean")].mean()
        if metric in std_metrics:
            mean_row[(model_name, renamed, "STD")] = \
                stats_df.loc[:, (model_name, renamed, "STD")].mean()
    stats_df.loc["Mean"] = pd.Series(mean_row)

    # Maintain ordering
    stats_df = stats_df.reindex(perturbations + ["Mean"])
    return stats_df


# ──────────────────────────────────────────────────────────────────────────────
#  evaluate_* FACTORY  (eliminates boiler-plate)
# ──────────────────────────────────────────────────────────────────────────────
def _mk_eval(
    perturb: str,
    TransformOptimiser,
    Transform,
    optim_params: dict,
    trans_params: dict,
):
    """
    Produces evaluate_<perturb>() functions that match your original API but
    now forward `post_transform` to Evaluator.
    """
    def _fn(model, model_name, dataset, dataloader, save_root, *,
            post_transform=None, device=device):
        save_dir = os.path.join(save_root, model_name, perturb)
        evaluator = Evaluator(
            model, model_name,
            dataset, dataloader,
            TransformOptimiser, Transform,
            optim_params, trans_params,
            device=device,
            save_dir=save_dir,
            post_transform=post_transform,
        )
        return get_metrics(evaluator.predict(adversarial=True))
    return _fn


# ──────────────────────────────────────────────────────────────────────────────
#  Concrete evaluate_* wrappers (exactly one line each)
# ──────────────────────────────────────────────────────────────────────────────
evaluate_pixel        = _mk_eval("pixel",         PGD,             PixelTransform,
                                 eval_pixel_optimiser_params,        eval_pixel_transform_params)
evaluate_stain        = _mk_eval("stain",         PGD,             StainTransform,
                                 eval_stain_optimiser_params,        eval_stain_transform_params)
evaluate_brightness   = _mk_eval("brightness",    PGD,             MeanTransform,
                                 eval_mean_optimiser_params,         eval_mean_transform_params)
evaluate_rotate       = _mk_eval("rotate",        StochasticSearch, RotateTransform,
                                 eval_rotate_optimiser_params,       eval_rotate_transform_params)
evaluate_crop         = _mk_eval("crop",          StochasticSearch, CropTransform,
                                 eval_crop_optimiser_params,         eval_crop_transform_params)
evaluate_blur         = _mk_eval("blur",          StochasticSearch, BlurTransform,
                                 eval_blur_optimiser_params,         eval_blur_transform_params)
evaluate_zoomin       = _mk_eval("zoomin",        StochasticSearch, ZoomInTransform,
                                 eval_zoom_in_optimiser_params,      eval_zoom_in_transform_params)
evaluate_zoomout      = _mk_eval("zoomout",       StochasticSearch, ZoomOutTransform,
                                 eval_zoom_out_optimiser_params,     eval_zoom_out_transform_params)
evaluate_hed          = _mk_eval("hed",           StochasticSearch, HEDTransform,
                                 eval_hed_optimiser_params,          eval_hed_transform_params)
evaluate_random_stain = _mk_eval("random_stain",  StochasticSearch, RandomStainTransform,
                                 eval_random_stain_optimiser_params, eval_random_stain_transform_params)
evaluate_jpeg         = _mk_eval("jpeg",          StochasticSearch, JPEGTransform,
                                 eval_jpeg_optimiser_params,         eval_jpeg_transform_params)
