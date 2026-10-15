"""Perturbation Performance Index (dissertation section 3.3).

PPI = area under the ROC-AUC vs normalised-constraint curve of a sweep, computed per run.
Reads the constraint-sweep workbooks and writes, per dataset,
``Stats/<dataset>/constraint_sweep/ppi_aggregate.xlsx`` (sheets ppi_mean / ppi_std) plus a bar
chart, a heatmap and PPI-vs-parameter-count scatter plots.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr

from .models import MODELS
from .paths import stats_dir
from .perturbations import SWEEPS

# Heatmap row order (sweep "mean" is reported as "brightness")
PERTURB_ORDER = ["blur", "crop", "jpeg", "brightness", "pixel", "random_stain", "rotate", "zoom_in", "zoom_out"]


def ppi_from_sweep(df: pd.DataFrame) -> list:
    """One PPI per `run_*` column of a sweep table (x = first column, min-max normalised)."""
    x = df.iloc[:, 0].values
    if len(np.unique(x)) < 2:
        return []
    run_ppis = []
    for run_col in [c for c in df.columns if c.startswith("run_")]:
        y = df[run_col].values
        x_norm = (x - x.min()) / (x.max() - x.min())
        sort_idx = np.argsort(x_norm)
        run_ppis.append(np.trapezoid(y[sort_idx], x_norm[sort_idx]))
    return run_ppis


def calculate_ppi_df(dataset: str, model_names=None, perturbations=None):
    """(ppi_mean_df, ppi_std_df): rows = perturbations + "Mean", columns = models sorted by mean PPI."""
    model_names = list(model_names or MODELS)
    perturbations = list(perturbations or SWEEPS)
    ppi_mean_df = pd.DataFrame(index=perturbations, columns=model_names, dtype=float)
    ppi_std_df = pd.DataFrame(index=perturbations, columns=model_names, dtype=float)

    for model in model_names:
        for perturb in perturbations:
            path = stats_dir(dataset, model, "constraint_sweep", f"{model}_{perturb}_sweep.xlsx")
            if not path.is_file():
                print(f"Missing: {path}")
                continue
            run_ppis = ppi_from_sweep(pd.read_excel(path))
            if not run_ppis:
                print(f"Not enough points for {model}/{perturb}")
                continue
            ppi_mean_df.loc[perturb, model] = np.mean(run_ppis)
            ppi_std_df.loc[perturb, model] = np.std(run_ppis, ddof=1)

    ppi_mean_df.rename(index={"mean": "brightness"}, inplace=True)
    ppi_std_df.rename(index={"mean": "brightness"}, inplace=True)

    ppi_mean_df.loc["Mean"] = ppi_mean_df.mean(axis=0)
    ppi_std_df.loc["Mean"] = ppi_std_df.mean(axis=0)

    order = ppi_mean_df.loc["Mean"].sort_values(ascending=True).index
    ppi_mean_df = ppi_mean_df[order]
    ppi_std_df = ppi_std_df[order]

    out_dir = stats_dir(dataset, "constraint_sweep")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ppi_aggregate.xlsx"
    with pd.ExcelWriter(out_path) as writer:
        ppi_mean_df.to_excel(writer, sheet_name="ppi_mean")
        ppi_std_df.to_excel(writer, sheet_name="ppi_std")
    print(f"Saved PPI mean/std tables to {out_path}")
    return ppi_mean_df, ppi_std_df


def plot_mean_bar_chart(ppi_mean_df, ppi_std_df, dataset):
    mean_ppi = ppi_mean_df.loc["Mean"]
    std_ppi = ppi_std_df.loc["Mean"]
    plt.figure(figsize=(10, 6))
    ax = mean_ppi.plot(kind="bar", yerr=std_ppi, color="darkorange", capsize=7,
                       error_kw=dict(lw=2, elinewidth=2))
    plt.ylabel("Mean PPI")
    plt.xlabel("Model")
    plt.title(f"{dataset} Mean PPI per Model")
    plt.ylim(0, 1)
    plt.tight_layout()
    for i, (v, std) in enumerate(zip(mean_ppi, std_ppi)):
        ax.text(i, v + std + 0.01, f"{v:.3f}", ha='center', va='bottom', fontsize=11, fontweight='medium')

    plot_path = stats_dir(dataset, "constraint_sweep", "ppi_mean_bar.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved bar chart to {plot_path}")


def _spread(x, y, min_x_dist=15, min_y_dist=0.04):
    """Jitter points that share an x value so labels do not overlap."""
    x = np.array(x)
    y = np.array(y)
    unique_x, counts_x = np.unique(x, return_counts=True)
    x_new = x.copy().astype(float)
    y_new = y.copy().astype(float)
    for u in unique_x[counts_x > 1]:
        idx = np.where(x == u)[0]
        n = len(idx)
        x_new[idx] = x[idx] + np.linspace(-min_x_dist, min_x_dist, n)
        y_new[idx] = y[idx] + np.linspace(-min_y_dist, min_y_dist, n)
    return x_new, y_new


def plot_ppi_vs_param_scatter(ppi_mean_df, dataset, log_x=True):
    """One scatter per perturbation (and Mean): PPI vs parameter count, with fit and Pearson r."""
    outdir = stats_dir(dataset, "constraint_sweep", "ppi")
    outdir.mkdir(parents=True, exist_ok=True)
    available_models = [m for m in ppi_mean_df.columns if m in MODELS]
    param_counts = np.array([MODELS[m].params_m for m in available_models])

    def plot_scatter(x, y, models, perturb_name, color, fname):
        x_spread, y_spread = _spread(x, y)
        plt.figure(figsize=(8, 5))
        plt.scatter(x_spread, y_spread, c=color, s=120, zorder=2)
        for xi, yi, m in zip(x_spread, y_spread, models):
            plt.annotate(m, (xi, yi + 0.01), fontsize=10, ha="center", va="bottom", zorder=3)
        if len(x) >= 2 and not np.all(x == x[0]):
            fit_fn = np.poly1d(np.polyfit(x, y, 1))
            x_line = np.linspace(min(x), max(x), 100)
            plt.plot(x_line, fit_fn(x_line), '--', color="gray", label="Best Fit", zorder=1)
            r, _ = pearsonr(x, y)
            plt.text(0.99, 0.02, f"Pearson r = {r:.2f}", fontsize=12, ha='right', va='bottom',
                     transform=plt.gca().transAxes)
        if log_x:
            plt.xscale("log")
            plt.xlabel("Parameters (millions, log scale)")
        else:
            plt.xlabel("Model Parameter Count (Millions)")
        plt.ylabel(f"PPI ({perturb_name})")
        plt.title(f"{dataset} PPI vs. Parameter Count: {perturb_name}")
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")

    for perturb in ppi_mean_df.index:
        if perturb == "Mean":
            continue
        y = ppi_mean_df.loc[perturb, available_models].values.astype(float)
        plot_scatter(param_counts, y, available_models, perturb, "dodgerblue", outdir / f"PPI_vs_param_{perturb}.png")

    y = ppi_mean_df.loc["Mean", available_models].values.astype(float)
    plot_scatter(param_counts, y, available_models, "Mean", "darkorange", outdir / "PPI_vs_param_Mean.png")


def plot_ppi_heatmap(ppi_mean_df, dataset):
    out_path = stats_dir(dataset, "constraint_sweep", "ppi", "ppi_heatmap.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [p for p in PERTURB_ORDER if p in ppi_mean_df.index]
    rows += [p for p in ppi_mean_df.index if p not in rows and p != "Mean"] + ["Mean"]
    plt.figure(figsize=(12, 5))
    sns.heatmap(ppi_mean_df.reindex(rows), annot=True, fmt=".3f", cmap="viridis", cbar_kws={'label': 'PPI'})
    plt.title(f"{dataset} PPI Heatmap")
    plt.ylabel("Perturbation")
    plt.xlabel("Model")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved heatmap to {out_path}")


def calculate_ppi_aggregate(dataset: str, model_names=None, perturbations=None):
    """PPI tables and all PPI plots for one dataset."""
    ppi_mean_df, ppi_std_df = calculate_ppi_df(dataset, model_names, perturbations)
    plot_mean_bar_chart(ppi_mean_df, ppi_std_df, dataset)
    plot_ppi_vs_param_scatter(ppi_mean_df, dataset)
    plot_ppi_heatmap(ppi_mean_df, dataset)
    return ppi_mean_df, ppi_std_df
