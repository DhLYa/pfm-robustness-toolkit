"""Cross-dataset summary figures built from the result workbooks in Stats/."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from scipy.stats import pearsonr

from .models import MODELS
from .paths import stats_dir


# ─── NR-Kfold per-fold figures ───────────────────────────────────────────────

def load_kfold_workbook(dataset: str) -> dict:
    """{model: per-fold DataFrame} from Stats/<dataset>_kfold_summary.xlsx."""
    xlsx_path = stats_dir(f"{dataset}_kfold_summary.xlsx")
    if not xlsx_path.is_file():
        raise FileNotFoundError(f"Missing workbook for dataset '{dataset}': {xlsx_path}")
    xfile = pd.ExcelFile(xlsx_path)
    sheets = {name: xfile.parse(name) for name in xfile.sheet_names}
    sheets.pop("SUMMARY", None)
    return sheets


def plot_fold_heatmap(dataset: str, model_tables: dict, metric: str = "test_roc_auc"):
    """Rows = models, columns = folds, annotated with `metric`."""
    all_folds = sorted({
        int(f) for df in model_tables.values()
        for f in pd.to_numeric(df.get("fold", pd.Series(dtype=float)), errors="coerce").dropna().unique()
    })
    if not all_folds:
        print(f"[{dataset}] No fold indices found; skipping heatmap.")
        return

    models = sorted(model_tables.keys())
    mat = np.full((len(models), len(all_folds)), np.nan, dtype=float)
    for i, m in enumerate(models):
        df = model_tables[m]
        if metric not in df.columns or "fold" not in df.columns:
            continue
        fold = pd.to_numeric(df["fold"], errors="coerce")
        vals = pd.to_numeric(df[metric], errors="coerce")
        keep = fold.notna() & vals.notna()
        for f, v in zip(fold[keep].astype(int), vals[keep].astype(float)):
            if f in all_folds:
                mat[i, all_folds.index(f)] = v

    if np.isnan(mat).all():
        print(f"[{dataset}] No valid '{metric}' values; skipping heatmap.")
        return

    fig, ax = plt.subplots(figsize=(max(8, len(all_folds) * 0.8), max(6, len(models) * 0.45)))
    im = ax.imshow(mat, aspect="auto", interpolation="nearest",
                   vmin=float(np.nanmin(mat)), vmax=float(np.nanmax(mat)), cmap="viridis")
    ax.set_xticks(range(len(all_folds)))
    ax.set_xticklabels(all_folds)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.set_xlabel("Fold")
    ax.set_ylabel("Model")
    ax.set_title(f"{dataset} — {metric} per fold")
    for i in range(len(models)):
        for j in range(len(all_folds)):
            val = mat[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8,
                        color="black" if im.norm(val) > 0.6 else "white")
    fig.colorbar(im, ax=ax).set_label(metric)
    fig.tight_layout()

    out_path = stats_dir("per_fold_visuals", f"{dataset}__{metric}__heatmap.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[{dataset}] Saved heatmap → {out_path}")


def plot_fold_boxplot(dataset: str, model_tables: dict, metric: str = "test_roc_auc"):
    models, data = [], []
    for m, df in model_tables.items():
        if metric not in df.columns:
            continue
        vals = pd.to_numeric(df[metric], errors="coerce").dropna().values
        if len(vals) > 0:
            models.append(m)
            data.append(vals)
    if not data:
        print(f"[{dataset}] No valid '{metric}' values; skipping boxplot.")
        return

    fig, ax = plt.subplots(figsize=(max(8, len(models) * 0.7), 6))
    ax.boxplot(data, labels=models, vert=True, showmeans=True)
    ax.set_ylabel(metric)
    ax.set_title(f"{dataset} — {metric} per fold (boxplot)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()

    out_path = stats_dir("per_fold_visuals", f"{dataset}__{metric}__boxplot.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[{dataset}] Saved boxplot → {out_path}")


def plot_kfold_vs_params(agg_xlsx=None):
    """Average NR-Kfold ROC-AUC and its std vs parameter count (dissertation Figure 44)."""
    agg_xlsx = agg_xlsx or stats_dir("aggregate_kfold_model_rankings.xlsx")
    agg = pd.read_excel(agg_xlsx, sheet_name="AGGREGATE_RANKINGS")
    agg = agg[agg["model"].isin(MODELS)]
    models = sorted(agg["model"], key=lambda m: (MODELS[m].params_m, m))
    agg = agg.set_index("model").loc[models]

    x = np.array([MODELS[m].params_m for m in models])
    y_auc = agg["avg_roc_auc_mean"].to_numpy(dtype=float)
    y_std = agg["avg_roc_auc_std"].to_numpy(dtype=float)
    cmap = plt.get_cmap('tab20')
    colors = {m: cmap(i % 20) for i, m in enumerate(models)}
    logx = np.log10(x)
    order = np.argsort(x)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True)
    for ax, y, ylabel, title in [
        (axes[0], y_auc, 'Average ROC-AUC across datasets', 'ROC-AUC vs Parameter Count'),
        (axes[1], y_std, 'STD of ROC-AUC across datasets', 'ROC-AUC STD vs Parameter Count'),
    ]:
        for m, xi, yi in zip(models, x, y):
            ax.scatter(xi, yi, s=60, color=colors[m], edgecolor='k', linewidth=0.5, zorder=3)
            ax.annotate(m, (xi, yi), textcoords="offset points", xytext=(4, 4), ha='left',
                        fontsize=8, color=colors[m])
        trend = np.poly1d(np.polyfit(logx, y, 1))(logx)
        ax.plot(x[order], trend[order], linestyle='--', linewidth=1, color='grey', alpha=0.8, zorder=2)
        r, _ = pearsonr(x, y)
        ax.set_xscale('log')
        ax.set_xlabel('Parameters (millions, log scale)')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{title} (r={r:.2f})')
        ax.grid(True, linestyle=':', alpha=0.4)
    axes[1].invert_yaxis()  # lower std = more consistent
    plt.tight_layout()

    out_path = stats_dir("per_fold_visuals", "kfold_roc_auc_vs_params.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ─── PPI 2x2 heatmaps ────────────────────────────────────────────────────────

def _add_mean_row_col(df, mean_label="Mean"):
    base = df.drop(columns=[mean_label], errors="ignore").drop(index=[mean_label], errors="ignore")
    out = base.copy()
    out[mean_label] = base.mean(axis=1, numeric_only=True)
    out.loc[mean_label] = list(base.mean(axis=0, numeric_only=True)) + [float(np.nanmean(base.to_numpy()))]
    return out


def read_ppi_matrix(dataset: str) -> pd.DataFrame:
    """perturbations x models PPI means from Stats/<dataset>/constraint_sweep/ppi_aggregate.xlsx."""
    xlsx = stats_dir(dataset, "constraint_sweep", "ppi_aggregate.xlsx")
    if not xlsx.is_file():
        raise FileNotFoundError(f"Missing: {xlsx}")
    df = pd.read_excel(xlsx, sheet_name="ppi_mean", header=0, index_col=0)
    df = df.apply(pd.to_numeric, errors="coerce")
    return df.sort_index(key=lambda s: s.astype(str).str.lower())


def plot_ppi_heatmaps(datasets, annotate=False, model_label_rot=60):
    """2x2 grid of PPI heatmaps (with Mean row/column); saved to Stats/PPI/ppi_heatmaps.png."""
    mats = {ds: read_ppi_matrix(ds) for ds in datasets}
    model_list = sorted(set().union(*[m.columns.astype(str) for m in mats.values()]))
    mats = {ds: df.reindex(columns=model_list) for ds, df in mats.items()}

    fig, axes = plt.subplots(2, 2, figsize=(22, 12), constrained_layout=True)
    axes = axes.ravel()
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="white")
    norm = Normalize(vmin=0.0, vmax=1.0)

    for idx, (ax, ds) in enumerate(zip(axes, datasets)):
        df = mats[ds]
        if df.empty:
            ax.axis("off")
            ax.set_title(f"{ds} (no data)", fontsize=18, fontweight="bold", pad=22)
            continue

        df_plot = _add_mean_row_col(df)
        arr = df_plot.to_numpy().astype(float)
        arr[-1, -1] = np.nan
        ax.imshow(np.ma.masked_invalid(arr), aspect="auto", cmap=cmap, norm=norm)

        ax.set_xticks(np.arange(df_plot.shape[1]))
        if idx < 2:
            ax.set_xticklabels(df_plot.columns.tolist(), rotation=model_label_rot, ha="left", fontsize=8)
            ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False)
            xt = ax.get_xticklabels()
            if xt:
                xt[-1].set_fontweight("bold")
        else:
            ax.set_xticklabels([])
            ax.tick_params(axis="x", top=False, labeltop=False, bottom=False, labelbottom=False)

        ax.set_yticks(np.arange(df_plot.shape[0]))
        if idx % 2 == 0:
            ax.set_yticklabels(df_plot.index.tolist(), fontsize=10)
            ax.set_ylabel("Perturbation")
            yt = ax.get_yticklabels()
            if yt:
                yt[-1].set_fontweight("bold")
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis="y", left=True, labelleft=False)

        ax.set_title(ds, fontsize=18)
        nrows, ncols = df_plot.shape
        ax.axhline(nrows - 1.5, color="k", lw=0.9)
        ax.axvline(ncols - 1.5, color="k", lw=0.9)

        if annotate:
            for i in range(nrows):
                for j in range(ncols):
                    if not np.isnan(arr[i, j]):
                        ax.text(j, i, f"{arr[i, j]:.3f}", ha="center", va="center", fontsize=8,
                                color="black" if norm(arr[i, j]) > 0.6 else "white")

    for k in range(len(datasets), 4):
        axes[k].axis("off")

    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, orientation="vertical",
                 fraction=0.03, pad=0.02).set_label("PPI")

    out_path = stats_dir("PPI", "ppi_heatmaps.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")
