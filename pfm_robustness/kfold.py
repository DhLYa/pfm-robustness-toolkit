"""Non-redundant K-fold (NR-Kfold) distribution-shift analysis (dissertation section 4.2).

1. ``build_folds``: embed train+val+test with a pretrained backbone, k-means each class
   separately, then assign whole clusters to K folds (greedy, size-balanced per class).
2. ``run_kfold_perceptron``: train a linear probe on K-1 folds, test on the held-out fold.
3. ``summarise_kfold``: per-dataset workbooks and cross-dataset rankings.

Outputs go to ``Stats/<dataset>/<model>/distribution_shift/`` and ``Stats/``.
"""

import gc
import glob
import json
import os
import random
import re

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from tqdm import tqdm

from .datasets import DATASETS, build_dataset, create_trainval_dict
from .embedders import KFOLD_EMBEDDERS, pil_transform
from .paths import default_device, ensure_dir, stats_dir
from .rankings import average_across_datasets, rank_desc

SEED = 42
CLUSTERS_PER_CLASS = 5
N_FOLDS = 5
TSNE_PERPLEXITY = 30
TSNE_ITER = 1000


def shift_dir(dataset, model):
    return stats_dir(dataset, model, "distribution_shift")


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ─── 1. embeddings, clusters, folds ──────────────────────────────────────────

def build_loaders_for_dataset(dataset: str):
    """train / val / test loaders of 0-1 tensors (per-model preprocessing happens later)."""
    spec = DATASETS[dataset]
    b = spec.budgets["kfold"]
    trainval = create_trainval_dict(
        spec.cls, spec.train_path(), b["batch_size"], transforms.Compose([transforms.ToTensor()]),
        b["trainval_multiplier"], b["trainval_size"], shuffle=False, **spec.train_kwargs,
    )
    _, test_loader = build_dataset(
        spec.cls, spec.test_path(), b["batch_size"], b["test_multiplier"],
        transform=spec.tensor_transform(), shuffle=False, **spec.test_kwargs,
    )
    return {'train': trainval['train'], 'val': trainval['val'], 'test': test_loader}


def _to_pil(x):
    if isinstance(x, Image.Image):
        return x
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
        if x.ndim == 3:
            return transforms.ToPILImage()(x)
        if x.ndim == 4 and x.size(0) == 1:
            return transforms.ToPILImage()(x[0])
    raise TypeError(f"Unsupported image type: {type(x)} | shape: {getattr(x, 'shape', None)}")


def _prepare_batch(img_batch, transform):
    imgs = [img_batch[i] for i in range(img_batch.size(0))] if isinstance(img_batch, torch.Tensor) else list(img_batch)
    processed = []
    for t in imgs:
        out = transform(_to_pil(t))
        processed.append(torch.from_numpy(out) if isinstance(out, np.ndarray) else out)
    return torch.stack(processed, dim=0)


@torch.no_grad()
def extract_embeddings(dataset: str, model_name: str, loaders_by_split, device):
    """Embed every split; saves embeddings.h5 + metadata.csv. Returns (features, metadata)."""
    save_root = shift_dir(dataset, model_name)
    ensure_dir(save_root / "embeddings")

    embedder = KFOLD_EMBEDDERS[model_name]
    model = embedder.init(device)
    model_transform = pil_transform(model_name, model)

    all_feats, all_labels, all_names, all_splits = [], [], [], []
    for split, loader in loaders_by_split.items():
        for imgs, labels, names in tqdm(loader, desc=f"{dataset}/{model_name} [{split}]"):
            labels = labels.detach().cpu().numpy().tolist() if isinstance(labels, torch.Tensor) else list(labels)
            feats = np.asarray(embedder.embed(model, _prepare_batch(imgs, model_transform), device))
            if feats.ndim == 1:
                feats = feats[None, :]
            all_feats.append(feats)
            all_labels.extend(labels)
            all_names.extend(list(names))
            all_splits.extend([split] * len(labels))

    feats = np.concatenate(all_feats, axis=0) if all_feats else np.zeros((0, 1))
    meta = pd.DataFrame({
        "name": all_names, "label": all_labels, "split": all_splits,
        "dataset": dataset, "model": model_name,
    })

    with h5py.File(save_root / "embeddings" / "embeddings.h5", "w") as h5:
        h5.create_dataset("features", data=feats, compression="gzip")
        h5.create_dataset("labels", data=np.array(all_labels, dtype=np.int64))
        str_dt = h5py.string_dtype(encoding='utf-8')
        h5.create_dataset("names", data=np.array(all_names, dtype=object), dtype=str_dt)
        h5.create_dataset("splits", data=np.array(all_splits, dtype=object), dtype=str_dt)
    meta.to_csv(save_root / "embeddings" / "metadata.csv", index=False)

    del model
    torch.cuda.empty_cache()
    gc.collect()
    return feats, meta


def cluster_by_class(feats, labels, k_per_class=CLUSTERS_PER_CLASS):
    """k-means within each class; returns (global cluster id per sample, {cluster: size})."""
    labels = np.array(labels)
    cluster_ids = np.full(feats.shape[0], fill_value=-1, dtype=int)
    global_id = 0
    sizes = {}
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        if len(idx) == 0:
            continue
        k = min(k_per_class, len(idx))
        if k <= 1:
            cluster_ids[idx] = global_id
            sizes[global_id] = len(idx)
            global_id += 1
            continue
        cid = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(feats[idx])
        for local in range(k):
            sel = idx[cid == local]
            cluster_ids[sel] = global_id
            sizes[global_id] = len(sel)
            global_id += 1
    return cluster_ids, sizes


def assign_folds_from_clusters(meta_with_clusters: pd.DataFrame, n_folds=N_FOLDS):
    """Greedily give each class's clusters (largest first) to the currently smallest fold."""
    df = meta_with_clusters.copy()
    folds = np.full(len(df), -1, dtype=int)
    for cls in sorted(df["label"].unique()):
        sub = df[df["label"] == cls]
        groups = sub.groupby("cluster_id").indices
        fold_sizes = [0] * n_folds
        for _, idxs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            f = int(np.argmin(fold_sizes))
            fold_sizes[f] += len(idxs)
            folds[sub.index.values[idxs]] = f
    assert np.all(folds >= 0)
    df["fold"] = folds
    return df


def save_clusters_and_folds(meta_folds: pd.DataFrame, dataset, model, n_folds=N_FOLDS):
    out_dir = shift_dir(dataset, model)
    meta_folds.drop(columns=["fold"]).to_csv(out_dir / "clusters.csv", index=False)
    sizes = meta_folds["cluster_id"].value_counts().sort_index().to_dict()
    with open(out_dir / "cluster_sizes.json", "w") as f:
        json.dump({str(k): int(v) for k, v in sizes.items()}, f, indent=2)

    ensure_dir(out_dir / "folds")
    meta_folds.to_csv(out_dir / "folds" / f"fold_assignments_k{n_folds}.csv", index=False)
    meta_folds.groupby(["fold", "label"]).size().unstack(fill_value=0) \
              .to_csv(out_dir / "folds" / f"fold_class_counts_k{n_folds}.csv")


def plot_tsne_folds(feats, meta_with_folds, dataset, model, perplexity=TSNE_PERPLEXITY,
                    n_iter=TSNE_ITER, show_counts=True):
    """t-SNE (no PCA) coloured by fold and by class."""
    out_dir = ensure_dir(shift_dir(dataset, model) / "tsne")
    X2 = TSNE(n_components=2, perplexity=perplexity, max_iter=n_iter, random_state=SEED,
              init="random", learning_rate="auto").fit_transform(feats)
    df = meta_with_folds.copy()
    df["x"], df["y"] = X2[:, 0], X2[:, 1]

    plt.figure(figsize=(8, 6))
    cmap = plt.get_cmap("tab20")
    for i, f in enumerate(sorted(df["fold"].unique())):
        sub = df[df["fold"] == f]
        plt.scatter(sub["x"], sub["y"], s=8, alpha=0.85, color=cmap(i % 20),
                    label=f"Fold {f}" + (f" (n={len(sub)})" if show_counts else ""))
    plt.title(f"{model} {dataset} t-SNE by Fold")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "tsne_byfold.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 6))
    class_color = {0: "tab:blue", 1: "tab:red"}
    class_name = {0: "FALSE/Normal", 1: "TRUE/Tumor"}
    for c in sorted(df["label"].unique()):
        sub = df[df["label"] == c]
        plt.scatter(sub["x"], sub["y"], s=8, alpha=0.85, color=class_color.get(c, "k"),
                    label=class_name.get(c, f"Class {c}") + (f" (n={len(sub)})" if show_counts else ""))
    plt.title(f"{model} {dataset} t-SNE by Class")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "tsne_byclass.png", dpi=150)
    plt.close()


def build_folds(datasets, model_names, device=None, n_folds=N_FOLDS, k_per_class=CLUSTERS_PER_CLASS):
    """Embeddings -> per-class clusters -> folds -> t-SNE, for every dataset x model."""
    device = device or default_device()
    seed_everything()
    for dataset in datasets:
        loaders = build_loaders_for_dataset(dataset)
        for model in model_names:
            print(f"\n=== {dataset} :: {model} ===")
            feats, meta = extract_embeddings(dataset, model, loaders, device)
            cluster_ids, _ = cluster_by_class(feats, meta["label"].tolist(), k_per_class)
            meta["cluster_id"] = cluster_ids
            meta_folds = assign_folds_from_clusters(meta, n_folds=n_folds)
            save_clusters_and_folds(meta_folds, dataset, model, n_folds)
            plot_tsne_folds(feats, meta_folds, dataset, model)


# ─── 2. linear probe per fold ────────────────────────────────────────────────

def _to_2d_features(arr) -> np.ndarray:
    x = np.asarray(arr)
    if x.ndim == 1:
        x = x[None, :]
    elif x.ndim == 2:
        return x
    elif x.ndim == 3:
        x = x.mean(axis=1)
    elif x.ndim == 4:
        x = x.mean(axis=(2, 3))
    else:
        raise ValueError(f"Unsupported feature shape {x.shape}")
    return x


def _normalize_str_array(a):
    s = pd.Series(a).apply(lambda v: v.decode() if isinstance(v, (bytes, bytearray)) else v)
    return s.astype(str).str.strip().to_numpy()


def _canonicalize_name(name) -> str:
    name = str(name).replace("\\", "/").strip().lower()
    base = name.split("/")[-1]
    if "." in base:
        base = ".".join(base.split(".")[:-1])
    return base


def _find_folds_file(save_root):
    cands = glob.glob(os.path.join(save_root, "folds", "fold_assignments_k*.csv"))
    if not cands:
        raise FileNotFoundError(f"No folds file in {os.path.join(save_root, 'folds')}")

    def _k(p):
        try:
            return int(os.path.basename(p).split("fold_assignments_k", 1)[1].split(".csv", 1)[0])
        except (IndexError, ValueError):
            return -1

    best = max(cands, key=_k)
    return best, _k(best)


def load_embeddings_and_folds(dataset, model):
    """Embeddings aligned to fold assignments by (canonicalised) sample name."""
    root = shift_dir(dataset, model)
    h5_path = root / "embeddings" / "embeddings.h5"
    if not h5_path.is_file():
        raise FileNotFoundError(f"Missing embeddings: {h5_path}")
    with h5py.File(h5_path, "r") as h5:
        X = _to_2d_features(np.array(h5["features"]))
        y = np.array(h5["labels"]).astype(int)
        names_raw = _normalize_str_array(np.array(h5["names"]))
        splits_raw = _normalize_str_array(np.array(h5["splits"])) if "splits" in h5 else np.array(["unknown"] * len(y))
    names_can = np.array([_canonicalize_name(n) for n in names_raw])

    folds_csv, K = _find_folds_file(root)
    df_folds = pd.read_csv(folds_csv)
    if not {"name", "fold"}.issubset(df_folds.columns):
        raise ValueError(f"{folds_csv} must contain columns ['name','fold'].")
    df_folds["name"] = _normalize_str_array(df_folds["name"].to_numpy())
    df_folds["name_can"] = df_folds["name"].map(_canonicalize_name)
    df_folds["fold"] = pd.to_numeric(df_folds["fold"], errors="coerce")
    df_folds = df_folds.drop_duplicates(subset=["name_can"], keep="first")

    df_embed = pd.DataFrame({"row_idx": np.arange(len(names_raw)), "name_raw": names_raw, "name_can": names_can})
    df_merge = df_embed.merge(df_folds[["name_can", "fold"]], on="name_can", how="left")

    if int(df_merge["fold"].isna().sum()) > 0:
        diag_dir = ensure_dir(root / "folds" / "_alignment_debug")
        df_merge.loc[df_merge["fold"].isna()].to_csv(diag_dir / "unmatched_embeddings.csv", index=False)
        df_folds.loc[~df_folds["name_can"].isin(df_merge["name_can"])].to_csv(diag_dir / "unmatched_folds.csv", index=False)

        matched = df_merge.dropna(subset=["fold"]).copy()
        keep_idx = matched["row_idx"].to_numpy().astype(int)
        X, y = X[keep_idx], y[keep_idx]
        names_raw, splits_raw = names_raw[keep_idx], splits_raw[keep_idx]
        folds = matched["fold"].astype(int).to_numpy()
    else:
        folds = df_merge.sort_values("row_idx")["fold"].astype(int).to_numpy()

    assert len(X) == len(y) == len(names_raw) == len(folds), \
        f"Length mismatch after alignment: X={len(X)} y={len(y)} names={len(names_raw)} folds={len(folds)}"
    return X, y, names_raw, splits_raw, folds, K, root


def compute_metrics(y_true, scores) -> dict:
    classes = np.unique(y_true)
    if scores.ndim == 1:
        y_prob = scores
        y_pred = (y_prob >= 0.5).astype(int)
        acc = accuracy_score(y_true, y_pred)
        try:
            roc = roc_auc_score(y_true, y_prob)
        except Exception:
            roc = np.nan
        try:
            pr = average_precision_score(y_true, y_prob)
        except Exception:
            pr = np.nan
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        tn = fp = fn = tp = 0
        if cm.shape == (2, 2):
            tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
        return dict(accuracy=acc, roc_auc=roc, pr_auc=pr, precision=prec, recall=rec, f1=f1,
                    tn=tn, fp=fp, fn=fn, tp=tp)
    y_pred = scores.argmax(axis=1)
    acc = accuracy_score(y_true, y_pred)
    Yb = label_binarize(y_true, classes=classes)
    try:
        roc = roc_auc_score(Yb, scores, average="macro", multi_class="ovr")
    except Exception:
        roc = np.nan
    try:
        pr = average_precision_score(Yb, scores, average="macro")
    except Exception:
        pr = np.nan
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return dict(accuracy=acc, roc_auc=roc, pr_auc=pr, precision=prec, recall=rec, f1=f1)


def _probs(model, X_t, device, n_classes):
    with torch.no_grad():
        p = torch.softmax(model(X_t.to(device)), dim=1)
    return (p[:, 1] if n_classes == 2 else p).cpu().numpy()


def run_kfold_perceptron(dataset, model, epochs=25, batch_size=64, lr=1e-3, weight_decay=1e-4,
                         val_fraction=0.3, patience=5, class_weight=None, seed=SEED):
    """Linear probe on standardised embeddings, early-stopped on a stratified val split."""
    X, y, _, _, folds, K, root = load_embeddings_and_folds(dataset, model)
    n_classes = len(np.unique(y))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fold_results = []
    for fold_id in range(K):
        train_idx = np.where(folds != fold_id)[0]
        test_idx = np.where(folds == fold_id)[0]

        if val_fraction and val_fraction > 0.0 and len(np.unique(y[train_idx])) > 1:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
            tr_sub_idx, val_sub_idx = next(sss.split(X[train_idx], y[train_idx]))
            tr_idx, val_idx = train_idx[tr_sub_idx], train_idx[val_sub_idx]
        else:
            tr_idx, val_idx = train_idx, None

        # Standardise using the training part only
        scaler = StandardScaler(with_mean=True, with_std=True)
        X_train_t = torch.tensor(scaler.fit_transform(X[tr_idx]), dtype=torch.float32)
        y_train_t = torch.tensor(y[tr_idx], dtype=torch.long)
        X_test_t = torch.tensor(scaler.transform(X[test_idx]), dtype=torch.float32)
        y_test_t = torch.tensor(y[test_idx], dtype=torch.long)
        if val_idx is not None:
            X_val_t = torch.tensor(scaler.transform(X[val_idx]), dtype=torch.float32)
            y_val_t = torch.tensor(y[val_idx], dtype=torch.long)

        probe = nn.Sequential(nn.Linear(X.shape[1], n_classes)).to(device)
        if class_weight == "balanced":
            cw = compute_class_weight(class_weight="balanced", classes=np.arange(n_classes), y=y[tr_idx])
            criterion = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32, device=device))
        else:
            criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)
        train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)

        best_state, best_auc, epochs_no_improve = None, -np.inf, 0
        for _ in range(epochs):
            probe.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                criterion(probe(xb), yb).backward()
                optimizer.step()

            if val_idx is not None:
                probe.eval()
                val_auc = compute_metrics(y_val_t.numpy(), _probs(probe, X_val_t, device, n_classes)).get("roc_auc", -np.inf)
                if val_auc > best_auc:
                    best_auc = val_auc
                    best_state = {k: v.detach().cpu().clone() for k, v in probe.state_dict().items()}
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if patience and epochs_no_improve >= patience:
                        break
            else:
                best_state = {k: v.detach().cpu().clone() for k, v in probe.state_dict().items()}

        if best_state is not None:
            probe.load_state_dict(best_state, strict=True)
        probe.eval()

        train_metrics = compute_metrics(y_train_t.numpy(), _probs(probe, X_train_t, device, n_classes))
        test_metrics = compute_metrics(y_test_t.numpy(), _probs(probe, X_test_t, device, n_classes))
        fold_results.append({
            "fold": int(fold_id),
            "n_train": int(len(tr_idx)),
            "n_test": int(len(test_idx)),
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"test_{k}": v for k, v in test_metrics.items()},
        })

    fold_df = pd.DataFrame(fold_results)
    test_metric_names = [c.replace("test_", "") for c in fold_df.columns if c.startswith("test_")]
    summary = {m: {"mean": float(fold_df[f"test_{m}"].mean()), "std": float(fold_df[f"test_{m}"].std())}
               for m in test_metric_names}
    summary["n_per_fold"] = {str(int(f)): int(n) for f, n in zip(fold_df["fold"], fold_df["n_test"])}

    save_dir = ensure_dir(root / "kfold_embed_perceptron")
    fold_df.to_csv(save_dir / f"metrics_k{K}.csv", index=False)
    with open(save_dir / f"summary_k{K}.json", "w") as f:
        json.dump(summary, f, indent=2)
    return fold_df, summary


# ─── 3. summaries ────────────────────────────────────────────────────────────

def _safe_sheet_name(name: str) -> str:
    return re.sub(r'[\[\]\*\?\/\\:]', '_', name)[:31]


def summarise_kfold(datasets, model_names, **perceptron_kwargs):
    """
    Runs the perceptron for every dataset x model and writes
    ``Stats/<dataset>_kfold_summary.xlsx`` and ``Stats/aggregate_kfold_model_rankings.xlsx``.
    """
    results = {ds: [] for ds in datasets}
    fold_tables = {ds: {} for ds in datasets}

    for dataset in tqdm(datasets, desc="Datasets"):
        for model in tqdm(model_names, desc=f"Models in {dataset}", leave=False):
            tqdm.write(f"\n===> K-FOLD PERCEPTRON (embeddings) | {dataset} :: {model}")
            try:
                fold_df, summary = run_kfold_perceptron(dataset, model, **perceptron_kwargs)
            except Exception as e:
                tqdm.write(f"Skipped {dataset}/{model}: {e}")
                continue
            fold_tables[dataset][model] = fold_df.copy()
            row = {"dataset": dataset, "model": model}
            for k, v in summary.items():
                if k != "n_per_fold":
                    row[f"{k}_mean"] = v["mean"]
                    row[f"{k}_std"] = v["std"]
            results[dataset].append(row)

    aggregate_rows = []
    for dataset, rows in results.items():
        if not rows:
            tqdm.write(f"No results for {dataset}; skipping outputs.")
            continue
        summary_df = pd.DataFrame(rows)
        ranks = rank_desc(summary_df["roc_auc_mean"]) if "roc_auc_mean" in summary_df else pd.Series(np.nan, index=summary_df.index)
        summary_df.insert(0, "rank", ranks.astype(pd.Int64Dtype()))
        summary_df = summary_df[["rank", "model"] + [c for c in summary_df.columns if c not in ("rank", "model")]]

        for _, r in summary_df.iterrows():
            aggregate_rows.append({
                "dataset": dataset,
                "model": r["model"],
                "roc_auc_mean": r.get("roc_auc_mean", np.nan),
                "roc_auc_std": r.get("roc_auc_std", np.nan),
                "rank_within_dataset": np.nan if pd.isna(r["rank"]) else int(r["rank"]),
            })

        xlsx_path = stats_dir(f"{dataset}_kfold_summary.xlsx")
        with pd.ExcelWriter(xlsx_path) as xw:
            summary_df.sort_values(by=["rank", "roc_auc_mean"], ascending=[True, False]) \
                      .to_excel(xw, sheet_name="SUMMARY", index=False)
            for model, fold_df in fold_tables[dataset].items():
                fold_df.to_excel(xw, sheet_name=_safe_sheet_name(model), index=False)
        tqdm.write(f"Saved dataset workbook with sheets to: {xlsx_path}")

    df_agg = pd.DataFrame(aggregate_rows)
    if df_agg.empty:
        tqdm.write("No aggregate data to save.")
        return None

    df_final = average_across_datasets(df_agg, "roc_auc_mean", "roc_auc_std", "rank_within_dataset",
                                       out_names=("avg_roc_auc_mean", "avg_roc_auc_std", "avg_rank"))
    agg_xlsx = stats_dir("aggregate_kfold_model_rankings.xlsx")
    with pd.ExcelWriter(agg_xlsx) as xw:
        df_final.to_excel(xw, sheet_name="AGGREGATE_RANKINGS", index=False, float_format="%.3f")
        df_agg.sort_values(by=["dataset", "rank_within_dataset"], ascending=[True, True]) \
              .to_excel(xw, sheet_name="PER_DATASET_RANKS", index=False, float_format="%.3f")
    tqdm.write(f"Saved aggregate rankings to: {agg_xlsx}")
    return df_final
