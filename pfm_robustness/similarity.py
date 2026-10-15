"""Train–test embedding similarity under perturbation (dissertation section 4.1).

For a pretrained backbone, the mean cosine similarity between (perturbed) test embeddings
and clean train embeddings is traced across each perturbation's sweep range, together with a
joint t-SNE. Writes ``Stats/<dataset>/<model>/embeddings/<perturbation>_{stats,tsne}.xlsx``
and plots; ``plot_similarity_grid`` summarises all models in
``Stats/<dataset>/similarity_aggregate/<dataset>_similarity_grid.png``.
"""

import copy
import glob
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import transforms
from tqdm import tqdm
from transformers.modeling_outputs import BaseModelOutputWithPooling

from .datasets import DATASETS, build_dataset, create_trainval_dict
from .embedders import SIMILARITY_BACKBONES, pil_transform
from .models import MODELS, model_colour
from .paths import default_device, stats_dir
from .perturbations import SWEEPS
from .reet.optimisers import untargeted_loss
from .sweep import apply_sweep_value

# Plot conventions for similarity curves
DROP_ZERO = {"crop", "jpeg", "zoom_out", "zoom_in"}
INVERT_AXIS = {"crop", "zoom_out", "jpeg"}
BASELINE_X = {"jpeg": 100.0, "zoom_out": 1.0}  # display position of the unperturbed baseline
CROP_ANCHOR_STRENGTH = 225.0
SIMILARITY_EXCLUDED = ("H0-mini", "Virchow", "Virchow2")  # excluded in the dissertation


def hf_untargeted_loss(outputs, labels):
    """untargeted_loss on whatever a HF / timm backbone returns, reduced to (B, C)."""
    if hasattr(outputs, "logits"):
        logits = outputs.logits
    elif isinstance(outputs, BaseModelOutputWithPooling) or hasattr(outputs, "pooler_output"):
        logits = outputs.pooler_output
    elif hasattr(outputs, "last_hidden_state"):
        logits = outputs.last_hidden_state[:, 0, :]
    elif isinstance(outputs, tuple) and isinstance(outputs[0], torch.Tensor):
        logits = outputs[0]
    elif isinstance(outputs, torch.Tensor):
        logits = outputs
    else:
        raise ValueError(f"Don't know how to unwrap model output of type {type(outputs)}")
    if logits.ndim == 4:
        logits = F.adaptive_avg_pool2d(logits, (1, 1)).view(logits.size(0), -1)
    return untargeted_loss(logits, labels)


class EmbeddingEvaluator:
    """Attaches a REET attack to a backbone and extracts (optionally perturbed) embeddings."""

    def __init__(self, model, model_name, TransformOptimiser, Transform, optimiser_params, trans_params,
                 criterion, device="cuda:0", post_transform=None):
        self.model = model.to(device).eval()
        self.model_name = model_name
        self.device = device
        self.criterion = criterion
        self.post_transform = post_transform
        self.set_attack(TransformOptimiser, Transform, optimiser_params, trans_params)

    def set_attack(self, TransformOptimiser, Transform, optimiser_params, trans_params):
        self.attack = TransformOptimiser(self.model, Transform, optimiser_params, trans_params,
                                         criterion=self.criterion, device=self.device)

    def _safe_post_transform(self, imgs):
        if self.post_transform is None:
            return imgs
        if isinstance(imgs, torch.Tensor):
            return imgs
        if isinstance(imgs, (list, tuple)):
            return torch.stack([self._safe_post_transform(i) for i in imgs])
        return self.post_transform(imgs)

    def extract_embeddings(self, dataloader, progbar_desc="Extracting embeddings", apply_attack=True) -> np.ndarray:
        """
        1) optionally attack (true labels); 2) clamp to 0-1, to PIL, apply post_transform;
        3) forward, taking hidden_states[-1] if available; 4) pool/flatten to (N, D).
        """
        self.model.eval()
        all_feats = []
        to_pil = transforms.ToPILImage()
        orig_hyp = copy.deepcopy(self.attack.hyperparameters)

        for imgs, labels, *_ in tqdm(dataloader, desc=progbar_desc, leave=False):
            imgs = imgs.to(self.device)
            labels = labels.to(self.device)

            if apply_attack:
                imgs = imgs.clone().detach().requires_grad_(True)
                try:
                    _, imgs = self.attack.optimise(imgs, targets=labels, reset_weights=True)
                except Exception as e:
                    print(f"[!] {self.model_name} attack failed: {e}. using clean imgs.")
                    imgs = imgs.detach()

            with torch.no_grad():
                processed = []
                for img in imgs.cpu():
                    out = self._safe_post_transform(to_pil(img.clamp(0, 1)))
                    if isinstance(out, dict) and "pixel_values" in out:
                        out = out["pixel_values"].squeeze(0)
                    processed.append(out)
                batch = torch.stack(processed).to(self.device)

                try:
                    out = self.model(batch, output_hidden_states=True)
                except TypeError:
                    out = self.model(batch)

                if hasattr(out, "hidden_states"):
                    feats = out.hidden_states[-1]
                elif isinstance(out, tuple):
                    feats = out[0]
                else:
                    feats = out

                if feats.ndim == 4:
                    feats = F.adaptive_avg_pool2d(feats, (1, 1)).view(feats.size(0), -1)
                else:
                    feats = feats.view(feats.size(0), -1)
                all_feats.append(feats.cpu())

        self.attack.hyperparameters = orig_hyp
        return torch.cat(all_feats, dim=0).numpy()


def sweep_embedding_similarity(evaluator, train_emb, test_loader, perturb_name, sweep, output_dir,
                               tsne_perplexity=30, tsne_randstate=42) -> pd.DataFrame:
    """
    Baseline + 10 equal steps over `sweep.range`: mean cosine similarity to the train
    embeddings at each strength, then a joint t-SNE of train + every test set.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    orig_hyp = copy.deepcopy(evaluator.attack.hyperparameters)

    start, end = sweep.range
    step_size = abs(start - end) / 10
    if start > end:
        strengths = np.arange(start, end - step_size, -step_size)
    else:
        strengths = np.arange(start, end + step_size, step_size)
    print(strengths)

    baseline_emb = evaluator.extract_embeddings(test_loader, progbar_desc="Test emb baseline", apply_attack=False)
    means = [cosine_similarity(baseline_emb, train_emb).mean()]
    tsne_batches = [(0.0, train_emb), (0.0, baseline_emb)]

    for s in tqdm(strengths, desc=f"Sweeping {perturb_name}"):
        evaluator.attack.hyperparameters = apply_sweep_value(copy.deepcopy(orig_hyp), sweep.param, s)
        emb = evaluator.extract_embeddings(test_loader, progbar_desc=f"Test emb {perturb_name}={s:.2f}",
                                           apply_attack=True)
        means.append(cosine_similarity(emb, train_emb).mean())
        tsne_batches.append((float(s), emb))

    evaluator.attack.hyperparameters = orig_hyp

    all_strengths = np.concatenate(([0.0], strengths))
    df_stats = pd.DataFrame({'strength': all_strengths, 'mean_sim': means})
    stats_path = output_dir / f"{perturb_name}_stats.xlsx"
    df_stats.to_excel(stats_path, index=False)
    print(f"Saved stats -> {stats_path}")

    all_emb = np.vstack([emb for _, emb in tsne_batches])
    tsne_out = TSNE(n_components=2, perplexity=tsne_perplexity, random_state=tsne_randstate).fit_transform(all_emb)

    records, offset = [], 0
    for idx, (strength, emb) in enumerate(tsne_batches):
        split = 'train' if idx == 0 else 'test'
        n = emb.shape[0]
        for x, y in tsne_out[offset:offset + n]:
            records.append({'strength': strength, 'split': split, 'tsne1': x, 'tsne2': y})
        offset += n

    tsne_path = output_dir / f"{perturb_name}_tsne.xlsx"
    pd.DataFrame(records).to_excel(tsne_path, index=False)
    print(f"Saved t-SNE coords -> {tsne_path}")
    return df_stats


def run_similarity_sweeps(dataset: str, model_names, sweeps=None, device=None):
    """Similarity sweeps for each pretrained backbone (no trained heads needed)."""
    device = device or default_device()
    sweeps = sweeps or SWEEPS
    spec = DATASETS[dataset]
    budget = spec.budgets["embed"]

    for name in model_names:
        print(f"\n=== Embedding sweeps for {name} on {dataset} ===")
        model = SIMILARITY_BACKBONES[name](device)
        post_tr = pil_transform(name)

        train_loader = create_trainval_dict(
            spec.cls, spec.train_path(), budget["batch_size"], post_tr,
            budget["trainval_multiplier"], budget["trainval_size"], **spec.train_kwargs,
        )["train"]
        _, test_loader = build_dataset(
            spec.cls, spec.test_path(), budget["batch_size"], budget["test_multiplier"],
            transform=post_tr, shuffle=False, **spec.test_kwargs,
        )
        print(f"[{dataset}] Test batches: {len(test_loader)}")

        first = next(iter(sweeps.values())).perturbation
        evaluator = EmbeddingEvaluator(
            model, name, first.optimiser, first.transform, first.optimiser_params, first.transform_params,
            criterion=hf_untargeted_loss, device=device, post_transform=post_tr,
        )
        train_emb = evaluator.extract_embeddings(train_loader, progbar_desc="Train emb", apply_attack=False)
        print(f"[{name}] train_emb shape: {train_emb.shape}")

        outdir = stats_dir(dataset, name, "embeddings")
        for pert_name, sw in sweeps.items():
            print(f"→ {name} embedding sweep '{pert_name}'")
            p = sw.perturbation
            evaluator.set_attack(p.optimiser, p.transform, p.optimiser_params, p.transform_params)
            sweep_embedding_similarity(evaluator, train_emb, test_loader, pert_name, sw, outdir)

        del model
        torch.cuda.empty_cache()


# ─── plots ───────────────────────────────────────────────────────────────────

def _ellipse(ax, pts, col, n_std=1.0, **kw):
    mu = pts.mean(axis=0)
    vals, vecs = np.linalg.eigh(np.cov(pts, rowvar=False))
    order = vals.argsort()[::-1]
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(vals[order])
    ax.add_patch(mpatches.Ellipse(mu, width, height, angle=angle, fill=False, edgecolor=col, **kw))
    ax.scatter(mu[0], mu[1], marker="X", s=100, c=[col], edgecolor="black")


def plot_embedding_sweeps(dataset: str, model_name: str, perturbations=None, plot_ellipse=True,
                          ellipse_std=1.0, normalize_y=True, fixed_y_range=False, tsne_step=2,
                          figsize_similarity=(8, 4), figsize_tsne=(10, 7)):
    """Per-perturbation similarity curve and t-SNE plot from the saved xlsx files."""
    embeddings_dir = stats_dir(dataset, model_name, "embeddings")
    label = model_name.lower()
    stats_paths = sorted(glob.glob(os.path.join(embeddings_dir, "*_stats.xlsx")))
    tsne_map = {os.path.basename(p).split("_tsne.xlsx")[0]: p
                for p in glob.glob(os.path.join(embeddings_dir, "*_tsne.xlsx"))}

    for stat_path in stats_paths:
        perturb = os.path.basename(stat_path).removesuffix("_stats.xlsx")
        if perturbations and perturb not in perturbations:
            continue

        df = pd.read_excel(stat_path)
        x_all = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy()
        y_all = pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy()

        if perturb in BASELINE_X:
            baseline_y = float(y_all[x_all == 0.0][0])
            mask = x_all != 0.0
            x = np.concatenate(([BASELINE_X[perturb]], x_all[mask]))
            y = np.concatenate(([baseline_y], y_all[mask]))
        else:
            mask = x_all != 0.0 if perturb in DROP_ZERO else np.ones_like(x_all, bool)
            x, y = x_all[mask], y_all[mask]
            baseline_y = y[0]

        if normalize_y:
            y = y / baseline_y

        fig, ax = plt.subplots(figsize=figsize_similarity)
        if perturb in BASELINE_X:
            ax.scatter([x[0]], [y[0]], marker="o", s=50, label="baseline")
            ax.plot(x[1:], y[1:], marker="o", linewidth=1)
        else:
            ax.plot(x, y, marker="o", linewidth=1)
        ax.set_title(f"{label.upper()} — {perturb}: Cosine similarity")
        ax.set_xlabel("strength")
        ax.set_ylabel("normalized cosine sim" if normalize_y else "mean cosine sim")
        ax.grid(True)
        if fixed_y_range:
            ax.set_ylim(0.0, 1.0)
        if perturb in INVERT_AXIS and x.size:
            ax.invert_xaxis()
        fig.tight_layout()
        fig.savefig(embeddings_dir / f"{perturb}_similarity_{label}.png")
        plt.close(fig)

        tsne_path = tsne_map.get(perturb)
        if not tsne_path:
            print(f"Skipping t-SNE for '{perturb}': coords missing.")
            continue

        df_tsne = pd.read_excel(tsne_path).astype({"tsne1": float, "tsne2": float})
        all_groups = list(df_tsne.groupby(["split", "strength"], sort=False))
        # keep train + test baselines, then every `tsne_step`-th strength (and the last)
        selected = all_groups[:2] + [
            grp for idx, grp in enumerate(all_groups[2:], start=2)
            if ((idx - 2) % tsne_step) == 0 or idx == len(all_groups) - 1
        ]
        n = len(selected)
        cmap = plt.cm.viridis_r

        fig, ax = plt.subplots(figsize=figsize_tsne)
        for idx, ((split, strg), sub) in enumerate(selected):
            pts = sub[["tsne1", "tsne2"]].to_numpy()
            col = cmap(idx / (n - 1)) if n > 1 else cmap(0.0)
            ax.scatter(pts[:, 0], pts[:, 1], c=[col], label=f"{split} {strg:.2f}", s=10, alpha=0.4)
            if plot_ellipse and pts.shape[0] > 2:
                _ellipse(ax, pts, col, ellipse_std, lw=2)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.set_title(f"{label.upper()} — {perturb}: t-SNE embeddings")
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        fig.tight_layout()
        fig.savefig(embeddings_dir / f"{perturb}_tsne_{label}.png")
        plt.close(fig)
        print(f"Saved plots for {model_name}/{perturb}")


def _load_normalized_curve(stat_path, perturb):
    df = pd.read_excel(stat_path)
    x_all = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy()
    y_all = pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy()
    baseline_y = float(y_all[x_all == 0.0][0]) if np.any(x_all == 0.0) else float(y_all[0])

    if perturb in BASELINE_X:
        mask = x_all != 0.0
        x = np.concatenate(([BASELINE_X[perturb]], x_all[mask]))
        y = np.concatenate(([baseline_y], y_all[mask]))
    else:
        mask = (x_all != 0.0) if perturb in DROP_ZERO else np.ones_like(x_all, dtype=bool)
        x, y = x_all[mask], y_all[mask]

    if perturb == "crop":
        if x.size:
            idx_anchor = np.where(np.isclose(x, CROP_ANCHOR_STRENGTH, atol=1e-6))[0]
            denom = float(y[idx_anchor[0]]) if idx_anchor.size > 0 else float(y[0])
        else:
            denom = 1.0
    else:
        denom = baseline_y
    y_norm = y / (denom if denom != 0 else 1.0)

    order = np.argsort(x)
    return x[order], y_norm[order]


def plot_similarity_grid(dataset: str, exclude=SIMILARITY_EXCLUDED, perturb_order=None):
    """3x3 grid of normalised similarity curves for every model with results."""
    ds_dir = stats_dir(dataset)
    models = [m for m in os.listdir(ds_dir)
              if (ds_dir / m / "embeddings").is_dir() and m not in exclude]
    order = list(MODELS)
    models.sort(key=lambda m: (order.index(m) if m in order else 999, m))
    if not models:
        print(f"[WARN] No models with embeddings for '{dataset}'. Skipping.")
        return

    perts_all = sorted({os.path.basename(p).replace("_stats.xlsx", "")
                        for m in models for p in glob.glob(os.path.join(ds_dir, m, "embeddings", "*_stats.xlsx"))})
    if not perts_all:
        print(f"[WARN] No perturbations found for '{dataset}'. Skipping.")
        return
    perts_used = ([p for p in perturb_order if p in perts_all] if perturb_order else perts_all)[:9]

    curves = {}
    for model in models:
        for perturb in perts_used:
            stat_path = ds_dir / model / "embeddings" / f"{perturb}_stats.xlsx"
            if stat_path.is_file():
                try:
                    curves[(model, perturb)] = _load_normalized_curve(stat_path, perturb)
                except Exception as e:
                    print(f"[SKIP] {dataset}/{model}/{perturb}: {e}")

    # Shared y-range across panels
    ys = [y for _, y in curves.values() if y.size]
    if ys:
        y_min = max(0.0, 0.9 * float(min(np.nanmin(y) for y in ys)))
        y_max = 1.05 * float(max(np.nanmax(y) for y in ys))
        if y_max <= y_min:
            y_max = y_min + 0.1
    else:
        y_min, y_max = 0.0, 1.1

    fig, axes = plt.subplots(3, 3, figsize=(14, 10), constrained_layout=True)
    axes = axes.flatten()
    legend_handles = {}
    for i, perturb in enumerate(perts_used):
        ax = axes[i]
        plotted_any = False
        for model in models:
            if (model, perturb) not in curves:
                continue
            x, y_norm = curves[(model, perturb)]
            if x.size == 0:
                continue
            line, = ax.plot(x, y_norm, marker="o", markersize=3.5, linewidth=1.2,
                            color=model_colour(model), label=model)
            legend_handles[model] = line
            plotted_any = True
        ax.set_title(perturb)
        ax.set_xlabel("strength")
        ax.set_ylabel("normalized similarity")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(y_min, y_max)
        ax.margins(x=0.01, y=0.02)
        if perturb in INVERT_AXIS:
            ax.invert_xaxis()
        if not plotted_any:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
    for j in range(len(perts_used), 9):
        axes[j].axis("off")

    labels = [m for m in models if m in legend_handles]
    ncol = 1 if len(labels) <= 12 else (2 if len(labels) <= 24 else 3)
    fig.legend([legend_handles[m] for m in labels], labels, title="Model",
               loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=18,
               title_fontsize=18, ncol=ncol, handlelength=1.5, handletextpad=0.6)

    out_dir = stats_dir(dataset, "similarity_aggregate")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f"{dataset}_similarity_grid.png"
    fig.savefig(out_png, dpi=220, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f"[OK] Saved grid → {out_png}")
