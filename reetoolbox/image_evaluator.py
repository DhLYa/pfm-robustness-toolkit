# This file has been modified from the original reetoolbox project.
# Additional changes by Dhyey Yajnik, 2025.

import os
import time
import uuid
import random
import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

from reetoolbox.optimisers import targeted_loss, untargeted_loss

__all__ = ["Evaluator"]

class Evaluator:
    def __init__(
        self,
        model: torch.nn.Module,
        model_name: str,
        dataset,
        dataloader,
        TransformOptimiser,
        Transform,
        optimiser_params: dict,
        trans_params: dict,
        criterion=untargeted_loss,
        device: str = "cuda:0",
        save_dir: str | None = None,
        verbose: bool = True,
        post_transform=None,         # ← NEW
        max_save_images: int = 30,   # ← NEW
    ):
        """
        Parameters
        ----------
        post_transform : callable or None
            Model-specific preprocessing executed *after* adversarial attack.
        max_save_images : int
            Cap on number of misclassified samples written to disk.
        """
        self.model        = model.to(device).eval()
        self.model_name   = model_name
        self.dataset      = dataset
        self.dataloader   = dataloader
        self.device       = device
        self.criterion    = criterion
        self.trans_params = trans_params
        self.save_dir     = save_dir
        self.verbose      = verbose
        self.post_transform = post_transform
        self.max_save_images = max_save_images

        # Instantiate optimiser
        self.attack = TransformOptimiser(
            self.model,
            Transform,
            optimiser_params,
            trans_params,
            criterion=criterion,
            device=device,
        )

        # Helper for Processor-based transforms that need PIL
        self._tensor_to_pil = transforms.ToPILImage()

    # ──────────────────────────────────────────────────────────────────────
    #  INTERNAL HELPERS

    # def _apply_post_transform(self, imgs: torch.Tensor) -> torch.Tensor:
    #     """Apply self.post_transform on Tensor batch or single image."""
    #     if self.post_transform is None:
    #         return imgs
    #     try:
    #         # torchvision transforms work directly on tensors
    #         return self.post_transform(imgs)
    def _apply_post_transform(self, imgs: torch.Tensor) -> torch.Tensor:
        if self.post_transform is None:
            return imgs
        try:
            out = self.post_transform(imgs)
            return out if out.ndim == 4 else out.unsqueeze(0)
        except Exception:
            # ProcessorTransform expects PIL images
            if imgs.dim() == 4:  # B,C,H,W
                pil_seq = [self._tensor_to_pil(img.cpu()) for img in imgs]
                proc    = [self.post_transform(p) for p in pil_seq]
                return torch.stack(proc)
            # single image C,H,W
            pil = self._tensor_to_pil(imgs.cpu())
            return self.post_transform(pil)

    # ──────────────────────────────────────────────────────────────────────
    #  MAIN PREDICTION ENTRY POINT
    # ──────────────────────────────────────────────────────────────────────
    def predict(
        self,
        adversarial: bool,
        perturbation_measure=None,
        weight_measure=None,
        log_pct: float = 0.50,
    ):
        """
        Returns
        -------
        dict  with keys:
            outputs, labels, (adversarial_outputs),
            mis_originals, mis_adversarials, mis_perturbations,
            mis_labels, mis_preds, mis_names
        """
        outputs, adv_outputs, labels_all = [], [], []
        mis_orig, mis_adv, mis_pert = [], [], []
        mis_labels, mis_preds, mis_names = [], [], []

        total_batches = len(self.dataloader)
        interval      = max(1, int(total_batches * log_pct))
        attack_total  = 0.0

        for batch_idx, (inputs_raw, labels, names) in enumerate(self.dataloader):
            labels = labels.to(self.device)
            inputs_raw = inputs_raw.to(self.device)

            # ── Generate adversarial examples if requested ────────────────
            if adversarial:
                t0 = time.time()
                orig_raw, adv_raw = self.attack.optimise(
                    inputs_raw, targets=labels, reset_weights=True
                )
                attack_total += time.time() - t0

                # Apply post-processing *after* attack
                orig_t = self._apply_post_transform(orig_raw).to(self.device)
                adv_t  = self._apply_post_transform(adv_raw).to(self.device)

                logits_adv = self.model(adv_t)
                preds_adv  = logits_adv.argmax(dim=1)
                mask       = preds_adv != labels

                if mask.any():
                    mis_orig.append(orig_raw[mask].cpu())
                    mis_adv.append(adv_raw[mask].cpu())
                    mis_pert.append((adv_raw - orig_raw)[mask].cpu())
                    mis_labels.append(labels[mask].cpu())
                    mis_preds.append(preds_adv[mask].cpu())
                    mis_names.extend([
                        names[i] for i, m in enumerate(mask.cpu()) if m
                    ])

                adv_outputs.extend(logits_adv.detach().cpu())

                # Clean prediction for baseline metrics
                logits_clean = self.model(orig_t)
            else:
                orig_t = self._apply_post_transform(inputs_raw).to(self.device)
                logits_clean = self.model(orig_t)

            outputs.extend(logits_clean.detach().cpu())
            labels_all.extend(labels.detach().cpu())

            if self.verbose and ((batch_idx + 1) % interval == 0 or
                                 batch_idx == total_batches - 1):
                pct = int((batch_idx + 1) / total_batches * 100)
                print(f"    batch {batch_idx+1}/{total_batches}  ({pct} %)")

        # ── Package results ───────────────────────────────────────────────
        res = {
            "outputs": torch.stack(outputs),
            "labels":  torch.stack(labels_all),
        }
        if adversarial:
            res["adversarial_outputs"] = torch.stack(adv_outputs)
        if mis_orig:
            res.update({
                "mis_originals":     torch.cat(mis_orig),
                "mis_adversarials":  torch.cat(mis_adv),
                "mis_perturbations": torch.cat(mis_pert),
                "mis_labels":        torch.cat(mis_labels),
                "mis_preds":         torch.cat(mis_preds),
                "mis_names":         mis_names,
            })
            if self.save_dir:
                n, folder = self._save_misclassified(res)
                if self.verbose:
                    print(f"    ➜  saved {n} failures to  {folder}")

        return res

    # ──────────────────────────────────────────────────────────────────────
    #  SAVE A RANDOM SUBSET OF FAILURES
    # ──────────────────────────────────────────────────────────────────────
    def _save_misclassified(self, res: dict):
        """Save ≤ `max_save_images` misclassified pairs (orig vs adv)."""
        origs = res["mis_originals"]
        advs  = res["mis_adversarials"]
        labs  = res["mis_labels"]
        preds = res["mis_preds"]
        names = res["mis_names"]

        os.makedirs(self.save_dir, exist_ok=True)
        total = origs.size(0)
        chosen = random.sample(range(total),
                               min(total, self.max_save_images))

        for idx in chosen:
            o = origs[idx]
            a = advs[idx]
            lbl = labs[idx].item()
            prd = preds[idx].item()
            fname = names[idx]

            true_str = "TUMOUR" if lbl == 1 else "NON-TUM"
            pred_str = "TUMOUR" if prd == 1 else "NON-TUM"

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))
            fig.suptitle(os.path.basename(self.save_dir).upper(), fontsize=14)

            o_img = o.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            a_img = a.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()


            ax1.imshow(o_img)
            ax1.set_title(f"True : {true_str}")
            ax1.axis("off")

            ax2.imshow(a_img)
            ax2.set_title(f"Pred : {pred_str}")
            ax2.axis("off")

            uid  = uuid.uuid4().hex[:8]
            base = os.path.splitext(os.path.basename(fname))[0]
            out  = f"{self.model_name.lower()}_{uid}_{base}.png"
            fig.savefig(os.path.join(self.save_dir, out), bbox_inches="tight")
            plt.close(fig)

        return len(chosen), self.save_dir

    # ──────────────────────────────────────────────────────────────────────
    #  PUBLIC HELPERS (unchanged from your original version)
    # ──────────────────────────────────────────────────────────────────────
    def compute_metric(self, metric, **params):
        return metric(self.predict(**params))

    def metric_vs_strength(self, param, param_range, step_size, metric, **mparams):
        start, end = param_range
        values = np.arange(start, end + step_size, step_size)
        orig_hyp = copy.deepcopy(self.attack.hyperparameters)
        scores = []
        for i, v in enumerate(values):
            self.attack.hyperparameters[param] = v
            if i % max(1, len(values) // 4) == 0 and self.verbose:
                print(f"{int(i * 100 / len(values))}% complete…")
            scores.append(self.compute_metric(metric, **mparams))
        self.attack.hyperparameters = orig_hyp
        return values.tolist(), scores

    def attack_inputs(self, indices, target_classes=None, criterion=None):
        imgs, labs = [], []
        for i in indices:
            img, lbl, _ = self.dataset[i]
            imgs.append(img)
            labs.append(lbl)
        inputs = torch.stack(imgs).to(self.device)
        labels = torch.tensor(labs, device=self.device)
        if target_classes is not None and criterion is not None:
            self.attack.criterion = criterion
            orig, adv = self.attack.optimise(
                inputs, targets=target_classes, reset_weights=True
            )
            self.attack.criterion = self.criterion
        else:
            orig, adv = self.attack.optimise(
                inputs, targets=labels, reset_weights=True
            )
        return orig, adv

    def set_attack_hyperparameters(self, hyperparameters: dict):
        self.attack.hyperparameters = hyperparameters

    def attack_inputs_comparison(self, input_indices, target_classes=None, criterion=None):
        inputs, labels = [], []
        for i in input_indices:
            img, lbl, _ = self.dataset[i]
            inputs.append(img)
            labels.append(lbl)
        inputs = torch.stack(inputs).to(self.device)
        labels = torch.tensor(labels).to(self.device)

        if target_classes is not None and criterion is not None:
            self.attack.criterion = criterion
            inputs, adv_inputs = self.attack.optimise(
                inputs, targets=target_classes, reset_weights=True
            )
            self.attack.criterion = self.criterion
        else:
            inputs, adv_inputs = self.attack.optimise(
                inputs, targets=labels, reset_weights=True
            )
        return inputs, adv_inputs