# Derived from the original reetoolbox project (Apache License 2.0).
# Modified by Dhyey Yajnik, 2025.

import os
import random
import uuid

import matplotlib.pyplot as plt
import torch
from torchvision import transforms

from .optimisers import untargeted_loss

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
        post_transform=None,
        max_save_images: int = 30,
    ):
        """
        Parameters
        ----------
        post_transform : callable or None
            Model-specific preprocessing (normalisation) applied *after* the attack, so the
            perturbation is optimised in raw pixel space.
        max_save_images : int
            Cap on number of misclassified samples written to `save_dir` per call.
        """
        self.model = model.to(device).eval()
        self.model_name = model_name
        self.dataset = dataset
        self.dataloader = dataloader
        self.device = device
        self.criterion = criterion
        self.trans_params = trans_params
        self.save_dir = save_dir
        self.verbose = verbose
        self.post_transform = post_transform
        self.max_save_images = max_save_images

        self.attack = TransformOptimiser(
            self.model,
            Transform,
            optimiser_params,
            trans_params,
            criterion=criterion,
            device=device,
        )

        # For processor-based transforms that need PIL input
        self._tensor_to_pil = transforms.ToPILImage()

    def _apply_post_transform(self, imgs: torch.Tensor) -> torch.Tensor:
        if self.post_transform is None:
            return imgs
        try:
            out = self.post_transform(imgs)
            return out if out.ndim == 4 else out.unsqueeze(0)
        except Exception:
            # Fall back to per-image PIL input
            if imgs.dim() == 4:
                pil_seq = [self._tensor_to_pil(img.cpu()) for img in imgs]
                proc = [self.post_transform(p) for p in pil_seq]
                return torch.stack(proc)
            pil = self._tensor_to_pil(imgs.cpu())
            return self.post_transform(pil)

    def predict(self, adversarial: bool, log_pct: float = 0.50):
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
        interval = max(1, int(total_batches * log_pct))

        for batch_idx, (inputs_raw, labels, names) in enumerate(self.dataloader):
            labels = labels.to(self.device)
            inputs_raw = inputs_raw.to(self.device)

            if adversarial:
                orig_raw, adv_raw = self.attack.optimise(
                    inputs_raw, targets=labels, reset_weights=True
                )

                # Model preprocessing happens *after* the attack
                orig_t = self._apply_post_transform(orig_raw).to(self.device)
                adv_t = self._apply_post_transform(adv_raw).to(self.device)

                logits_adv = self.model(adv_t)
                preds_adv = logits_adv.argmax(dim=1)
                mask = preds_adv != labels

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

        res = {
            "outputs": torch.stack(outputs),
            "labels": torch.stack(labels_all),
        }
        if adversarial:
            res["adversarial_outputs"] = torch.stack(adv_outputs)
        if mis_orig:
            res.update({
                "mis_originals": torch.cat(mis_orig),
                "mis_adversarials": torch.cat(mis_adv),
                "mis_perturbations": torch.cat(mis_pert),
                "mis_labels": torch.cat(mis_labels),
                "mis_preds": torch.cat(mis_preds),
                "mis_names": mis_names,
            })
            if self.save_dir:
                n, folder = self._save_misclassified(res)
                if self.verbose:
                    print(f"    saved {n} failures to {folder}")

        return res

    def _save_misclassified(self, res: dict):
        """Save up to `max_save_images` misclassified pairs (original vs perturbed)."""
        origs = res["mis_originals"]
        advs = res["mis_adversarials"]
        labs = res["mis_labels"]
        preds = res["mis_preds"]
        names = res["mis_names"]

        os.makedirs(self.save_dir, exist_ok=True)
        total = origs.size(0)
        chosen = random.sample(range(total), min(total, self.max_save_images))

        for idx in chosen:
            o = origs[idx]
            a = advs[idx]
            lbl = labs[idx].item()
            prd = preds[idx].item()
            fname = names[idx]

            # Labels assume a binary tumour / non-tumour task
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

            uid = uuid.uuid4().hex[:8]
            base = os.path.splitext(os.path.basename(fname))[0]
            out = f"{self.model_name.lower()}_{uid}_{base}.png"
            fig.savefig(os.path.join(self.save_dir, out), bbox_inches="tight")
            plt.close(fig)

        return len(chosen), self.save_dir

    def compute_metric(self, metric, **params):
        return metric(self.predict(**params))

    def attack_inputs(self, indices, target_classes=None, criterion=None):
        """Attack the dataset items at `indices`; returns (originals, perturbed)."""
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
