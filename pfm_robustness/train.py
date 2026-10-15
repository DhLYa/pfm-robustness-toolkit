"""Train the 2-class head of each model on a dataset (frozen backbone; ResNets fine-tuned).

Images are fed as 0-1 tensors (resized to 224) and the model's normalisation is applied per
batch, i.e. exactly the preprocessing used at evaluation time. Checkpoints are written to
``Models/<dataset>/<weight_key>_<dataset>.pth``.
"""

import copy
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torchvision import transforms
from tqdm import tqdm

from .datasets import DATASETS, count_class_distribution, create_trainval_dict
from .models import get_transform, load_for_training, save_trained
from .paths import default_device


def batch_transform(transform):
    """Apply `transform` to a (B,C,H,W) batch, falling back to per-image PIL (as the Evaluator does)."""
    to_pil = transforms.ToPILImage()

    def fn(batch):
        try:
            out = transform(batch)
            return out if out.ndim == 4 else out.unsqueeze(0)
        except Exception:
            return torch.stack([transform(to_pil(img)) for img in batch])
    return fn


def train_loop(model, dataloaders, criterion, optimizer, epochs, device="cuda:0", transform_func=None, patience=2):
    """Train/val loop with early stopping on validation ROC-AUC; returns (best model, history)."""
    acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_rocauc = 0.0
    epochs_no_improve = 0

    for epoch in range(epochs):
        print(f'\nEpoch {epoch + 1}/{epochs}\n' + '-' * 30)
        epoch_start = time.time()

        for stage, loader in dataloaders.items():
            model.train() if stage == "train" else model.eval()

            running_loss, running_corrects, num_examples = 0.0, 0, 0
            all_labels, all_preds = [], []

            for inputs, labels, _ in tqdm(loader, desc=f"{stage}", leave=False):
                if transform_func:
                    inputs = transform_func(inputs)
                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.set_grad_enabled(stage == "train"):
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    probs = torch.softmax(outputs, dim=1)[:, 1]
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if stage == "train":
                        loss.backward()
                        optimizer.step()

                bs = labels.size(0)
                running_loss += loss.item() * bs
                running_corrects += torch.sum(preds == labels).item()
                num_examples += bs
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(probs.detach().cpu().numpy())

            epoch_loss = running_loss / num_examples
            epoch_acc = running_corrects / num_examples
            try:
                rocauc = roc_auc_score(all_labels, all_preds)
            except ValueError:
                rocauc = 0.0  # only one class present
            print(f'{stage} — Loss: {epoch_loss:.3f}  Acc: {epoch_acc:.3f}  ROC AUC: {rocauc:.3f}')

            if stage == "val":
                if rocauc > best_rocauc:
                    best_rocauc = rocauc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

        acc_history.append(best_rocauc)
        print(f"Epoch time: {round(time.time() - epoch_start, 2)}s")

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered after {patience} epochs with no ROC AUC improvement.\n")
            break

    print(f"\nTraining complete. Best val ROC AUC: {best_rocauc:.3f}")
    model.load_state_dict(best_model_wts)
    return model, np.array(acc_history)


def train_model(dataset: str, model_name: str, epochs: int = 10, patience: int = 2, device=None,
                show_class_counts: bool = False):
    """Train `model_name`'s head on `dataset` and save the checkpoint; returns its path."""
    device = device or default_device()
    spec = DATASETS[dataset]
    b = spec.budgets["train"]

    print(f"Loading {model_name} model for training...")
    model, optimizer = load_for_training(model_name, device)

    loaders = create_trainval_dict(
        spec.cls, spec.train_path(), b["batch_size"],
        transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]),
        b["trainval_multiplier"], b["trainval_size"], **spec.train_kwargs,
    )
    if show_class_counts:
        print(f"Training: {count_class_distribution(loaders['train'])}")
        print(f"Validation: {count_class_distribution(loaders['val'])}")

    model, _ = train_loop(model, loaders, nn.CrossEntropyLoss(), optimizer, epochs=epochs, device=device,
                          transform_func=batch_transform(get_transform(model_name, model)), patience=patience)
    path = save_trained(model_name, dataset, model)
    print(f"Saved {path}")
    return path
