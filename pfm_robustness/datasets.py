"""Patch datasets, stratified subsampling helpers, and the DATASETS registry.

Every dataset returns ``(image, label, name)`` and exposes per-item labels (``.labels`` or
``.samples[i]["label"]``) so subsets can be stratified.

To add a dataset:
    1. write a ``torch.utils.data.Dataset`` following that contract (binary labels, 0/1);
    2. add a ``DatasetSpec`` to ``DATASETS`` with its folders, constructor kwargs and the
       subsample fractions (``budgets``) each pipeline stage should use.
"""

import os
from dataclasses import dataclass, field

import h5py
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

from .paths import data_dir


# ─── dataset classes ─────────────────────────────────────────────────────────

class NCTDataSet(Dataset):
    CLASSES = ['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM']

    def __init__(self, root_dir, classification_mode='multiclass',
                 selected_classes=None, transform=None):
        self.paths, self.labels = [], []
        self.transform = transform
        for cls in self.CLASSES:
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_dir):
                continue

            if classification_mode == 'multiclass':
                label = self.CLASSES.index(cls)
            elif classification_mode == 'tum_vs_all':
                label = 1 if cls == 'TUM' else 0
            elif classification_mode == 'tum_vs_norm':
                if cls not in ('TUM', 'NORM'):
                    continue
                label = 1 if cls == 'TUM' else 0
            elif classification_mode == 'tum_vs_selected':
                if selected_classes is None:
                    raise ValueError("must supply selected_classes for 'tum_vs_selected'")
                if cls not in (['TUM'] + selected_classes):
                    continue
                label = 1 if cls == 'TUM' else 0
            else:
                raise ValueError(f"Unknown mode {classification_mode!r}")

            for fname in os.listdir(cls_dir):
                if fname.lower().endswith('.tif'):
                    self.paths.append(os.path.join(cls_dir, fname))
                    self.labels.append(label)

        if not self.paths:
            raise RuntimeError(f"No images found in {root_dir!r} for mode {classification_mode}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx], os.path.basename(path)


class PanNukeDataset(Dataset):
    """Tumour (>= min_positive neoplastic nuclei) vs none; 1..min_positive-1 are dropped."""

    def __init__(self, root_dir, folds=(1, 2, 3), transform=None, min_positive=5):
        self.samples = []
        self.transform = transform
        for fold in folds:
            img_path = os.path.join(root_dir, f'Fold {fold}/images/fold{fold}/images.npy')
            mask_path = os.path.join(root_dir, f'Fold {fold}/masks/fold{fold}/masks.npy')
            images = np.load(img_path)
            masks = np.load(mask_path)
            for idx, (img, msk) in enumerate(zip(images, masks)):
                count = len(np.unique(msk[..., 0][msk[..., 0] != 0]))  # neoplastic nuclei instances
                if count >= min_positive:
                    label = 1
                elif count == 0:
                    label = 0
                else:
                    continue
                self.samples.append({
                    'img': img,
                    'label': label,
                    'name': f'Fold{fold}_{idx:06d}.npy'
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.fromarray(sample['img'].astype(np.uint8)).resize((224, 224))
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        return img, sample['label'], sample['name']


class PandaDataset(Dataset):
    """PANDA patches; with zero_vs_all, Gleason 0 -> 0 and Gleason 3/4/5 -> 1."""

    def __init__(self, root_dir, split='train', transform=None, stratify=False,
                 zero_vs_all=True, max_samples_per_class=None, random_state=42):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.samples = []

        csv_path = os.path.join(root_dir, f"{split}.csv")
        df = pd.read_csv(csv_path)
        df['class'] = df['class'].astype(int)

        if stratify:
            grouped = df.groupby('class')
            min_count = grouped.size().min() if max_samples_per_class is None else max_samples_per_class
            sampled = []
            for _, group in grouped:
                if len(group) > min_count:
                    group = group.sample(n=min_count, random_state=random_state)
                sampled.append(group)
            df = pd.concat(sampled, ignore_index=True)

        for _, row in df.iterrows():
            orig_label = int(row['class'])
            label = (0 if orig_label == 0 else 1) if zero_vs_all else orig_label
            self.samples.append({
                'img_path': os.path.join(root_dir, split, row['image']),
                'label': label,
                'name': row['image'],
            })

        if not self.samples:
            raise RuntimeError(f"No samples found in {csv_path}!")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample['img_path']).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, sample['label'], sample['name']


class PatchCamelyonDataset(Dataset):
    """PCam level-2 patches (96x96) from the official .h5 files and meta CSVs."""

    # create_trainval_dict uses these official splits instead of re-splitting one set
    official_trainval_splits = ('train', 'valid')

    def __init__(self, root_dir, split='train', transform=None):
        self.transform = transform
        images_h5 = os.path.join(root_dir, 'images', f'camelyonpatch_level_2_split_{split}_x.h5')
        meta_csv = os.path.join(root_dir, 'images', f'camelyonpatch_level_2_split_{split}_meta.csv')
        self._h5 = h5py.File(images_h5, 'r')
        self._images = self._h5[list(self._h5.keys())[0]]
        df = pd.read_csv(meta_csv)
        self.labels = df['tumor_patch'].astype(int).tolist()
        self.names = df.index.astype(str).tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.fromarray(self._images[idx].astype('uint8'))
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        return img, self.labels[idx], self.names[idx]


# ─── subsampling helpers ─────────────────────────────────────────────────────

def dataset_labels(ds, label_getter=None):
    """Per-item labels for stratified splitting."""
    indices = range(len(ds))
    if label_getter is not None:
        return [label_getter(ds, i) for i in indices]
    if hasattr(ds, "labels"):
        return [ds.labels[i] for i in indices]
    if hasattr(ds, "samples"):
        sample = ds.samples[0]
        if isinstance(sample, dict) and "label" in sample:
            return [ds.samples[i]['label'] for i in indices]
        if isinstance(sample, (list, tuple)) and len(sample) > 1:
            return [ds.samples[i][1] for i in indices]
        raise RuntimeError("Unknown sample structure in .samples.")
    raise RuntimeError("Unable to infer labels for stratified split.")


def stratified_subset(indices, labels, fraction, random_state=42):
    """Stratified `fraction` of `indices`; everything if fraction >= 1."""
    if fraction >= 1.0:
        return list(indices)
    subset, _ = train_test_split(indices, train_size=fraction, stratify=labels, random_state=random_state)
    return subset


def build_dataset(dataset_class, root_dir, batch_size, test_multiplier, transform=None,
                  stratify_label_getter=None, shuffle=False, **dataset_kwargs):
    """Stratified subset of one dataset split -> (Subset, DataLoader)."""
    ds = dataset_class(root_dir, transform=transform, **dataset_kwargs)
    indices = list(range(len(ds)))
    labels = dataset_labels(ds, stratify_label_getter)
    test_data = Subset(ds, stratified_subset(indices, labels, test_multiplier))
    return test_data, DataLoader(test_data, batch_size=batch_size, shuffle=shuffle)


def create_trainval_dict(dataset_class, root_dir, batch_size, transform, trainval_multiplier,
                         trainval_size, shuffle=True, label_getter=None, random_state=42,
                         **dataset_kwargs):
    """
    {'train', 'val'} loaders.

    Datasets with ``official_trainval_splits`` (PatchCamelyon) subsample each official split:
    `trainval_multiplier` of train, `trainval_size` of valid. Otherwise one split is
    subsampled to `trainval_multiplier` and then `trainval_size` of that is held out for val.
    """
    official = getattr(dataset_class, "official_trainval_splits", None)
    if official:
        train_split, val_split = official
        train_ds = dataset_class(root_dir, split=train_split, transform=transform, **dataset_kwargs)
        val_ds = dataset_class(root_dir, split=val_split, transform=transform, **dataset_kwargs)

        def _subsample(ds, multiplier):
            ix = list(range(len(ds)))
            if multiplier < 1.0:
                ix, _ = train_test_split(ix, train_size=multiplier, stratify=ds.labels,
                                         random_state=random_state)
            return ix

        loaders = {
            'train': DataLoader(Subset(train_ds, _subsample(train_ds, trainval_multiplier)),
                                batch_size=batch_size, shuffle=shuffle),
            'val': DataLoader(Subset(val_ds, _subsample(val_ds, trainval_size)),
                              batch_size=batch_size, shuffle=shuffle),
        }
    else:
        ds = dataset_class(root_dir, transform=transform, **dataset_kwargs)
        ix = list(range(len(ds)))
        labels = dataset_labels(ds, label_getter)

        trainval_ix, _ = train_test_split(
            ix, train_size=trainval_multiplier, stratify=labels, random_state=random_state
        )
        train_ix, val_ix = train_test_split(
            trainval_ix, test_size=trainval_size,
            stratify=[labels[i] for i in trainval_ix], random_state=random_state
        )
        loaders = {
            'train': DataLoader(Subset(ds, train_ix), batch_size=batch_size, shuffle=shuffle),
            'val': DataLoader(Subset(ds, val_ix), batch_size=batch_size, shuffle=shuffle),
        }

    print(f"Train batches: {len(loaders['train'])}")
    print(f" Val batches: {len(loaders['val'])}")
    return loaders


def count_class_distribution(dataloader):
    """Label counts over a loader, e.g. tensor([n_neg, n_pos])."""
    import torch

    counts = None
    for _, labels, _ in dataloader:
        bincount = torch.bincount(labels, minlength=labels.max().item() + 1)
        if counts is None:
            counts = bincount
        else:
            if bincount.numel() > counts.numel():
                counts = torch.cat([counts, torch.zeros(bincount.numel() - counts.numel(), dtype=counts.dtype)])
            counts[:bincount.numel()] += bincount
    return counts


# ─── registry ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DatasetSpec:
    """
    budgets: per-stage subsampling, as used for the dissertation runs.
        train / embed / kfold : trainval_multiplier, trainval_size (and test_multiplier for
                                embed / kfold), batch_size
        eval / sweep          : test_multiplier, batch_size
    """
    name: str
    cls: type
    train_root: tuple          # path parts under Data/
    test_root: tuple
    train_kwargs: dict
    test_kwargs: dict
    budgets: dict
    resize_to_224: bool = False  # native patch size differs from 224 (PCam is 96x96)
    description: str = field(default="", compare=False)

    def train_path(self):
        return str(data_dir(*self.train_root))

    def test_path(self):
        return str(data_dir(*self.test_root))

    def tensor_transform(self):
        """Raw image -> 0-1 tensor (resized to 224 where needed); no model normalisation."""
        steps = [transforms.Resize((224, 224))] if self.resize_to_224 else []
        return transforms.Compose(steps + [transforms.ToTensor()])


DATASETS = {spec.name: spec for spec in [
    DatasetSpec(
        name="NCT", cls=NCTDataSet,
        train_root=("NCT", "NCT-CRC-HE-100K"), test_root=("NCT", "CRC-VAL-HE-7K"),
        train_kwargs=dict(classification_mode='tum_vs_all'),
        test_kwargs=dict(classification_mode='tum_vs_all'),
        budgets={
            "train": dict(trainval_multiplier=0.05, trainval_size=0.3, batch_size=16),
            "eval": dict(test_multiplier=0.1, batch_size=8),
            "sweep": dict(test_multiplier=0.012, batch_size=8),
            "embed": dict(trainval_multiplier=0.03666, trainval_size=0.002, test_multiplier=0.075, batch_size=16),
            "kfold": dict(trainval_multiplier=0.1, trainval_size=0.3, test_multiplier=0.5, batch_size=16),
        },
        description="Colorectal tissue, TUM vs all other 8 classes (1:8).",
    ),
    DatasetSpec(
        name="PanNuke", cls=PanNukeDataset,
        train_root=("PanNuke",), test_root=("PanNuke",),
        train_kwargs=dict(folds=(1, 2), min_positive=5),
        test_kwargs=dict(folds=(3,), min_positive=5),
        budgets={
            "train": dict(trainval_multiplier=0.999, trainval_size=0.3, batch_size=16),
            "eval": dict(test_multiplier=0.285, batch_size=8),
            "sweep": dict(test_multiplier=0.035, batch_size=8),
            "embed": dict(trainval_multiplier=0.465, trainval_size=0.002, test_multiplier=0.1, batch_size=16),
            "kfold": dict(trainval_multiplier=0.999, trainval_size=0.3, test_multiplier=0.999, batch_size=16),
        },
        description="Pan-cancer; >=5 neoplastic nuclei vs none. Folds 1+2 train, fold 3 test.",
    ),
    DatasetSpec(
        name="PANDA", cls=PandaDataset,
        train_root=("PANDA",), test_root=("PANDA",),
        train_kwargs=dict(split='train'),
        test_kwargs=dict(split='test'),
        budgets={
            "train": dict(trainval_multiplier=0.077, trainval_size=0.3, batch_size=16),
            "eval": dict(test_multiplier=0.097, batch_size=8),
            "sweep": dict(test_multiplier=0.012, batch_size=8),
            "embed": dict(trainval_multiplier=0.0565, trainval_size=0.002, test_multiplier=0.069, batch_size=16),
            "kfold": dict(trainval_multiplier=0.154, trainval_size=0.3, test_multiplier=0.6, batch_size=16),
        },
        description="Prostate biopsies; Gleason 0 vs Gleason 3/4/5.",
    ),
    DatasetSpec(
        name="PatchCamelyon", cls=PatchCamelyonDataset,
        train_root=("PatchCamelyon",), test_root=("PatchCamelyon",),
        train_kwargs={},
        test_kwargs=dict(split='test'),
        budgets={
            "train": dict(trainval_multiplier=0.0134, trainval_size=0.046, batch_size=16),
            "eval": dict(test_multiplier=0.022, batch_size=8),
            "sweep": dict(test_multiplier=0.00264, batch_size=8),
            "embed": dict(trainval_multiplier=0.0128, trainval_size=0.001, test_multiplier=0.01558, batch_size=16),
            "kfold": dict(trainval_multiplier=0.0268, trainval_size=0.046, test_multiplier=0.135, batch_size=16),
        },
        resize_to_224=True,
        description="Lymph-node metastasis; official train/valid/test splits.",
    ),
]}
