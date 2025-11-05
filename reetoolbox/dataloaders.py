from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import os
import numpy as np
import torch
from torchvision import transforms

from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

def build_dataset(
    dataset_class,
    root_dir,
    batch_size,
    test_multiplier,
    transform=None,
    stratify_label_getter=None,
    shuffle=False,
    **dataset_kwargs
):
    """
    dataset_class: e.g., NCTDataSet, PanNukeDataset, or MIDOGpp (pass the class, not an instance)
    root_dir: path to the root data dir
    batch_size: DataLoader batch size
    test_multiplier: fraction (e.g., 0.05 for 5%) or >=1 for all
    transform: torchvision transforms to apply to each image
    stratify_label_getter: function to get the stratification labels from the dataset (if needed)
    shuffle: if True, DataLoader shuffles batches
    **dataset_kwargs: additional arguments for the dataset class
    """
    ds = dataset_class(root_dir, transform=transform, **dataset_kwargs)
    indices = list(range(len(ds)))

    # Get labels for stratified sampling
    if stratify_label_getter is not None:
        labels = [stratify_label_getter(ds, i) for i in indices]
    elif hasattr(ds, "labels"):
        labels = [ds.labels[i] for i in indices]
    elif hasattr(ds, "samples"):
        sample = ds.samples[0]
        if isinstance(sample, dict) and "label" in sample:
            labels = [ds.samples[i]['label'] for i in indices]
        elif isinstance(sample, (list, tuple)) and len(sample) > 1:
            labels = [ds.samples[i][1] for i in indices]
        else:
            raise RuntimeError("Unknown sample structure in .samples.")
    else:
        raise RuntimeError("Unable to infer labels for stratified split.")

    # Stratified subsample
    if test_multiplier >= 1.0:
        test_ix = indices
    else:
        test_ix, _ = train_test_split(
            indices,
            train_size=test_multiplier,
            stratify=labels,
            random_state=42,
        )

    test_data = Subset(ds, test_ix)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=shuffle)
    return test_data, test_loader

def create_trainval_dict(
    dataset_class,
    root_dir,
    batch_size,
    transform,
    trainval_multiplier,
    trainval_size,
    shuffle=True,
    label_getter=None,
    random_state=42,
    **dataset_kwargs
):
    # --- special case for PatchCamelyon ---
    if getattr(dataset_class, "__name__", "") == "PatchCamelyonDataset":
        # load full train & valid sets
        train_ds = dataset_class(root_dir, split='train', transform=transform, **dataset_kwargs)
        val_ds   = dataset_class(root_dir, split='valid', transform=transform, **dataset_kwargs)

        # helper to subsample one split
        def _subsample(ds, multiplier, rs):
            ix = list(range(len(ds)))
            labels = ds.labels
            if multiplier < 1.0:
                ix, _ = train_test_split(
                    ix,
                    train_size=multiplier,
                    stratify=labels,
                    random_state=rs
                )
            return ix

        train_ix = _subsample(train_ds, trainval_multiplier, random_state)
        val_ix   = _subsample(val_ds,   trainval_size,      random_state)

        train_loader = DataLoader(
            Subset(train_ds, train_ix),
            batch_size=batch_size, shuffle=shuffle
        )
        val_loader   = DataLoader(
            Subset(val_ds,   val_ix),
            batch_size=batch_size, shuffle=shuffle
        )

        print(f"Train batches: {len(train_loader)}")
        print(f" Val batches: {len(val_loader)}")
        return {'train': train_loader, 'val': val_loader}

    # --- fallback to original logic for NCT, PanNuke, PANDA, etc. ---
    ds = dataset_class(root_dir, transform=transform, **dataset_kwargs)
    ix = list(range(len(ds)))

    # Universal label getter logic
    if label_getter is not None:
        labels = [label_getter(ds, i) for i in ix]
    elif hasattr(ds, 'labels'):
        labels = [ds.labels[i] for i in ix]
    elif hasattr(ds, 'samples'):
        sample0 = ds.samples[0]
        if isinstance(sample0, dict) and "label" in sample0:
            labels = [ds.samples[i]['label'] for i in ix]
        elif isinstance(sample0, (list, tuple)) and len(sample0) > 1:
            labels = [ds.samples[i][1] for i in ix]
        else:
            raise RuntimeError("Unknown sample structure in .samples.")
    else:
        raise RuntimeError("Unable to infer labels for stratified split.")

    # First: train+val split
    trainval_ix, _ = train_test_split(
        ix, train_size=trainval_multiplier, stratify=labels, random_state=random_state
    )
    # Then: train/val split
    train_ix, val_ix = train_test_split(
        trainval_ix,
        test_size=trainval_size,
        stratify=[labels[i] for i in trainval_ix],
        random_state=random_state
    )

    train_data = Subset(ds, train_ix)
    val_data   = Subset(ds, val_ix)

    dataloaders_dict = {
        'train': DataLoader(train_data, batch_size=batch_size, shuffle=shuffle),
        'val':   DataLoader(val_data,   batch_size=batch_size, shuffle=shuffle),
    }

    print(f"Train batches: {len(dataloaders_dict['train'])}")
    print(f" Val batches: {len(dataloaders_dict['val'])}")
    return dataloaders_dict

class NCTDataSet(Dataset):
    CLASSES = ['ADI','BACK','DEB','LYM','MUC','MUS','NORM','STR','TUM']
    def __init__(self, root_dir, classification_mode='multiclass',
                 selected_classes=None, transform=None):
        self.paths, self.labels = [], []
        self.transform = transform
        for cls in self.CLASSES:
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_dir): continue

            if classification_mode == 'multiclass':
                label = self.CLASSES.index(cls)
            elif classification_mode == 'tum_vs_all':
                label = 1 if cls=='TUM' else 0
            elif classification_mode == 'tum_vs_norm':
                if cls not in ('TUM','NORM'): continue
                label = 1 if cls=='TUM' else 0
            elif classification_mode == 'tum_vs_selected':
                if selected_classes is None:
                    raise ValueError("must supply selected_classes for 'tum_vs_selected'")
                if cls not in (['TUM'] + selected_classes): continue
                label = 1 if cls=='TUM' else 0
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
        img  = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        name  = os.path.basename(path)
        label = self.labels[idx]
        return img, label, name

class PanNukeDataset(Dataset):
    def __init__(self, root_dir, folds=(1,2,3), transform=None, min_positive=5):
        """
        root_dir: top directory with Fold 1/, Fold 2/, etc
        folds: tuple/list of fold numbers to include (e.g., (1,2) for trainval, (3,) for test)
        transform: torchvision transforms
        min_positive: threshold for 'positive' (tumorous) label
        """
        self.samples = []
        self.transform = transform
        for fold in folds:
            img_path = os.path.join(root_dir, f'Fold {fold}/images/fold{fold}/images.npy')
            mask_path = os.path.join(root_dir, f'Fold {fold}/masks/fold{fold}/masks.npy')
            images = np.load(img_path)
            masks = np.load(mask_path)
            for idx, (img, msk) in enumerate(zip(images, masks)):
                count = len(np.unique(msk[..., 0][msk[..., 0] != 0]))  # count neoplastic nuclei instances
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
        img = sample['img']
        label = sample['label']
        name = sample['name']

        img = Image.fromarray(img.astype(np.uint8)).resize((224, 224))
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        return img, label, name
    
import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class PandaDataset(Dataset):
    """
    PANDA dataset with optional per-class stratification and binary "0 vs all" mode.
    """
    def __init__(
        self,
        root_dir,
        split='train',
        transform=None,
        stratify=False,
        zero_vs_all=True,
        max_samples_per_class=None,
        random_state=42
    ):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.stratify = stratify
        self.zero_vs_all = zero_vs_all
        self.samples = []

        csv_path = os.path.join(root_dir, f"{split}.csv")
        df = pd.read_csv(csv_path)
        df['class'] = df['class'].astype(int)
        rng = np.random.default_rng(random_state)

        if stratify:
            grouped = df.groupby('class')
            if max_samples_per_class is None:
                min_count = grouped.size().min()
            else:
                min_count = max_samples_per_class
            sampled = []
            for cls, group in grouped:
                if len(group) > min_count:
                    group = group.sample(n=min_count, random_state=random_state)
                sampled.append(group)
            df = pd.concat(sampled, ignore_index=True)

        for _, row in df.iterrows():
            img_name = row['image']
            orig_label = int(row['class'])
            img_path = os.path.join(root_dir, split, img_name)
            if zero_vs_all:
                label = 0 if orig_label == 0 else 1
            else:
                label = orig_label
            self.samples.append({
                'img_path': img_path,
                'label': label,
                'name': img_name
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
        name = sample['name']
        label = sample['label']
        return img, label, name

import os
import pandas as pd
import h5py
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

class PatchCamelyonDataset(Dataset):
    """
    Reads the Camelyon16 level-2 patches out of .h5 plus their meta CSV.
    Exposes .labels so downstream splitting logic can stratify.
    """
    def __init__(self, root_dir, split='train', transform=None):
        self.transform = transform
        # paths
        images_h5 = os.path.join(
            root_dir, 'images',
            f'camelyonpatch_level_2_split_{split}_x.h5'
        )
        meta_csv = os.path.join(
            root_dir, 'images',
            f'camelyonpatch_level_2_split_{split}_meta.csv'
        )
        # open HDF5
        self._h5 = h5py.File(images_h5, 'r')
        # assume the first dataset in the file is the image array
        key = list(self._h5.keys())[0]
        self._images = self._h5[key]
        # load metadata
        df = pd.read_csv(meta_csv)
        # must have a column "tumor_patch" for stratification
        self.labels = df['tumor_patch'].astype(int).tolist()
        # name each sample by its original row index
        self.names  = df.index.astype(str).tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        arr = self._images[idx]              # numpy array H×W×C
        img = Image.fromarray(arr.astype('uint8'))
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        return img, self.labels[idx], self.names[idx]