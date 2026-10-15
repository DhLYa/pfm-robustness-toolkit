"""Model loaders, input normalisation, and the MODELS registry.

Every classifier is ``backbone -> Linear(num_features, 2) -> LogSoftmax``. Foundation-model
backbones are frozen, so only the head is trained and saved; ResNets are fine-tuned end to
end and saved as whole pickled models.

To add a model:
    1. write ``build_<model>(n_classes, device) -> (model, optimizer)``; the model must expose
       ``.head`` (usually via ``_attach_head_and_freeze``);
    2. add a ``ModelSpec`` to ``MODELS`` with its normalisation and parameter count.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import timm
import torch
import torch.nn as nn
import torch.optim as optim
from timm.data import resolve_data_config
from timm.layers import SwiGLUPacked
from torchvision import models, transforms
from transformers import AutoImageProcessor, AutoModel

from .paths import ensure_exaonepath, models_dir


# ─── shared building blocks ──────────────────────────────────────────────────

def _attach_head_and_freeze(model: nn.Module, n_classes: int, device: str):
    num_features = getattr(model, "num_features", None)
    if num_features is None:
        # fallback for timm classifiers named .head or .classifier
        if hasattr(model, "head") and hasattr(model.head, "in_features"):
            num_features = model.head.in_features
        elif hasattr(model, "classifier") and hasattr(model.classifier, "in_features"):
            num_features = model.classifier.in_features
        else:
            raise AttributeError("Cannot infer feature dimension")
    model.head = nn.Sequential(
        nn.Linear(num_features, n_classes),
        nn.LogSoftmax(dim=1)
    )
    for p in model.parameters():
        p.requires_grad = False
    for p in model.head.parameters():
        p.requires_grad = True
    model = model.to(device).train()
    optim_head = optim.SGD(model.head.parameters(), lr=1e-3, momentum=0.9)
    return model, optim_head


class HFCLSWrapper(nn.Module):
    """HuggingFace vision backbone -> CLS token features, with a replaceable `.head`."""

    def __init__(self, backbone: AutoModel):
        super().__init__()
        self.backbone = backbone
        self.num_features = getattr(
            backbone.config, "hidden_size",
            getattr(backbone.config, "vision_dim", None)
        )
        if self.num_features is None:
            raise ValueError("Cannot find hidden size in config")
        self.head = nn.Identity()

    def forward(self, pixel_values=None, **kwargs):
        outputs = self.backbone(pixel_values=pixel_values, **kwargs)
        last_hidden = (
            outputs.last_hidden_state
            if hasattr(outputs, "last_hidden_state")
            else outputs[0]
        )
        feats = last_hidden[:, 0, :]
        return self.head(feats)


class ClsMeanPatchWrapper(nn.Module):
    """timm token backbone -> concat(CLS token, mean of tokens[:, 5:])."""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.num_features = backbone.embed_dim * 2
        self.head = nn.Identity()

    def forward(self, x):
        tokens = self.backbone(x)
        cls_tok = tokens[:, 0]
        # Skips CLS + 4 register tokens. Correct for Virchow2 and H0-mini; Virchow v1 has
        # no register tokens (see README "Known issues").
        patch_tok = tokens[:, 5:].mean(1)
        feats = torch.cat([cls_tok, patch_tok], dim=-1)
        return self.head(feats)


# ─── builders: (n_classes, device) -> (model, optimizer) ─────────────────────

def load_resnet18(n_classes: int = 2, device: str = "cuda:0"):
    model = models.resnet18(pretrained=True)
    model.fc = nn.Sequential(nn.Linear(512, n_classes), nn.LogSoftmax(dim=1))
    model = model.to(device)
    model.train()
    return model, optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


def load_resnet50(n_classes: int = 2, device: str = "cuda:0"):
    model = models.resnet50(pretrained=True)
    model.fc = nn.Sequential(nn.Linear(2048, n_classes), nn.LogSoftmax(dim=1))
    model = model.to(device)
    model.train()
    return model, optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


def load_uni(n_classes: int = 2, device: str = "cuda:0"):
    """MahmoodLab/UNI (ViT-L/16)."""
    model = timm.create_model(
        "hf-hub:MahmoodLab/uni", pretrained=True, num_classes=0,
        init_values=1e-5, dynamic_img_size=True,
    )
    return _attach_head_and_freeze(model, n_classes, device)


def load_uni2(n_classes: int = 2, device: str = "cuda:0"):
    """MahmoodLab/UNI2-h (ViT-H/14)."""
    timm_kwargs = {
        'img_size': 224, 'patch_size': 14, 'depth': 24, 'num_heads': 24,
        'init_values': 1e-5, 'embed_dim': 1536, 'mlp_ratio': 2.66667 * 2,
        'num_classes': 0, 'no_embed_class': True, 'mlp_layer': SwiGLUPacked,
        'act_layer': nn.SiLU, 'reg_tokens': 8, 'dynamic_img_size': True
    }
    model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
    return _attach_head_and_freeze(model, n_classes, device)


def load_gigapath(n_classes: int = 2, device: str = "cuda:0"):
    model = timm.create_model("hf-hub:prov-gigapath/prov-gigapath", pretrained=True, num_classes=0)
    return _attach_head_and_freeze(model, n_classes, device)


def load_virchow(n_classes: int = 2, device: str = "cuda"):
    vit = timm.create_model(
        "hf-hub:paige-ai/Virchow", pretrained=True,
        mlp_layer=SwiGLUPacked, act_layer=nn.SiLU, num_classes=0,
    )
    return _attach_head_and_freeze(ClsMeanPatchWrapper(vit), n_classes, device)


def load_virchow2(n_classes: int = 2, device: str = "cuda"):
    vit = timm.create_model(
        "hf-hub:paige-ai/Virchow2", pretrained=True,
        mlp_layer=SwiGLUPacked, act_layer=nn.SiLU, num_classes=0,
    )
    return _attach_head_and_freeze(ClsMeanPatchWrapper(vit), n_classes, device)


def load_h0_mini(n_classes: int = 2, device: str = "cuda:0"):
    vit = timm.create_model(
        "hf-hub:bioptimus/H0-mini", pretrained=True,
        mlp_layer=SwiGLUPacked, act_layer=nn.SiLU, num_classes=0,
    )
    return _attach_head_and_freeze(ClsMeanPatchWrapper(vit), n_classes, device)


def load_h_optimus0(n_classes: int = 2, device: str = "cuda"):
    model = timm.create_model(
        "hf-hub:bioptimus/H-optimus-0", pretrained=True, num_classes=0,
        init_values=1e-5, dynamic_img_size=False,
    )
    return _attach_head_and_freeze(model, n_classes, device)


def load_h_optimus1(n_classes: int = 2, device: str = "cuda"):
    # Note: unlike H-Optimus-0 this does not pass num_classes=0 (kept as in the dissertation runs).
    model = timm.create_model(
        "hf-hub:bioptimus/H-optimus-1", pretrained=True, init_values=1e-5, dynamic_img_size=False
    )
    return _attach_head_and_freeze(model, n_classes, device)


class ExaoneWrapper(nn.Module):
    def __init__(self, backbone, n_classes):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(backbone.embed_dim, n_classes),
            nn.LogSoftmax(dim=1)
        )
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.head(self.backbone(x))


def load_exaonepath(n_classes: int = 2, device: str = "cuda"):
    ensure_exaonepath()
    from vision_transformer import VisionTransformer

    backbone = VisionTransformer.from_pretrained("LGAI-EXAONE/EXAONEPath")
    model = ExaoneWrapper(backbone, n_classes).to(device).train()
    return model, optim.SGD(model.head.parameters(), lr=1e-3, momentum=0.9)


def _load_hf_cls(repo_id: str, n_classes: int, device: str, **hf_kwargs):
    backbone = AutoModel.from_pretrained(repo_id, **hf_kwargs)
    return _attach_head_and_freeze(HFCLSWrapper(backbone), n_classes, device)


def load_hibou_b(n_classes: int = 2, device: str = "cuda"):
    return _load_hf_cls("histai/hibou-b", n_classes, device, trust_remote_code=True)


def load_hibou_l(n_classes: int = 2, device: str = "cuda"):
    return _load_hf_cls("histai/hibou-L", n_classes, device, trust_remote_code=True)


def load_phikon(n_classes: int = 2, device: str = "cuda"):
    return _load_hf_cls("owkin/phikon", n_classes, device)


def load_phikon_v2(n_classes: int = 2, device: str = "cuda"):
    return _load_hf_cls("owkin/phikon-v2", n_classes, device)


# ─── input normalisation (applied to 0-1 tensors *after* the perturbation) ───

IMAGENET_NORM = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
HOPTIMUS_NORM = transforms.Normalize((0.707223, 0.578729, 0.703617), (0.211883, 0.230117, 0.177517))


class ProcessorTransform:
    """Use a HuggingFace image processor for normalisation only (no resize/crop)."""

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, img):
        extra = dict(do_resize=False, do_center_crop=False)
        # skip the extra 1/255 when we already have float tensors
        if isinstance(img, torch.Tensor) and img.dtype.is_floating_point:
            extra["do_rescale"] = False
        out = self.processor(img, return_tensors="pt", **extra)["pixel_values"]
        return out if out.ndim == 4 else out.squeeze(0)


@lru_cache(maxsize=None)
def hf_processor(repo_id: str):
    return AutoImageProcessor.from_pretrained(repo_id, trust_remote_code=True)


def processor_norm(repo_id: str):
    return lambda model: ProcessorTransform(hf_processor(repo_id))


def timm_norm(model):
    """Normalisation from the timm pretrained config (model or model.backbone)."""
    if hasattr(model, "pretrained_cfg"):
        timm_model = model
    elif hasattr(model, "backbone") and hasattr(model.backbone, "pretrained_cfg"):
        timm_model = model.backbone
    else:
        raise AttributeError("Cannot find .pretrained_cfg on the model or its backbone.")
    cfg = resolve_data_config(timm_model.pretrained_cfg, model=timm_model)
    return transforms.Normalize(mean=cfg["mean"], std=cfg["std"])


def fixed_norm(norm):
    return lambda model: norm


# ─── registry ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ModelSpec:
    name: str                 # display name; also the Stats/<dataset>/<name>/ folder
    weight_key: str           # checkpoint file: Models/<dataset>/<weight_key>_<dataset>.pth
    build: Callable           # (n_classes, device) -> (model, optimizer)
    normalisation: Callable   # model -> transform applied to 0-1 tensors
    params_m: float           # parameter count in millions (for PPI / k-fold plots)
    full_checkpoint: bool = False  # True: save/load the whole model, not just the head


MODELS = {spec.name: spec for spec in [
    ModelSpec("ResNet18",    "resnet18",   load_resnet18,   fixed_norm(IMAGENET_NORM), 11.2, full_checkpoint=True),
    ModelSpec("ResNet50",    "resnet50",   load_resnet50,   fixed_norm(IMAGENET_NORM), 23.5, full_checkpoint=True),
    ModelSpec("UNI",         "uni",        load_uni,        fixed_norm(IMAGENET_NORM), 303.4),
    ModelSpec("UNI2",        "uni2",       load_uni2,       fixed_norm(IMAGENET_NORM), 681.4),
    ModelSpec("GigaPath",    "gigapath",   load_gigapath,   fixed_norm(IMAGENET_NORM), 1135.0),
    ModelSpec("Virchow",     "virchow",    load_virchow,    timm_norm,                 631.2),
    ModelSpec("Virchow2",    "virchow2",   load_virchow2,   timm_norm,                 631.2),
    ModelSpec("H0-mini",     "h0_mini",    load_h0_mini,    timm_norm,                 85.7),
    ModelSpec("H-Optimus-0", "hoptimus0",  load_h_optimus0, fixed_norm(HOPTIMUS_NORM), 1134.8),
    ModelSpec("H-Optimus-1", "hoptimus1",  load_h_optimus1, fixed_norm(HOPTIMUS_NORM), 1134.8),
    ModelSpec("EXAONEPath",  "exaonepath", load_exaonepath, fixed_norm(IMAGENET_NORM), 85.8),
    ModelSpec("Hibou-B",     "hibou_b",    load_hibou_b,    processor_norm("histai/hibou-b"), 85.7),
    ModelSpec("Hibou-L",     "hibou_l",    load_hibou_l,    processor_norm("histai/hibou-L"), 303.7),
    ModelSpec("Phikon v1",   "phikon",     load_phikon,     processor_norm("owkin/phikon"),   86.4),
    ModelSpec("Phikon v2",   "phikon_v2",  load_phikon_v2,  processor_norm("owkin/phikon-v2"), 303.4),
]}


def model_colour(name: str):
    """Stable tab20 colour per model, in registry order (matches the dissertation figures)."""
    import matplotlib.pyplot as plt
    names = list(MODELS)
    idx = names.index(name) if name in names else abs(hash(name))
    return plt.get_cmap("tab20").colors[idx % 20]


def get_transform(name: str, model: nn.Module):
    return MODELS[name].normalisation(model)


def weight_path(name: str, dataset: str):
    return models_dir(dataset, f"{MODELS[name].weight_key}_{dataset}.pth")


def load_for_training(name: str, device, n_classes: int = 2):
    """Fresh model with an untrained head: (model, optimizer)."""
    return MODELS[name].build(n_classes=n_classes, device=device)


def load_for_eval(name: str, dataset: str, device, n_classes: int = 2):
    """Model with the head trained on `dataset`, in eval mode."""
    spec = MODELS[name]
    path = weight_path(name, dataset)
    if spec.full_checkpoint:
        model = torch.load(path, weights_only=False)
    else:
        model, _ = spec.build(n_classes=n_classes, device=device)
        model.head.load_state_dict(torch.load(path))
    return model.to(device).eval()


def save_trained(name: str, dataset: str, model: nn.Module):
    path = weight_path(name, dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    if MODELS[name].full_checkpoint:
        torch.save(model, path)
    else:
        torch.save(model.head.state_dict(), path)
    return path
