"""Pretrained backbones (no trained head) for the embedding analyses.

* ``KFOLD_EMBEDDERS`` – used by NR-Kfold (kfold.py): backbone, PIL preprocessing and an
  embedding function per model.
* ``SIMILARITY_BACKBONES`` – used by the train–test similarity sweeps (similarity.py).

These intentionally mirror the dissertation runs, including their quirks (e.g. the ResNet
"embeddings" are ImageNet logits; see README "Known issues"). To add a model to these
analyses, add an entry to each dict.
"""

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import timm
import torch
from timm.data import create_transform, resolve_data_config
from timm.layers import SwiGLUPacked
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel, AutoModelForImageClassification, ResNetForImageClassification

from .paths import ensure_exaonepath

IMAGENET = dict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))


# ─── PIL -> tensor preprocessing (full pipelines incl. resize/crop) ──────────

class PILProcessorTransform:
    """HuggingFace image processor applied to a PIL image."""

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, img):
        return self.processor(img, return_tensors="pt")["pixel_values"].squeeze(0)


@lru_cache(maxsize=None)
def _processor_transform(repo_id: str):
    return PILProcessorTransform(AutoImageProcessor.from_pretrained(repo_id, trust_remote_code=True))


UNI_PIL = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(**IMAGENET),
])
GIGA_PIL = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(**IMAGENET),
])
EXAONE_PIL = GIGA_PIL
HOPTIMUS_PIL = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.707223, 0.578729, 0.703617), std=(0.211883, 0.230117, 0.177517)),
])
VIRCHOW_PIL = transforms.Compose([
    transforms.Resize(size=224, interpolation=transforms.InterpolationMode.BICUBIC, max_size=None, antialias=True),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4850, 0.4560, 0.4060), std=(0.2290, 0.2240, 0.2250)),
])

# Static PIL preprocessing per model name.
_PIL_TRANSFORMS = {
    "ResNet18": lambda: _processor_transform("microsoft/resnet-18"),
    "ResNet34": lambda: _processor_transform("microsoft/resnet-18"),
    "ResNet50": lambda: _processor_transform("microsoft/resnet-18"),
    "ResNet101": lambda: _processor_transform("microsoft/resnet-18"),
    "ResNet152": lambda: _processor_transform("microsoft/resnet-18"),
    "UNI": lambda: UNI_PIL,
    "UNI2": lambda: UNI_PIL,
    "GigaPath": lambda: GIGA_PIL,
    "Virchow": lambda: VIRCHOW_PIL,
    "Virchow2": lambda: VIRCHOW_PIL,
    "EXAONEPath": lambda: EXAONE_PIL,
    "Hibou-B": lambda: _processor_transform("histai/hibou-b"),
    "Hibou-L": lambda: _processor_transform("histai/hibou-L"),
    "H-Optimus-0": lambda: HOPTIMUS_PIL,
    "H-Optimus-1": lambda: HOPTIMUS_PIL,
    "H0-mini": lambda: HOPTIMUS_PIL,
    "Phikon v1": lambda: _processor_transform("owkin/phikon"),
    "Phikon v2": lambda: _processor_transform("owkin/phikon-v2"),
}


def pil_transform(name: str, model=None):
    """PIL preprocessing for `name`. Virchow/Virchow2/H0-mini use timm's own config when a model is given."""
    if name in {"Virchow", "Virchow2", "H0-mini"} and model is not None:
        timm_model = model if hasattr(model, "pretrained_cfg") else model.backbone
        return create_transform(**resolve_data_config(timm_model.pretrained_cfg, model=timm_model))
    return _PIL_TRANSFORMS[name]()


# ─── NR-Kfold embedders ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class Embedder:
    init: Callable    # device -> backbone in eval mode
    embed: Callable   # (model, batch_tensor, device) -> np.ndarray (B, D)


def _timm(repo, **kw):
    return lambda device: timm.create_model(repo, pretrained=True, **kw).eval().to(device)


def _hf(repo, cls=AutoModel, **kw):
    return lambda device: cls.from_pretrained(repo, **kw).eval().to(device)


def _exaonepath(device):
    ensure_exaonepath()
    from vision_transformer import VisionTransformer
    return VisionTransformer.from_pretrained(
        "LGAI-EXAONE/EXAONEPath", use_auth_token=os.environ.get("HF_TOKEN")
    ).eval().to(device)


def _plain(model, x, device):
    return model(x.to(device)).detach().cpu().numpy()


def _hf_logits(model, x, device):
    return model(pixel_values=x.to(device)).logits.detach().cpu().numpy()


def _hf_cls(model, x, device):
    return model(pixel_values=x.to(device)).last_hidden_state[:, 0, :].detach().cpu().numpy()


def _cls_mean(skip):
    """concat(CLS, mean(tokens[:, skip:])). The model is called twice, as in the original runs."""
    def fn(model, x, device):
        return torch.cat([
            model(x.to(device))[:, 0],
            model(x.to(device))[:, skip:].mean(1)
        ], dim=-1).detach().cpu().numpy()
    return fn


_UNI2_KW = dict(img_size=224, patch_size=14, depth=24, num_heads=24, init_values=1e-5, embed_dim=1536,
                mlp_ratio=2.66667 * 2, num_classes=0, no_embed_class=True, mlp_layer=SwiGLUPacked,
                act_layer=torch.nn.SiLU, reg_tokens=8, dynamic_img_size=True)
_SWIGLU_KW = dict(mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)

KFOLD_EMBEDDERS = {
    "ResNet18": Embedder(_hf("microsoft/resnet-18", AutoModelForImageClassification), _hf_logits),
    "ResNet34": Embedder(_hf("microsoft/resnet-34", AutoModelForImageClassification), _hf_logits),
    "ResNet50": Embedder(_timm("resnet50"), _plain),
    "ResNet101": Embedder(_timm("resnet101"), _plain),
    "ResNet152": Embedder(_timm("resnet152"), _plain),
    "UNI": Embedder(_timm("hf-hub:MahmoodLab/uni", init_values=1e-5, dynamic_img_size=True), _plain),
    "UNI2": Embedder(_timm("hf-hub:MahmoodLab/UNI2-h", **_UNI2_KW), _plain),
    "GigaPath": Embedder(_timm("hf_hub:prov-gigapath/prov-gigapath"), _plain),
    "Virchow": Embedder(_timm("hf-hub:paige-ai/Virchow", **_SWIGLU_KW), _cls_mean(1)),
    "Virchow2": Embedder(_timm("hf-hub:paige-ai/Virchow2", **_SWIGLU_KW), _cls_mean(5)),
    "EXAONEPath": Embedder(_exaonepath, lambda m, x, d: m(x.to(d)).squeeze().detach().cpu().numpy()),
    "Hibou-B": Embedder(_hf("histai/hibou-b", trust_remote_code=True), _hf_cls),
    "Hibou-L": Embedder(_hf("histai/hibou-L", trust_remote_code=True), _hf_cls),
    "Phikon v1": Embedder(_hf("owkin/phikon", trust_remote_code=True), _hf_cls),
    "Phikon v2": Embedder(_hf("owkin/phikon-v2", trust_remote_code=True), _hf_cls),
    "H-Optimus-0": Embedder(_timm("hf-hub:bioptimus/H-optimus-0", init_values=1e-5, dynamic_img_size=False), _plain),
    "H-Optimus-1": Embedder(_timm("hf-hub:bioptimus/H-optimus-1", init_values=1e-5, dynamic_img_size=False), _plain),
    "H0-mini": Embedder(_timm("hf-hub:bioptimus/H0-mini", **_SWIGLU_KW), _cls_mean(1)),
}


# ─── train–test similarity backbones ─────────────────────────────────────────

SIMILARITY_BACKBONES = {
    "ResNet18": _hf("microsoft/resnet-18", AutoModelForImageClassification),
    "ResNet50": _hf("microsoft/resnet-50", ResNetForImageClassification),
    "UNI": _timm("hf-hub:MahmoodLab/uni", init_values=1e-5, dynamic_img_size=True),
    "UNI2": _timm("hf-hub:MahmoodLab/UNI2-h", **{k: v for k, v in _UNI2_KW.items() if k != "num_classes"}),
    "GigaPath": _timm("hf_hub:prov-gigapath/prov-gigapath"),
    "H-Optimus-0": _timm("hf-hub:bioptimus/H-optimus-0", init_values=1e-5, dynamic_img_size=False),
    "H-Optimus-1": _timm("hf-hub:bioptimus/H-optimus-1", init_values=1e-5, dynamic_img_size=False),
    "EXAONEPath": lambda device: _exaonepath(device),
    "Hibou-B": _hf("histai/hibou-b", trust_remote_code=True),
    "Hibou-L": _hf("histai/hibou-L", trust_remote_code=True),
    "Phikon v1": _hf("owkin/phikon"),
    "Phikon v2": _hf("owkin/phikon-v2"),
    # Excluded from the dissertation similarity analysis (embedding extraction issues):
    "H0-mini": _timm("hf-hub:bioptimus/H0-mini", **_SWIGLU_KW),
    "Virchow": _timm("hf-hub:paige-ai/Virchow", **_SWIGLU_KW),
    "Virchow2": _timm("hf-hub:paige-ai/Virchow2", **_SWIGLU_KW),
}
