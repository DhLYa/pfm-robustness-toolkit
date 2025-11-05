import os
import sys
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from timm.layers import SwiGLUPacked
from transformers import AutoModel, AutoImageProcessor
from vision_transformer import VisionTransformer

def _attach_head_and_freeze(model: nn.Module, n_classes: int, device: str):
    # infer feature‐dim
    num_features = getattr(model, 'num_features', None)
    if num_features is None:
        # fallback for timm classifiers named .head or .classifier
        if hasattr(model, 'head') and hasattr(model.head, 'in_features'):
            num_features = model.head.in_features
        elif hasattr(model, 'classifier') and hasattr(model.classifier, 'in_features'):
            num_features = model.classifier.in_features
        else:
            raise AttributeError("Cannot infer feature dimension")
    # replace head
    model.head = nn.Sequential(
        nn.Linear(num_features, n_classes),
        nn.LogSoftmax(dim=1)
    )
    # freeze backbone
    for p in model.parameters():
        p.requires_grad = False
    # unfreeze head
    for p in model.head.parameters():
        p.requires_grad = True
    model = model.to(device).train()
    optim_head = optim.SGD(model.head.parameters(), lr=1e-3, momentum=0.9)
    return model, optim_head

# ---------------------------------------------------------------------
# helpers for HF vision models
# ---------------------------------------------------------------------
class HFCLSWrapper(nn.Module):
    """
    Wrap a HuggingFace vision backbone so that:
      • forward() returns CLS features
      • .num_features is available for your helper
      • .head exists (will be replaced)
    """
    def __init__(self, backbone: AutoModel):
        super().__init__()
        self.backbone = backbone
        self.num_features = getattr(
            backbone.config, "hidden_size",
            getattr(backbone.config, "vision_dim", None)
        )
        if self.num_features is None:
            raise ValueError("Cannot find hidden size in config")
        self.head = nn.Identity()      # placeholder – replaced later

    def forward(self, pixel_values=None, **kwargs):
        outputs = self.backbone(pixel_values=pixel_values, **kwargs)
        # outputs.last_hidden_state → (B, N+1, D); CLS is at index 0
        last_hidden = (
            outputs.last_hidden_state
            if hasattr(outputs, "last_hidden_state")
            else outputs[0]
        )
        feats = last_hidden[:, 0, :]           # (B, D)
        return self.head(feats)

# ---------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------
def load_h0_mini(n_classes: int = 2, device: str = "cuda:0"):
    """
    Loads bioptimus/H0-mini with a trainable classification head.
    Freezes backbone; returns (model_with_head, optimizer_for_head)
    """
    import timm
    import torch
    import torch.nn as nn
    from timm.layers import SwiGLUPacked

    # Create the backbone with proper layers
    vit = timm.create_model(
        "hf-hub:bioptimus/H0-mini",
        pretrained=True,
        mlp_layer=SwiGLUPacked,
        act_layer=nn.SiLU,
        num_classes=0  # we'll add our own head
    )

    class H0MiniWrapper(nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone
            # Feature dimension is backbone.embed_dim * 2 (for [CLS, mean(patch)])
            self.num_features = backbone.embed_dim * 2
            self.head = nn.Identity()  # will be replaced

        def forward(self, x):
            tokens = self.backbone(x)
            cls_tok = tokens[:, 0]
            patch_tok = tokens[:, 5:].mean(1)  # ignore 4 register tokens (matches Virchow style)
            feats = torch.cat([cls_tok, patch_tok], dim=-1)
            return self.head(feats)

    wrapper = H0MiniWrapper(vit)
    return _attach_head_and_freeze(wrapper, n_classes, device)
    
def load_h_optimus0(n_classes: int = 2, device: str = "cuda"):
    """bioptimus/H-optimus-0 (ViT-B/16)"""
    timm_kwargs = dict(init_values=1e-5, dynamic_img_size=False)
    model = timm.create_model(
        "hf-hub:bioptimus/H-optimus-0",
        pretrained=True, num_classes=0, **timm_kwargs
    )
    return _attach_head_and_freeze(model, n_classes, device)

def load_h_optimus1(n_classes: int = 2, device: str = "cuda"):
    model = timm.create_model(
        "hf-hub:bioptimus/H-optimus-1", pretrained=True, init_values=1e-5, dynamic_img_size=False
    )
    return _attach_head_and_freeze(model, n_classes, device)

def load_exaonepath(n_classes: int = 2, device: str = "cuda"):
    """EXAONEPath loader that properly wires backbone and head"""

    backbone = VisionTransformer.from_pretrained("LGAI-EXAONE/EXAONEPath")

    # Wrapper Module
    class ExaoneWrapper(nn.Module):
        def __init__(self, backbone, n_classes):
            super().__init__()
            self.backbone = backbone
            self.head = nn.Sequential(
                nn.Linear(backbone.embed_dim, n_classes),
                nn.LogSoftmax(dim=1)
            )

            # Freeze backbone
            for p in self.backbone.parameters():
                p.requires_grad = False

        def forward(self, x):
            feats = self.backbone(x)  # <-- ensure this returns raw features, no .detach()
            return self.head(feats)

    model = ExaoneWrapper(backbone, n_classes).to(device).train()
    optimizer = optim.SGD(model.head.parameters(), lr=1e-3, momentum=0.9)
    return model, optimizer

def load_hibou_b(n_classes: int = 2, device: str = "cuda"):
    """histai/hibou-b"""
    processor = AutoImageProcessor.from_pretrained("histai/hibou-b", trust_remote_code=True)
    backbone  = AutoModel.from_pretrained("histai/hibou-b", trust_remote_code=True)
    model, opt = _attach_head_and_freeze(HFCLSWrapper(backbone), n_classes, device)
    return processor, model, opt

def load_hibou_l(n_classes: int = 2, device: str = "cuda"):
    """histai/hibou-L"""
    processor = AutoImageProcessor.from_pretrained("histai/hibou-L", trust_remote_code=True)
    backbone  = AutoModel.from_pretrained("histai/hibou-L", trust_remote_code=True)
    model, opt = _attach_head_and_freeze(HFCLSWrapper(backbone), n_classes, device)
    return processor, model, opt


def _load_phikon_like(repo_id: str, n_classes: int, device: str):
    processor = AutoImageProcessor.from_pretrained(repo_id)
    backbone  = AutoModel.from_pretrained(repo_id)
    model, opt = _attach_head_and_freeze(HFCLSWrapper(backbone), n_classes, device)
    return processor, model, opt

def load_phikon(n_classes: int = 2, device: str = "cuda"):
    """owkin/phikon (v1)"""
    return _load_phikon_like("owkin/phikon", n_classes, device)

def load_phikon_v2(n_classes: int = 2, device: str = "cuda"):
    """owkin/phikon-v2"""
    return _load_phikon_like("owkin/phikon-v2", n_classes, device)

def load_uni2(n_classes: int = 2, device: str = "cuda:0"):
    """MahmoodLab/UNI2-h (ViT-H/14)"""
    timm_kwargs = {
        'img_size': 224, 'patch_size': 14, 'depth': 24, 'num_heads': 24,
        'init_values': 1e-5, 'embed_dim': 1536, 'mlp_ratio': 2.66667*2,
        'num_classes': 0, 'no_embed_class': True, 'mlp_layer': timm.layers.SwiGLUPacked,
        'act_layer': nn.SiLU, 'reg_tokens': 8, 'dynamic_img_size': True
    }
    model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
    return _attach_head_and_freeze(model, n_classes, device)

def load_uni(n_classes: int = 2, device: str = "cuda:0"):
    """MahmoodLab/UNI (ViT-L/16) — must pass init_values & dynamic_img_size"""
    timm_kwargs = {
        'init_values': 1e-5,
        'dynamic_img_size': True
    }
    # note repo id is lowercase "uni"
    model = timm.create_model(
        "hf-hub:MahmoodLab/uni",
        pretrained=True,
        num_classes=0,
        **timm_kwargs
    )
    return _attach_head_and_freeze(model, n_classes, device)

def load_gigapath(n_classes: int = 2, device: str = "cuda:0"):
    """prov-gigapath/prov-gigapath patch-level model with a binary head."""
    model = timm.create_model(
        "hf-hub:prov-gigapath/prov-gigapath",
        pretrained=True,
        num_classes=0
    )
    return _attach_head_and_freeze(model, n_classes, device)

# -------------------------ResNet Models-------------------------

def load_resnet18(n_classes: int = 2, device: str = "cuda:0"):
    model = models.resnet18(pretrained=True)  
    model.fc = nn.Sequential(nn.Linear(512, n_classes),        
                            nn.LogSoftmax(dim=1))
    model = model.to(device)
    model.train()  

    params_to_update = model.parameters()
    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    return model, optimizer_ft

def load_resnet50(n_classes: int = 2, device: str = "cuda:0"):
    model = models.resnet50(pretrained=True)
    model.fc = nn.Sequential(
        nn.Linear(2048, n_classes),
        nn.LogSoftmax(dim=1)
    )
    model = model.to(device)
    model.train()

    params_to_update = model.parameters()
    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    return model, optimizer_ft

# ─── Virchow loaders ───────────────────────────────────────────────
def load_virchow(n_classes: int = 2, device: str = "cuda"):
    """
    paige-ai/Virchow — ViT-L with 4 register tokens.
    2 560-d feature vector: [CLS token, mean(patch tokens)].
    Returns (model_with_head, optimizer_for_head).
    """
    import timm
    import torch
    import torch.nn as nn
    from timm.layers import SwiGLUPacked

    vit = timm.create_model(
        "hf-hub:paige-ai/Virchow",
        pretrained=True,
        mlp_layer=SwiGLUPacked,
        act_layer=torch.nn.SiLU,
        num_classes=0
    )

    class VirchowWrapper(nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone
            self.num_features = backbone.embed_dim * 2
            self.head = nn.Identity()

        def forward(self, x):
            tokens = self.backbone(x)
            cls_tok = tokens[:, 0]
            patch_tok = tokens[:, 5:].mean(1)
            feats = torch.cat([cls_tok, patch_tok], dim=-1)
            return self.head(feats)

    wrapper = VirchowWrapper(vit)
    return _attach_head_and_freeze(wrapper, n_classes, device)


def load_virchow2(n_classes: int = 2, device: str = "cuda"):
    """
    paige-ai/Virchow2 — ViT-L with 4 register tokens.
    A 2 560-d feature vector is built from:
        [CLS token,  mean(patch tokens)].
    Returns (model_with_head, optimizer_for_head)
    """
    import timm
    import torch
    import torch.nn as nn
    from timm.layers import SwiGLUPacked

    # 1. backbone (returns all tokens, no classifier head)
    vit = timm.create_model(
        "hf-hub:paige-ai/Virchow2",
        pretrained=True,
        mlp_layer=SwiGLUPacked,          # required by repo
        act_layer=torch.nn.SiLU,
        num_classes=0                    # we want features, not logits
    )

    # 2. Wrap to expose a single feature tensor + num_features
    class Virchow2Wrapper(nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone
            self.num_features = backbone.embed_dim * 2        # 1 280 × 2 = 2 560
            self.head = nn.Identity()                         # will be replaced

        def forward(self, x):
            tokens = self.backbone(x)            # (B, 261, 1280)
            cls_tok = tokens[:, 0]               # (B, 1280)
            patch_tok = tokens[:, 5:].mean(1)    # ignore 4 register tokens
            feats = torch.cat([cls_tok, patch_tok], dim=-1)   # (B, 2560)
            return self.head(feats)

    wrapper = Virchow2Wrapper(vit)

    # 3. Add your classification head & freeze backbone
    return _attach_head_and_freeze(wrapper, n_classes, device)

# ---------------------------------------------------------------------