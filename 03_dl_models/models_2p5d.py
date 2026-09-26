"""2.5D network definitions.

The candidate networks are taken from the torchvision classification model zoo, which is the same
model zoo used by the deep-learning pipeline of the study, plus two lightweight baselines
(`smallcnn`, `simplevit`) that do not require downloading anything.

Every model accepts a configurable number of input channels and returns a single logit per case
(binary pCR prediction). No pretrained weights are downloaded unless ``pretrained=True``.

The manuscript compares **32 candidate 2.5D networks**; the exact list of the 32 is declared in
``dl_config.yaml`` under ``candidates``. ``MODEL_NAMES`` below is the pool of architectures from
which those candidates were chosen.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SmallCNN(nn.Module):
    """Compact convolutional baseline (no download required)."""

    def __init__(self, in_channels: int = 3, width: int = 32, dropout: float = 0.3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(width, width * 2, 3, padding=1),
            nn.BatchNorm2d(width * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(width * 2, width * 4, 3, padding=1),
            nn.BatchNorm2d(width * 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(width * 4, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x)).squeeze(-1)


class SimpleViT2p5D(nn.Module):
    """Minimal patch-attention baseline (the 'SimpleViT'-style model of the comparison)."""

    def __init__(self, in_channels: int = 3, patch: int = 16, dim: int = 128, depth: int = 4,
                 heads: int = 4, image_size: int = 224) -> None:
        super().__init__()
        self.patch = patch
        self.num_patches = (image_size // patch) ** 2
        self.proj = nn.Conv2d(in_channels, dim, kernel_size=patch, stride=patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, self.num_patches + 1, dim))
        layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, batch_first=True,
                                           dim_feedforward=dim * 4, dropout=0.1)
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls.expand(x.size(0), -1, -1), x], dim=1)
        x = x + self.pos[:, : x.size(1)]
        x = self.encoder(x)
        return self.head(self.norm(x[:, 0])).squeeze(-1)


# torchvision model-zoo entries available as 2.5D candidates, grouped by family.
TV_ZOO: dict[str, str] = {
    # AlexNet
    "alexnet": "alexnet",
    # VGG
    "vgg11": "vgg11", "vgg11_bn": "vgg11_bn", "vgg13": "vgg13", "vgg13_bn": "vgg13_bn",
    "vgg16": "vgg16", "vgg16_bn": "vgg16_bn", "vgg19": "vgg19", "vgg19_bn": "vgg19_bn",
    # ResNet / ResNeXt / Wide ResNet
    "resnet18": "resnet18", "resnet34": "resnet34", "resnet50": "resnet50",
    "resnet101": "resnet101", "resnet152": "resnet152",
    "resnext50_32x4d": "resnext50_32x4d", "resnext101_32x8d": "resnext101_32x8d",
    "wide_resnet50_2": "wide_resnet50_2", "wide_resnet101_2": "wide_resnet101_2",
    # DenseNet
    "densenet121": "densenet121", "densenet161": "densenet161",
    "densenet169": "densenet169", "densenet201": "densenet201",
    # Inception / GoogLeNet
    "googlenet": "googlenet", "inception_v3": "inception_v3",
    # SqueezeNet
    "squeezenet1_0": "squeezenet1_0", "squeezenet1_1": "squeezenet1_1",
    # ShuffleNet v2
    "shufflenet_v2_x0_5": "shufflenet_v2_x0_5", "shufflenet_v2_x1_0": "shufflenet_v2_x1_0",
    "shufflenet_v2_x1_5": "shufflenet_v2_x1_5", "shufflenet_v2_x2_0": "shufflenet_v2_x2_0",
    # MobileNet
    "mobilenet_v2": "mobilenet_v2", "mobilenet_v3_large": "mobilenet_v3_large",
    "mobilenet_v3_small": "mobilenet_v3_small",
    # MNASNet
    "mnasnet0_5": "mnasnet0_5", "mnasnet0_75": "mnasnet0_75",
    "mnasnet1_0": "mnasnet1_0", "mnasnet1_3": "mnasnet1_3",
    # Vision transformers
    "vit": "vit_b_16",  # the "ViT" entry of the manuscript (torchvision ViT-B/16)
    "vit_b_16": "vit_b_16", "vit_b_32": "vit_b_32", "vit_l_16": "vit_l_16",
    "swin_t": "swin_t", "swin_s": "swin_s", "swin_b": "swin_b",
    "convnext_tiny": "convnext_tiny", "convnext_small": "convnext_small",
    "convnext_base": "convnext_base",
    "maxvit_t": "maxvit_t",
}

CUSTOM_NAMES = ["smallcnn", "simplevit"]

MODEL_NAMES = CUSTOM_NAMES + sorted(TV_ZOO)

# Input size required by the architecture (inception_v3 needs 299 x 299).
INPUT_SIZE = {"inception_v3": 299}


def input_size_for(name: str, default: int = 224) -> int:
    return INPUT_SIZE.get(name.lower(), default)


def _replace_first_conv(model: nn.Module, in_channels: int) -> None:
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d):
            new = nn.Conv2d(
                in_channels,
                module.out_channels,
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                bias=module.bias is not None,
            )
            setattr(model, name, new)
            return
        _replace_first_conv(module, in_channels)


def _wrap_classifier(model: nn.Module, out_features: int = 1) -> nn.Module:
    """Replace the final classification layer so that the model outputs a single logit per case."""
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        model.fc = nn.Linear(model.fc.in_features, out_features)
    elif hasattr(model, "classifier"):
        clf = model.classifier
        if isinstance(clf, nn.Sequential):
            linears = [m for m in clf if isinstance(m, nn.Linear)]
            convs = [m for m in clf if isinstance(m, nn.Conv2d)]
            if linears:  # e.g. vgg, densenet, efficientnet, maxvit
                last = linears[-1]
                clf[list(clf).index(last)] = nn.Linear(last.in_features, out_features)
            elif convs:  # e.g. squeezenet: 1x1 convolution classifier
                last = convs[-1]
                clf[list(clf).index(last)] = nn.Conv2d(
                    last.in_channels, out_features, kernel_size=last.kernel_size
                )
        elif isinstance(clf, nn.Linear):
            model.classifier = nn.Linear(clf.in_features, out_features)
    elif hasattr(model, "heads") and isinstance(model.heads, nn.Sequential):  # torchvision ViT
        linears = [m for m in model.heads if isinstance(m, nn.Linear)]
        if linears:
            last = linears[-1]
            model.heads[list(model.heads).index(last)] = nn.Linear(last.in_features, out_features)
    elif hasattr(model, "head") and isinstance(model.head, nn.Linear):
        model.head = nn.Linear(model.head.in_features, out_features)
    else:
        raise ValueError(f"Cannot locate the classification layer of {type(model).__name__}.")
    return _FlattenOutput(model)


class _FlattenOutput(nn.Module):
    """Flatten the backbone output so that every model returns a tensor of shape (N,)."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(x)
        if isinstance(out, (tuple, list)):  # e.g. GoogLeNet with auxiliary classifiers
            out = out[0]
        elif hasattr(out, "logits"):
            out = out.logits
        out = out.flatten(1)
        return out[:, 0] if out.shape[1] == 1 else out


def build_model(name: str, in_channels: int = 3, pretrained: bool = False) -> nn.Module:
    """Return a 2.5D network by name (see ``MODEL_NAMES``)."""
    key = name.lower()

    if key == "smallcnn":
        return SmallCNN(in_channels=in_channels)

    if key == "simplevit":
        return SimpleViT2p5D(in_channels=in_channels)

    if key not in TV_ZOO:
        raise ValueError(
            f"Unknown model '{name}'. Available: {', '.join(MODEL_NAMES)}"
        )

    try:
        from torchvision import models
    except ImportError as exc:  # pragma: no cover
        raise ImportError("torchvision is required for the torchvision backbones.") from exc

    factory = getattr(models, TV_ZOO[key])
    kwargs: dict = {}
    if key in ("googlenet", "inception_v3") and not (
        pretrained and key == "inception_v3"
    ):
        # avoid the auxiliary classifiers so that the model returns a single logit tensor
        kwargs["aux_logits"] = False
    weights = "DEFAULT" if pretrained else None
    try:
        model = factory(weights=weights, **kwargs)
    except TypeError:  # older torchvision releases use pretrained=
        model = factory(pretrained=pretrained, **kwargs)
    if in_channels != 3:
        _replace_first_conv(model, in_channels)
    return _wrap_classifier(model)
