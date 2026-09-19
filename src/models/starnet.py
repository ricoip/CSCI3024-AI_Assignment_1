"""StarNet from Ma et al., 'Rewrite the Stars', CVPR 2024.

The architecture follows the official implementation at
https://github.com/ma-xu/Rewrite-the-Stars/blob/main/imagenet/starnet.py
with two assignment-specific changes:

- `use_star=False` replaces the element-wise product with a sum (the paper's
  DemoNet / StarNet ablation).
- `stem_stride` defaults to 1 for 32x32 CIFAR images (the paper uses 2 for 224).

DropPath and truncated-normal init are inlined so `timm` is not required.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
from torchvision.models import resnet18


class DropPath(nn.Module):
    """Stochastic depth. Identity at eval time or when drop_prob is 0."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep


class ConvBN(nn.Sequential):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        with_bn: bool = True,
    ) -> None:
        super().__init__()
        self.add_module(
            "conv",
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
                bias=not with_bn,
            ),
        )
        if with_bn:
            bn = nn.BatchNorm2d(out_planes)
            nn.init.constant_(bn.weight, 1.0)
            nn.init.constant_(bn.bias, 0.0)
            self.add_module("bn", bn)


class Block(nn.Module):
    """StarNet block. Star path: ReLU6(f1(x)) * f2(x); sum path uses +."""

    def __init__(
        self,
        dim: int,
        mlp_ratio: int = 4,
        drop_path: float = 0.0,
        use_star: bool = True,
    ) -> None:
        super().__init__()
        self.use_star = use_star
        hidden = mlp_ratio * dim
        self.dwconv = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=True)
        self.f1 = ConvBN(dim, hidden, 1, with_bn=False)
        self.f2 = ConvBN(dim, hidden, 1, with_bn=False)
        self.g = ConvBN(hidden, dim, 1, with_bn=True)
        self.dwconv2 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=False)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2 if self.use_star else self.act(x1) + x2
        x = self.dwconv2(self.g(x))
        return residual + self.drop_path(x)


class StarNet(nn.Module):
    def __init__(
        self,
        base_dim: int = 32,
        depths: Sequence[int] = (3, 3, 12, 5),
        mlp_ratio: int = 4,
        drop_path_rate: float = 0.0,
        num_classes: int = 1000,
        stem_stride: int = 2,
        use_star: bool = True,
        in_chans: int = 3,
        **kwargs,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.use_star = use_star
        self.depths = list(depths)
        self.in_channel = 32
        self.stem = nn.Sequential(
            ConvBN(in_chans, self.in_channel, kernel_size=3, stride=stem_stride, padding=1),
            nn.ReLU6(),
        )
        dpr = torch.linspace(0, drop_path_rate, sum(self.depths)).tolist()
        self.stages = nn.ModuleList()
        cur = 0
        for i_layer, depth in enumerate(self.depths):
            embed_dim = base_dim * (2 ** i_layer)
            down_sampler = ConvBN(self.in_channel, embed_dim, 3, 2, 1)
            self.in_channel = embed_dim
            blocks = [
                Block(self.in_channel, mlp_ratio, dpr[cur + i], use_star=use_star)
                for i in range(depth)
            ]
            cur += depth
            self.stages.append(nn.Sequential(down_sampler, *blocks))
        self.norm = nn.BatchNorm2d(self.in_channel)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(self.in_channel, num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return self.norm(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = torch.flatten(self.avgpool(x), 1)
        return self.head(x)


def starnet_s1(**kwargs) -> StarNet:
    return StarNet(24, [2, 2, 8, 3], **kwargs)


def sumnet_s1(**kwargs) -> StarNet:
    kwargs = dict(kwargs)
    kwargs["use_star"] = False
    return StarNet(24, [2, 2, 8, 3], **kwargs)


def cifar_resnet18(num_classes: int = 10, **kwargs) -> nn.Module:
    """ResNet-18 with a CIFAR-friendly 3x3 stride-1 stem (no 7x7 / maxpool)."""
    model = resnet18(num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def build_model(
    name: str,
    num_classes: int = 10,
    stem_stride: int = 1,
    drop_path_rate: float = 0.0,
    **kwargs,
) -> nn.Module:
    key = name.lower().replace("-", "_")
    if key in {"starnet_s1", "starnet"}:
        return starnet_s1(
            num_classes=num_classes,
            stem_stride=stem_stride,
            drop_path_rate=drop_path_rate,
            use_star=True,
            **kwargs,
        )
    if key in {"sumnet_s1", "sumnet"}:
        return sumnet_s1(
            num_classes=num_classes,
            stem_stride=stem_stride,
            drop_path_rate=drop_path_rate,
            **kwargs,
        )
    if key in {"resnet18", "resnet_18"}:
        return cifar_resnet18(num_classes=num_classes, **kwargs)
    raise ValueError(f"Unknown model: {name}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_flops(model: nn.Module, input_size: Sequence[int] = (1, 3, 32, 32)) -> int:
    """Approximate MACs from Conv2d / Linear layers during one forward pass."""
    macs = 0
    handles = []

    def conv_hook(mod: nn.Conv2d, inp, out) -> None:
        nonlocal macs
        n, _, oh, ow = out.shape
        k_h, k_w = mod.kernel_size
        macs += n * out.shape[1] * oh * ow * (mod.in_channels // mod.groups) * k_h * k_w

    def linear_hook(mod: nn.Linear, inp, out) -> None:
        nonlocal macs
        macs += out.numel() * mod.in_features

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(linear_hook))

    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(tuple(input_size), device=device)
        model(dummy)
    model.train(was_training)
    for h in handles:
        h.remove()
    return int(macs)
