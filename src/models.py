"""Model definitions, reimplemented from scratch.

Two backbones, chosen by input resolution (this is deliberate and matches the
MedMNIST v2 protocol):

* ``size == 28``  -> a CIFAR-style ResNet: 3x3 stride-1 stem, *no* max-pool,
  four residual stages with strides ``[1, 2, 2, 2]`` and widths
  ``[64, 128, 256, 512]``. ResNet-18 uses ``BasicBlock`` with ``[2,2,2,2]``;
  ResNet-50 uses ``Bottleneck`` (expansion 4) with ``[3,4,6,3]``.
* ``size == 224`` -> ``torchvision.models.resnet18/50`` with the standard
  ImageNet 7x7 stride-2 stem + max-pool.

A ``width_mult`` knob gives the lightweight 0.5x variant (widths
``[32,64,128,256]``) used by the efficiency extension.

None of this is copied from ``MedMNIST/experiments`` or ``kuangliu/pytorch-cifar``;
it is written to the spec above.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return F.relu(out + identity, inplace=True)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return F.relu(out + identity, inplace=True)


class CifarResNet(nn.Module):
    """CIFAR-style ResNet for 28x28 inputs (3x3 stem, no max-pool)."""

    def __init__(self, block, layers, num_classes, in_channels=3, width_mult=1.0):
        super().__init__()
        widths = [int(w * width_mult) for w in (64, 128, 256, 512)]
        self.in_planes = widths[0]

        self.conv1 = nn.Conv2d(in_channels, widths[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(widths[0])

        self.layer1 = self._make_layer(block, widths[0], layers[0], stride=1)
        self.layer2 = self._make_layer(block, widths[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, widths[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, widths[3], layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(widths[3] * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_planes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.in_planes, planes, stride, downsample)]
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def build_model(model_name, size, num_classes, in_channels=3, width_mult=1.0):
    """Build a model for the given architecture / resolution.

    Args:
        model_name: ``"resnet18"`` or ``"resnet50"``.
        size: 28 (CIFAR-style) or 224 (torchvision ImageNet backbone).
        num_classes: number of output logits.
        in_channels: input channels (3 for the RGB MedMNIST datasets we use).
        width_mult: channel multiplier; 0.5 gives the lightweight variant.
                    Only supported at size 28.
    """
    model_name = model_name.lower()
    if model_name not in ("resnet18", "resnet50"):
        raise ValueError(f"unknown model {model_name!r}")

    if size == 28:
        if model_name == "resnet18":
            return CifarResNet(BasicBlock, [2, 2, 2, 2], num_classes,
                               in_channels=in_channels, width_mult=width_mult)
        return CifarResNet(Bottleneck, [3, 4, 6, 3], num_classes,
                           in_channels=in_channels, width_mult=width_mult)

    if size == 224:
        if width_mult != 1.0:
            raise ValueError("width_mult is only supported for the size-28 CIFAR backbone")
        if in_channels != 3:
            raise ValueError("torchvision resnet backbone expects 3 input channels")
        import torchvision.models as tvm

        try:  # newer torchvision
            ctor = getattr(tvm, model_name)
            return ctor(weights=None, num_classes=num_classes)
        except TypeError:  # older torchvision without `weights=`
            ctor = getattr(tvm, model_name)
            return ctor(pretrained=False, num_classes=num_classes)

    raise ValueError(f"unsupported size {size}; expected 28 or 224")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_mb(model):
    """On-disk size in MB of the model's float32 state dict."""
    n_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    n_bytes += sum(b.numel() * b.element_size() for b in model.buffers())
    return n_bytes / (1024 ** 2)
