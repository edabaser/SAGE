
"""
Model/resnet8.py — Orijinal SAGE ResNet8

Orijinal SAGE paperındaki custom küçük ResNet.
CIFAR10/CIFAR100/SVHN gibi küçük görüntüler için tasarlanmış.
HAM10000 ile kullanılıyorsa avgpool otomatik düzeltilir.
"""

import math
import torch
import torch.nn as nn


def norm2d(group_norm_num_groups, planes):
    if group_norm_num_groups is not None and group_norm_num_groups > 0:
        return nn.GroupNorm(group_norm_num_groups, planes)
    else:
        return nn.BatchNorm2d(planes)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, out_planes, stride=1,
                 downsample=None, group_norm_num_groups=None):
        super().__init__()
        self.conv1 = conv3x3(in_planes, out_planes, stride)
        self.bn1   = norm2d(group_norm_num_groups, out_planes)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_planes, out_planes)
        self.bn2   = norm2d(group_norm_num_groups, out_planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)


class ResNet8(nn.Module):
    """
    Orijinal SAGE custom ResNet-8.
    resnet_size=8, scaling=4 → CIFAR için standart konfigürasyon.
    """

    def __init__(self, resnet_size=8, scaling=4, save_activations=False,
                 group_norm_num_groups=None, freeze_bn=False,
                 freeze_bn_affine=False, num_classes=10):
        super().__init__()
        self.freeze_bn       = freeze_bn
        self.freeze_bn_affine = freeze_bn_affine
        self.num_classes     = num_classes
        self.save_activations = save_activations

        if resnet_size % 6 != 2:
            raise ValueError(f"resnet_size must be 6n+2, got {resnet_size}")

        block_nums = (resnet_size - 2) // 6
        block_fn   = BasicBlock  # resnet_size < 44

        self.inplanes = int(16 * scaling)
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1   = norm2d(group_norm_num_groups, self.inplanes)
        self.relu  = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block_fn, int(16 * scaling),
                                       block_nums, stride=1,
                                       gn=group_norm_num_groups)
        self.layer2 = self._make_layer(block_fn, int(32 * scaling),
                                       block_nums, stride=2,
                                       gn=group_norm_num_groups)
        self.layer3 = self._make_layer(block_fn, int(64 * scaling),
                                       block_nums, stride=2,
                                       gn=group_norm_num_groups)

        # Orijinal SAGE: sabit 8x8 pool — küçük görüntüler için
        # HAM10000 gibi büyük görüntülerde build_model() bunu AdaptiveAvgPool2d ile değiştirir
        self.avgpool    = nn.AvgPool2d(kernel_size=8)
        self.classifier = nn.Linear(
            int(64 * scaling * block_fn.expansion), num_classes)

        self._init_weights()
        self.activations = None

    def _make_layer(self, block_fn, planes, num_blocks, stride=1, gn=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block_fn.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block_fn.expansion,
                          kernel_size=1, stride=stride, bias=False),
                norm2d(gn, planes * block_fn.expansion),
            )
        layers = [block_fn(self.inplanes, planes, stride, downsample, gn)]
        self.inplanes = planes * block_fn.expansion
        for _ in range(1, num_blocks):
            layers.append(block_fn(self.inplanes, planes,
                                   group_norm_num_groups=gn))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        a1 = x = self.layer1(x)
        a2 = x = self.layer2(x)
        a3 = x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        feature = x
        y = self.classifier(x)
        if self.save_activations:
            self.activations = [a1, a2, a3]
        return feature, y

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_bn:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    if self.freeze_bn_affine:
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False
