# from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
# from torch.nn import Module, Conv2d, Linear, MaxPool2d
# # import math
# # import torch.nn as nn
# # import copy
# # import torch

# # class ResNetBase(nn.Module):
# #     def _decide_num_classes(self):
# #         if self.dataset == "cifar10" or self.dataset == "svhn":
# #             return 10
# #         elif self.dataset == "cifar100":
# #             return 100
# #         elif "imagenet" in self.dataset:
# #             return 1000
# #         elif "femnist" == self.dataset:
# #             return 62

# #     def _weight_initialization(self):
# #         for m in self.modules():
# #             if isinstance(m, nn.Conv2d):
# #                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
# #                 m.weight.data.normal_(0, math.sqrt(2.0 / n))
# #             elif isinstance(m, nn.BatchNorm2d):
# #                 m.weight.data.fill_(1)
# #                 m.bias.data.zero_()

# #     def _make_block(
# #         self, block_fn, planes, block_num, stride=1, group_norm_num_groups=None
# #     ):
# #         downsample = None
# #         if stride != 1 or self.inplanes != planes * block_fn.expansion:
# #             downsample = nn.Sequential(
# #                 nn.Conv2d(
# #                     self.inplanes,
# #                     planes * block_fn.expansion,
# #                     kernel_size=1,
# #                     stride=stride,
# #                     bias=False,
# #                 ),
# #                 norm2d(group_norm_num_groups, planes=planes * block_fn.expansion),
# #             )

# #         layers = []
# #         layers.append(
# #             block_fn(
# #                 in_planes=self.inplanes,
# #                 out_planes=planes,
# #                 stride=stride,
# #                 downsample=downsample,
# #                 group_norm_num_groups=group_norm_num_groups,
# #             )
# #         )
# #         self.inplanes = planes * block_fn.expansion

# #         for _ in range(1, block_num):
# #             layers.append(
# #                 block_fn(
# #                     in_planes=self.inplanes,
# #                     out_planes=planes,
# #                     group_norm_num_groups=group_norm_num_groups,
# #                 )
# #             )
# #         return nn.Sequential(*layers)

# #     def train(self, mode=True):
# #         super(ResNetBase, self).train(mode)

# #         if self.freeze_bn:
# #             for m in self.modules():
# #                 if isinstance(m, nn.BatchNorm2d):
# #                     m.eval()
# #                     if self.freeze_bn_affine:
# #                         m.weight.requires_grad = False
# #                         m.bias.requires_grad = False


# # def norm2d(group_norm_num_groups, planes):
# #     if group_norm_num_groups is not None and group_norm_num_groups > 0:
# #         # group_norm_num_groups == planes -> InstanceNorm
# #         # group_norm_num_groups == 1 -> LayerNorm
# #         return nn.GroupNorm(group_norm_num_groups, planes)
# #     else:
# #         return nn.BatchNorm2d(planes)


# # class Bottleneck(nn.Module):
# #     """
# #     [1 * 1, x]
# #     [3 * 3, x]
# #     [1 * 1, x * 4]
# #     """

# #     expansion = 4

# #     def __init__(
# #         self,
# #         in_planes,
# #         out_planes,
# #         stride=1,
# #         downsample=None,
# #         group_norm_num_groups=None,
# #     ):
# #         super(Bottleneck, self).__init__()
# #         self.conv1 = nn.Conv2d(
# #             in_channels=in_planes, out_channels=out_planes, kernel_size=1, bias=False
# #         )
# #         self.bn1 = norm2d(group_norm_num_groups, planes=out_planes)

# #         self.conv2 = nn.Conv2d(
# #             in_channels=out_planes,
# #             out_channels=out_planes,
# #             kernel_size=3,
# #             stride=stride,
# #             padding=1,
# #             bias=False,
# #         )
# #         self.bn2 = norm2d(group_norm_num_groups, planes=out_planes)

# #         self.conv3 = nn.Conv2d(
# #             in_channels=out_planes,
# #             out_channels=out_planes * 4,
# #             kernel_size=1,
# #             bias=False,
# #         )
# #         self.bn3 = norm2d(group_norm_num_groups, planes=out_planes * 4)
# #         self.relu = nn.ReLU(inplace=True)

# #         self.downsample = downsample
# #         self.stride = stride

# #     def forward(self, x):
# #         residual = x

# #         out = self.conv1(x)
# #         out = self.bn1(out)
# #         out = self.relu(out)

# #         out = self.conv2(out)
# #         out = self.bn2(out)
# #         out = self.relu(out)

# #         out = self.conv3(out)
# #         out = self.bn3(out)

# #         if self.downsample is not None:
# #             residual = self.downsample(x)

# #         out += residual
# #         out = self.relu(out)

# #         return out


# # def conv3x3(in_planes, out_planes, stride=1):
# #     "3x3 convolution with padding."
# #     return nn.Conv2d(
# #         in_channels=in_planes,
# #         out_channels=out_planes,
# #         kernel_size=3,
# #         stride=stride,
# #         padding=1,
# #         bias=False,
# #     )



# # class BasicBlock(nn.Module):
# #     """
# #     [3 * 3, 64]
# #     [3 * 3, 64]
# #     """

# #     expansion = 1

# #     def __init__(
# #         self,
# #         in_planes,
# #         out_planes,
# #         stride=1,
# #         downsample=None,
# #         group_norm_num_groups=None,
# #     ):
# #         super(BasicBlock, self).__init__()
# #         self.conv1 = conv3x3(in_planes, out_planes, stride)
# #         self.bn1 = norm2d(group_norm_num_groups, planes=out_planes)
# #         self.relu = nn.ReLU(inplace=True)

# #         self.conv2 = conv3x3(out_planes, out_planes)
# #         self.bn2 = norm2d(group_norm_num_groups, planes=out_planes)

# #         self.downsample = downsample
# #         self.stride = stride

# #     def forward(self, x):
# #         residual = x

# #         out = self.conv1(x)
# #         out = self.bn1(out)
# #         out = self.relu(out)

# #         out = self.conv2(out)
# #         out = self.bn2(out)

# #         if self.downsample is not None:
# #             residual = self.downsample(x)

# #         out += residual
# #         out = self.relu(out)

# #         return out


# # class ResNet(ResNetBase):
# #     def __init__(
# #         self,
# #         resnet_size=8,
# #         scaling=4,
# #         save_activations=False,
# #         group_norm_num_groups=None,
# #         freeze_bn=False,
# #         freeze_bn_affine=False,
# #         num_classes=10
# #     ):
# #         super(ResNet, self).__init__()
# #         self.freeze_bn = freeze_bn
# #         self.freeze_bn_affine = freeze_bn_affine


# #         # define Model.
# #         if resnet_size % 6 != 2:
# #             raise ValueError("resnet_size must be 6n + 2:", resnet_size)
# #         block_nums = (resnet_size - 2) // 6
# #         block_fn = Bottleneck if resnet_size >= 44 else BasicBlock

# #         # decide the num of classes.
# #         self.num_classes = num_classes

# #         # define layers.
# #         assert int(16 * scaling) > 0
# #         self.inplanes = int(16 * scaling)
# #         self.conv1 = nn.Conv2d(
# #             in_channels=3,
# #             out_channels=(16 * scaling),
# #             kernel_size=3,
# #             stride=1,
# #             padding=1,
# #             bias=False,
# #         )
# #         self.bn1 = norm2d(group_norm_num_groups, planes=int(16 * scaling))
# #         self.relu = nn.ReLU(inplace=True)

# #         self.layer1 = self._make_block(
# #             block_fn=block_fn,
# #             planes=int(16 * scaling),
# #             block_num=block_nums,
# #             group_norm_num_groups=group_norm_num_groups,
# #         )
# #         self.layer2 = self._make_block(
# #             block_fn=block_fn,
# #             planes=int(32 * scaling),
# #             block_num=block_nums,
# #             stride=2,
# #             group_norm_num_groups=group_norm_num_groups,
# #         )
# #         self.layer3 = self._make_block(
# #             block_fn=block_fn,
# #             planes=int(64 * scaling),
# #             block_num=block_nums,
# #             stride=2,
# #             group_norm_num_groups=group_norm_num_groups,
# #         )

# #         self.avgpool = nn.AvgPool2d(kernel_size=8)
# #         self.classifier = nn.Linear(
# #             in_features=int(64 * scaling * block_fn.expansion),
# #             out_features=self.num_classes,
# #         )

# #         # weight initialization based on layer type.
# #         self._weight_initialization()

# #         # a placeholder for activations in the intermediate layers.
# #         self.save_activations = save_activations
# #         self.activations = None

# #     def forward(self, x):
# #         x = self.conv1(x)
# #         x = self.bn1(x)
# #         x = self.relu(x)

# #         x = self.layer1(x)
# #         activation1 = x
# #         x = self.layer2(x)
# #         activation2 = x
# #         x = self.layer3(x)
# #         activation3 = x
# #         x = self.avgpool(x)
# #         x = x.view(x.size(0), -1)

# #         feature = x

# #         y = self.classifier(x)

# #         if self.save_activations:
# #             self.activations = [activation1, activation2, activation3]

# #         return feature, y


# # if __name__ == '__main__':
# #     model = ResNet(resnet_size=8, scaling=4, save_activations=False,
# #                    group_norm_num_groups=None, freeze_bn=False, freeze_bn_affine=False, num_classes=10)


# # import math
# # import torch
# # import torch.nn as nn

# # def norm2d(group_norm_num_groups, planes):
# #     if group_norm_num_groups is not None and group_norm_num_groups > 0:
# #         return nn.GroupNorm(group_norm_num_groups, planes)
# #     else:
# #         return nn.BatchNorm2d(planes)

# # def conv3x3(in_planes, out_planes, stride=1):
# #     return nn.Conv2d(
# #         in_channels=in_planes,
# #         out_channels=out_planes,
# #         kernel_size=3,
# #         stride=stride,
# #         padding=1,
# #         bias=False,
# #     )

# # class BasicBlock(nn.Module):
# #     expansion = 1
# #     def __init__(self, in_planes, out_planes, stride=1, downsample=None, group_norm_num_groups=None):
# #         super(BasicBlock, self).__init__()
# #         self.conv1 = conv3x3(in_planes, out_planes, stride)
# #         self.bn1 = norm2d(group_norm_num_groups, planes=out_planes)
# #         self.relu = nn.ReLU(inplace=True)
# #         self.conv2 = conv3x3(out_planes, out_planes)
# #         self.bn2 = norm2d(group_norm_num_groups, planes=out_planes)
# #         self.downsample = downsample
# #         self.stride = stride

# #     def forward(self, x):
# #         residual = x
# #         out = self.conv1(x)
# #         out = self.bn1(out)
# #         out = self.relu(out)
# #         out = self.conv2(out)
# #         out = self.bn2(out)
# #         if self.downsample is not None:
# #             residual = self.downsample(x)
# #         out += residual
# #         out = self.relu(out)
# #         return out

# # class Bottleneck(nn.Module):
# #     expansion = 4
# #     def __init__(self, in_planes, out_planes, stride=1, downsample=None, group_norm_num_groups=None):
# #         super(Bottleneck, self).__init__()
# #         self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)
# #         self.bn1 = norm2d(group_norm_num_groups, planes=out_planes)
# #         self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
# #         self.bn2 = norm2d(group_norm_num_groups, planes=out_planes)
# #         self.conv3 = nn.Conv2d(out_planes, out_planes * 4, kernel_size=1, bias=False)
# #         self.bn3 = norm2d(group_norm_num_groups, planes=out_planes * 4)
# #         self.relu = nn.ReLU(inplace=True)
# #         self.downsample = downsample

# #     def forward(self, x):
# #         residual = x
# #         out = self.relu(self.bn1(self.conv1(x)))
# #         out = self.relu(self.bn2(self.conv2(out)))
# #         out = self.bn3(self.conv3(out))
# #         if self.downsample is not None:
# #             residual = self.downsample(x)
# #         out += residual
# #         return self.relu(out)

# # class ResNet(nn.Module):
# #     def __init__(self, resnet_size=8, scaling=4, save_activations=False, 
# #                  group_norm_num_groups=None, freeze_bn=False, freeze_bn_affine=False, num_classes=10):
# #         super(ResNet, self).__init__()
# #         self.freeze_bn = freeze_bn
# #         self.freeze_bn_affine = freeze_bn_affine
# #         self.num_classes = num_classes

# #         if resnet_size % 6 != 2:
# #             raise ValueError("resnet_size must be 6n + 2:", resnet_size)
# #         block_nums = (resnet_size - 2) // 6
# #         block_fn = Bottleneck if resnet_size >= 44 else BasicBlock

# #         self.inplanes = int(16 * scaling)
# #         self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
# #         self.bn1 = norm2d(group_norm_num_groups, planes=self.inplanes)
# #         self.relu = nn.ReLU(inplace=True)

# #         self.layer1 = self._make_block(block_fn, int(16 * scaling), block_nums, 1, group_norm_num_groups)
# #         self.layer2 = self._make_block(block_fn, int(32 * scaling), block_nums, 2, group_norm_num_groups)
# #         self.layer3 = self._make_block(block_fn, int(64 * scaling), block_nums, 2, group_norm_num_groups)

# #         # KRİTİK DEĞİŞİKLİK: Sabit kernel yerine AdaptiveAvgPool
# #         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
# #         self.classifier = nn.Linear(int(64 * scaling * block_fn.expansion), self.num_classes)
# #         self._weight_initialization()
# #         self.save_activations = save_activations

# #     def _make_block(self, block_fn, planes, block_num, stride=1, group_norm_num_groups=None):
# #         downsample = None
# #         if stride != 1 or self.inplanes != planes * block_fn.expansion:
# #             downsample = nn.Sequential(
# #                 nn.Conv2d(self.inplanes, planes * block_fn.expansion, kernel_size=1, stride=stride, bias=False),
# #                 norm2d(group_norm_num_groups, planes=planes * block_fn.expansion),
# #             )
# #         layers = [block_fn(self.inplanes, planes, stride, downsample, group_norm_num_groups)]
# #         self.inplanes = planes * block_fn.expansion
# #         for _ in range(1, block_num):
# #             layers.append(block_fn(self.inplanes, planes, 1, None, group_norm_num_groups))
# #         return nn.Sequential(*layers)

# #     def _weight_initialization(self):
# #         for m in self.modules():
# #             if isinstance(m, nn.Conv2d):
# #                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
# #             elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
# #                 nn.init.constant_(m.weight, 1)
# #                 nn.init.constant_(m.bias, 0)

# #     def forward(self, x):
# #         x = self.relu(self.bn1(self.conv1(x)))
# #         x = self.layer1(x)
# #         a1 = x
# #         x = self.layer2(x)
# #         a2 = x
# #         x = self.layer3(x)
# #         a3 = x
        
# #         x = self.avgpool(x)
# #         x = torch.flatten(x, 1)
# #         feature = x
# #         y = self.classifier(x)
        
# #         if self.save_activations:
# #             self.activations = [a1, a2, a3]
# #         return feature, y

# #     def train(self, mode=True):
# #         super(ResNet, self).train(mode)
# #         if self.freeze_bn:
# #             for m in self.modules():
# #                 if isinstance(m, nn.BatchNorm2d):
# #                     m.eval()
# #                     if self.freeze_bn_affine:
# #                         m.weight.requires_grad = False
# #                         m.bias.requires_grad = False
# import math
# import torch
# import torch.nn as nn

# def norm2d(group_norm_num_groups, planes):
#     # Eğer group_norm parametresi verildiyse ve 0'dan büyükse GroupNorm kullan
#     if group_norm_num_groups is not None and group_norm_num_groups > 0:
#         # Uyarı: planes (kanal sayısı), num_groups'a tam bölünmelidir!
#         return nn.GroupNorm(group_norm_num_groups, planes)
#     else:
#         # Aksi halde standart BatchNorm kullan
#         return nn.BatchNorm2d(planes)

# def conv3x3(in_planes, out_planes, stride=1):
#     return nn.Conv2d(
#         in_channels=in_planes,
#         out_channels=out_planes,
#         kernel_size=3,
#         stride=stride,
#         padding=1,
#         bias=False,
#     )

# class BasicBlock(nn.Module):
#     expansion = 1
#     def __init__(self, in_planes, out_planes, stride=1, downsample=None, group_norm_num_groups=None):
#         super(BasicBlock, self).__init__()
#         self.conv1 = conv3x3(in_planes, out_planes, stride)
#         self.bn1 = norm2d(group_norm_num_groups, planes=out_planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(out_planes, out_planes)
#         self.bn2 = norm2d(group_norm_num_groups, planes=out_planes)
#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x):
#         residual = x
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.conv2(out)
#         out = self.bn2(out)
#         if self.downsample is not None:
#             residual = self.downsample(x)
#         out += residual
#         out = self.relu(out)
#         return out

# class Bottleneck(nn.Module):
#     expansion = 4
#     def __init__(self, in_planes, out_planes, stride=1, downsample=None, group_norm_num_groups=None):
#         super(Bottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)
#         self.bn1 = norm2d(group_norm_num_groups, planes=out_planes)
#         self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn2 = norm2d(group_norm_num_groups, planes=out_planes)
#         self.conv3 = nn.Conv2d(out_planes, out_planes * 4, kernel_size=1, bias=False)
#         self.bn3 = norm2d(group_norm_num_groups, planes=out_planes * 4)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample

#     def forward(self, x):
#         residual = x
#         out = self.relu(self.bn1(self.conv1(x)))
#         out = self.relu(self.bn2(self.conv2(out)))
#         out = self.bn3(self.conv3(out))
#         if self.downsample is not None:
#             residual = self.downsample(x)
#         out += residual
#         return self.relu(out)

# class ResNet(nn.Module):
#     def __init__(self, resnet_size=8, scaling=4, save_activations=False, 
#                  group_norm_num_groups=16, freeze_bn=False, freeze_bn_affine=False, num_classes=7):
#         super(ResNet, self).__init__()
#         self.freeze_bn = freeze_bn
#         self.freeze_bn_affine = freeze_bn_affine
#         self.num_classes = num_classes

#         if resnet_size % 6 != 2:
#             raise ValueError("resnet_size must be 6n + 2:", resnet_size)
#         block_nums = (resnet_size - 2) // 6
#         block_fn = Bottleneck if resnet_size >= 44 else BasicBlock

#         self.inplanes = int(16 * scaling)
#         self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1 = norm2d(group_norm_num_groups, planes=self.inplanes)
#         self.relu = nn.ReLU(inplace=True)

#         self.layer1 = self._make_block(block_fn, int(16 * scaling), block_nums, 1, group_norm_num_groups)
#         self.layer2 = self._make_block(block_fn, int(32 * scaling), block_nums, 2, group_norm_num_groups)
#         self.layer3 = self._make_block(block_fn, int(64 * scaling), block_nums, 2, group_norm_num_groups)

#         # HAM10000 için esnek ve doğru pooling
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
#         self.classifier = nn.Linear(int(64 * scaling * block_fn.expansion), self.num_classes)
#         self._weight_initialization()
#         self.save_activations = save_activations

#     def _make_block(self, block_fn, planes, block_num, stride=1, group_norm_num_groups=None):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block_fn.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.inplanes, planes * block_fn.expansion, kernel_size=1, stride=stride, bias=False),
#                 norm2d(group_norm_num_groups, planes=planes * block_fn.expansion),
#             )
#         layers = [block_fn(self.inplanes, planes, stride, downsample, group_norm_num_groups)]
#         self.inplanes = planes * block_fn.expansion
#         for _ in range(1, block_num):
#             layers.append(block_fn(self.inplanes, planes, 1, None, group_norm_num_groups))
#         return nn.Sequential(*layers)

#     def _weight_initialization(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#     def forward(self, x):
#         x = self.relu(self.bn1(self.conv1(x)))
#         x = self.layer1(x)
#         a1 = x
#         x = self.layer2(x)
#         a2 = x
#         x = self.layer3(x)
#         a3 = x
        
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         feature = x
#         y = self.classifier(x)
        
#         if self.save_activations:
#             self.activations = [a1, a2, a3]
#         return feature, y

#     def train(self, mode=True):
#         super(ResNet, self).train(mode)
#         if self.freeze_bn:
#             for m in self.modules():
#                 if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
#                     m.eval()
#                     if self.freeze_bn_affine:
#                         m.weight.requires_grad = False
#                         m.bias.requires_grad = False


import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class ResNet(nn.Module):
    def __init__(self, num_classes=7, pretrained=True, freeze_bn=False):
        super(ResNet, self).__init__()
        
        # 1. Gerçek ve Pretrained ResNet18'i Yükle
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = resnet18(weights=weights)
        
        # 2. HAM10000 için Sınıflandırıcıyı Değiştir (1000 sınıf -> 7 sınıf)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, num_classes)
        
        self.freeze_bn = freeze_bn

    def forward(self, x):
        # FixMatch'in bekleidiği formatta çıktı veriyoruz: (Özellikler, Tahminler)
        # ResNet'in içinden özellikleri almak için küçük bir hile:
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        feature = torch.flatten(x, 1) # Özellik çıkarımı (Feature extraction)
        y = self.backbone.fc(feature) # Sınıflandırma (Classification)
        
        return feature, y

    def train(self, mode=True):
        super(ResNet, self).train(mode)
        if self.freeze_bn:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False
