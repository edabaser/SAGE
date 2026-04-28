"""
ResNet-18 — Federe Öğrenme için Düzenlenmiş Versiyon

Değişiklikler:
1. Tüm BatchNorm2d katmanları GroupNorm(16) ile değiştirildi.
   Neden: Non-IID dağılımda BatchNorm, her client'ın farklı istatistiklerini
   global modele aktarır ve "istatistik zehirlenmesi" yaşanır.
   GroupNorm ise batch boyutundan bağımsız çalışır.

2. Pretrained ImageNet ağırlıkları yükleniyor (transfer learning).
   Neden: 700 etiketli görüntü sıfırdan öğrenme için yetersiz.
   ImageNet özellikleri iyi bir başlangıç noktası sağlar.

3. forward() → (feature, logits) tuple döner — SAGE/FixMatch uyumlu.

4. BN→GN dönüşümü: ağırlıklar (weight/bias) korunuyor,
   running_mean/running_var parametreleri drop ediliyor (GN'de yok).
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNet(nn.Module):
    def __init__(self, num_classes=7, pretrained=True, freeze_bn=False):
        super(ResNet, self).__init__()

        # 1. Pretrained ResNet-18 yükle
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = resnet18(weights=weights)

        # 2. Son katmanı bizim görevimize göre değiştir (1000 → num_classes)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, num_classes)

        # 3. BatchNorm → GroupNorm dönüşümü
        #    Bu aşama pretrained ağırlıkları KORUR (gamma/beta aktarılır)
        self._convert_bn_to_gn(num_groups=16)

        self.freeze_bn = freeze_bn

    def _convert_bn_to_gn(self, num_groups=16):
        """
        Modeldeki tüm BatchNorm2d'leri GroupNorm ile değiştirir.

        Önemli: GroupNorm'un kanal sayısı num_groups'a tam bölünmeli.
        ResNet-18 kanal sayıları: 64, 128, 256, 512 → hepsi 16'ya bölünür.

        Pretrained gamma (weight) ve beta (bias) değerleri aktarılır.
        running_mean / running_var GroupNorm'da yoktur, taşınmaz.
        """
        def replace_bn_in_module(module, prefix=''):
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name

                if isinstance(child, nn.BatchNorm2d):
                    num_channels = child.num_features

                    # Güvenlik: kanal sayısı num_groups'a bölünemiyor mu?
                    actual_groups = num_groups
                    if num_channels % actual_groups != 0:
                        # En yakın bölen bul
                        for g in [8, 4, 2, 1]:
                            if num_channels % g == 0:
                                actual_groups = g
                                break

                    gn = nn.GroupNorm(actual_groups, num_channels,
                                      affine=True, eps=1e-5)

                    # Pretrained affine parametreleri aktar
                    if child.weight is not None:
                        with torch.no_grad():
                            gn.weight.copy_(child.weight)
                    if child.bias is not None:
                        with torch.no_grad():
                            gn.bias.copy_(child.bias)

                    setattr(module, name, gn)

                else:
                    # Recursive: alt modüllere de gir
                    replace_bn_in_module(child, full_name)

        replace_bn_in_module(self.backbone)

    def forward(self, x):
        """
        SAGE/FixMatch uyumlu forward.
        Dönüş: (feature_vector, logits) — her ikisi de tensor
        """
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)    # İsmi bn1 ama artık GroupNorm
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        feature = torch.flatten(x, 1)   # [B, 512]
        logits  = self.backbone.fc(feature)  # [B, num_classes]

        return feature, logits

    def train(self, mode=True):
        super(ResNet, self).train(mode)
        # GroupNorm kullandığımız için istatistik dondurma gerekmez.
        # Geriye dönük uyumluluk için bu blok korunuyor.
        if self.freeze_bn:
            for m in self.modules():
                if isinstance(m, (nn.GroupNorm, nn.BatchNorm2d)):
                    m.eval()
                    if m.weight is not None:
                        m.weight.requires_grad = False
                    if m.bias is not None:
                        m.bias.requires_grad = False
