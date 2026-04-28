import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class ResNet(nn.Module):
    def __init__(self, num_classes=7, pretrained=True, freeze_bn=False):
        super(ResNet, self).__init__()
        
        # 1. Gerçek ve Pretrained ResNet18'i Yükle
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = resnet18(weights=weights)
        
        # 2. Sınıflandırıcıyı Değiştir (1000 sınıf -> Bizim sınıf sayımız, örn: 7)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, num_classes)
        
        # 3. KRTİTİK: Dengesiz (Non-IID) Federe Öğrenme için BatchNorm'ları GroupNorm(16) yap
        self._convert_to_groupnorm(num_groups=16)

        self.freeze_bn = freeze_bn

    def _convert_to_groupnorm(self, num_groups=16):
        """
        Modeldeki tüm BatchNorm2d katmanlarını bulur ve GroupNorm ile değiştirir.
        Bu sayede istemcilerdeki sınıf dengesizliği modelin istatistiklerini zehirleyemez.
        """
        for name, module in self.backbone.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                num_channels = module.num_features
                gn = nn.GroupNorm(num_groups, num_channels)
                
                # Katmanı parent objenin içinde değiştir
                parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
                child_name = name.rsplit('.', 1)[-1]
                
                if parent_name == '':
                    setattr(self.backbone, child_name, gn)
                else:
                    # Parent objeyi bul
                    parent = self.backbone
                    for part in parent_name.split('.'):
                        parent = getattr(parent, part)
                    setattr(parent, child_name, gn)

    def forward(self, x):
        # FixMatch'in beklediği format: (Özellikler, Tahminler)
        # ResNet'in içinden özellikleri almak için manuel ileri besleme (forward pass)
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x) # İsmi bn1 kalsa da artık burası bir GroupNorm
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
        # GroupNorm kullandığımız için aslında dondurmaya gerek yok, 
        # ancak kodun geriye dönük uyumluluğu için bu fonksiyonu koruyoruz.
        if self.freeze_bn:
            for m in self.modules():
                if isinstance(m, nn.GroupNorm) or isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False
