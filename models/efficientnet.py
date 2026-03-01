"""
models/efficientnet.py - EfficientNet-B0 Modeli (Transfer Learning)
====================================================================
ImageNet pretrained EfficientNet-B0 kullanarak yüz ifadesi tanıma.

- Pretrained ağırlıklar ile Transfer Learning
- Grayscale (1 kanal) → RGB (3 kanal) dönüşümü otomatik
- Son katman 6 sınıfa uyarlanmış (6 Ekman)
- ~5.3M parametre (4M frozen + 1.3M trainable)

Referans:
    Tan, M. & Le, Q.V. (2019). EfficientNet: Rethinking Model Scaling
"""

import torch
import torch.nn as nn
import torchvision.models as models

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class EfficientNetB0(nn.Module):
    """
    EfficientNet-B0 tabanlı duygu tanıma modeli.

    Transfer Learning stratejisi:
        1. ImageNet pretrained EfficientNet-B0 yüklenir
        2. İlk katmanlar dondurulur (frozen) — genel özellikler korunur
        3. Son katmanlar eğitilir (fine-tuning) — duygu tanımaya özelleşir
        4. Classifier katmanı 6 sınıfa uyarlanır

    Giriş: [batch, 1, 48, 48] (grayscale) → otomatik [batch, 3, 48, 48] (RGB)
    Çıkış: [batch, 6] (6 duygu sınıfı logit)
    """

    def __init__(self, num_classes=6, in_channels=1, freeze_backbone=True, unfreeze_last_n=2):
        """
        Parametreler:
            num_classes (int): Sınıf sayısı (varsayılan: 6)
            in_channels (int): Giriş kanalı (1=grayscale, 3=RGB)
            freeze_backbone (bool): Backbone'u dondur (Transfer Learning)
            unfreeze_last_n (int): Son kaç bloğu çöz (fine-tuning)
        """
        super(EfficientNetB0, self).__init__()

        self.in_channels = in_channels

        # Grayscale → RGB dönüşüm katmanı
        if in_channels == 1:
            self.channel_adapter = nn.Sequential(
                nn.Conv2d(1, 3, kernel_size=1, bias=False),
                nn.BatchNorm2d(3)
            )
        else:
            self.channel_adapter = None

        # ImageNet pretrained EfficientNet-B0 yükle
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        self.backbone = models.efficientnet_b0(weights=weights)

        # Orijinal classifier'ı kaldır
        in_features = self.backbone.classifier[1].in_features  # 1280

        # Yeni classifier: Dropout → FC → ReLU → Dropout → FC
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes)
        )

        # Backbone'u dondur (opsiyonel)
        if freeze_backbone:
            self._freeze_backbone(unfreeze_last_n)

    def _freeze_backbone(self, unfreeze_last_n=2):
        """
        Backbone katmanlarını dondur, son N bloğu çöz.

        EfficientNet-B0 features yapısı:
            - features[0]: Stem (Conv + BN)
            - features[1-8]: MBConv blokları
            - features[8]: Head (Conv + BN)
        """
        # Önce hepsini dondur
        for param in self.backbone.features.parameters():
            param.requires_grad = False

        # Son N bloğu çöz (fine-tuning)
        total_blocks = len(self.backbone.features)
        for i in range(max(0, total_blocks - unfreeze_last_n), total_blocks):
            for param in self.backbone.features[i].parameters():
                param.requires_grad = True

        # Classifier her zaman eğitilebilir
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True

    def unfreeze_all(self):
        """Tüm katmanları eğitilebilir yap (tam fine-tuning)."""
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x):
        """
        İleri yönlü geçiş.

        Parametreler:
            x (Tensor): [batch, 1, 48, 48] veya [batch, 3, 48, 48]

        Returns:
            Tensor: [batch, num_classes] logit çıkışı
        """
        # Grayscale → RGB dönüşümü
        if self.channel_adapter is not None:
            x = self.channel_adapter(x)

        return self.backbone(x)

    def get_feature_vector(self, x):
        """
        Son FC katmanı öncesindeki özellik vektörünü döndürür.
        Transfer learning veya t-SNE görselleştirme için.

        Returns:
            Tensor: [batch, 256] boyutunda özellik vektörü
        """
        if self.channel_adapter is not None:
            x = self.channel_adapter(x)

        # Features kısmından geçir
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)

        # Classifier'ın ilk 3 katmanından geçir (Dropout → Linear → ReLU)
        for i in range(3):
            x = self.backbone.classifier[i](x)

        return x


def get_efficientnet_model(num_classes=None, in_channels=None, pretrained_path=None,
                           freeze_backbone=True, unfreeze_last_n=2):
    """
    EfficientNet-B0 model factory fonksiyonu.

    Parametreler:
        num_classes (int): Sınıf sayısı
        in_channels (int): Giriş kanalı
        pretrained_path (str): Daha önce eğitilmiş model yolu
        freeze_backbone (bool): Backbone'u dondur
        unfreeze_last_n (int): Son kaç bloğu çöz

    Returns:
        EfficientNetB0: Model instance
    """
    if num_classes is None:
        num_classes = config.NUM_CLASSES
    if in_channels is None:
        in_channels = config.NUM_CHANNELS

    model = EfficientNetB0(
        num_classes=num_classes,
        in_channels=in_channels,
        freeze_backbone=freeze_backbone,
        unfreeze_last_n=unfreeze_last_n
    )

    # Daha önce eğitilmiş model varsa yükle
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"[INFO] EfficientNet model yükleniyor: {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location=config.DEVICE)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        print("[INFO] EfficientNet model başarıyla yüklendi.")

    # Parametre sayısını yazdır
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"\n[MODEL] EfficientNet-B0 (Transfer Learning)")
    print(f"  Toplam parametre:     {total_params:,}")
    print(f"  Eğitilebilir param.:  {trainable_params:,}")
    print(f"  Dondurulan param.:    {frozen_params:,}")
    print(f"  Sınıf sayısı:        {num_classes}")
    print(f"  Giriş kanalı:        {in_channels}")
    print(f"  Backbone frozen:     {freeze_backbone}")

    return model
