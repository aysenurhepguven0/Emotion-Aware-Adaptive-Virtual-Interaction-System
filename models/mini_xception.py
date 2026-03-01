"""
models/mini_xception.py - Mini-Xception CNN Modeli
===================================================
Depthwise Separable Convolution tabanlı hafif CNN mimarisi.
Orijinal Xception'dan esinlenilmiş, FER2013 için optimize edilmiş.

Referans:
- Arriaga et al., "Real-time Convolutional Neural Networks for
  Emotion and Gender Classification" (2017)
- Chollet, "Xception: Deep Learning with Depthwise Separable
  Convolutions" (2017)

Mimari Özellikleri:
- Depthwise Separable Convolution: Standart conv'a göre ~8-9x daha az parametre
- Residual (Skip) Connections: Gradyan akışını iyileştirir
- Global Average Pooling: FC katmanlarını azaltır, overfitting'i önler
- Batch Normalization + Dropout: Regularization
- ~60K parametre: i5 + 4GB RAM'de rahat çalışır
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class SeparableConv2d(nn.Module):
    """
    Depthwise Separable Convolution bloğu.

    İki aşamadan oluşur:
    1. Depthwise Conv: Her kanal ayrı ayrı filtrelenir
    2. Pointwise Conv (1x1): Kanal bilgileri birleştirilir

    Bu yaklaşım standart convolution'a göre çok daha az parametre kullanır.
    Örnek: 3x3 conv, 64->128 kanal:
      - Standart: 64 x 128 x 3 x 3 = 73,728 parametre
      - Separable: (64 x 1 x 3 x 3) + (64 x 128 x 1 x 1) = 576 + 8,192 = 8,768 parametre
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(SeparableConv2d, self).__init__()

        # Depthwise: Her input kanalı ayrı filtre ile işlenir
        self.depthwise = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,  # groups=in_channels → depthwise
            bias=False
        )

        # Pointwise: 1x1 conv ile kanallar birleştirilir
        self.pointwise = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1,
            bias=False
        )

        # Batch Normalization: Eğitimi stabilize eder
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return x


class ResidualBlock(nn.Module):
    """
    Residual (artık) bağlantılı blok.

    Yapı:
        input → SepConv → ReLU → SepConv → + → ReLU → MaxPool → output
          |                                  ↑
          └──────── 1x1 Conv ────────────────┘  (skip connection)

    Skip connection sayesinde:
    - Gradyanlar daha kolay akabilir (vanishing gradient problemi azalır)
    - Ağ, identity mapping öğrenebilir
    """

    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        # Ana yol: 2 adet Separable Convolution
        self.sep_conv1 = SeparableConv2d(in_channels, out_channels)
        self.sep_conv2 = SeparableConv2d(out_channels, out_channels)

        # Max Pooling: Boyutu yarıya düşürür
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Skip connection: Kanal sayısı değişiyorsa 1x1 conv uygula
        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        # Ana yol
        residual = x
        x = F.relu(self.sep_conv1(x))
        x = self.sep_conv2(x)
        x = self.pool(x)

        # Skip connection
        residual = self.skip(residual)

        # Element-wise toplama
        x = x + residual
        x = F.relu(x)
        return x


class MiniXception(nn.Module):
    """
    Mini-Xception: Yüz ifadesi tanıma için hafif CNN modeli.

    Mimari:
        Input [1, 48, 48]
          ↓
        Conv2d (5x5, 8 filtre) → BN → ReLU          [8, 48, 48]
          ↓
        Conv2d (5x5, 8 filtre) → BN → ReLU          [8, 48, 48]
          ↓
        ResidualBlock (8 → 16)                        [16, 24, 24]
          ↓
        ResidualBlock (16 → 32)                       [32, 12, 12]
          ↓
        ResidualBlock (32 → 64)                       [64, 6, 6]
          ↓
        ResidualBlock (64 → 128)                      [128, 3, 3]
          ↓
        Conv2d (3x3, 256) → BN → ReLU               [256, 3, 3]
          ↓
        Global Average Pooling                        [256]
          ↓
        Dropout (0.5)                                 [256]
          ↓
        Fully Connected → 7 sınıf                    [7]

    Parametreler:
        num_classes (int): Çıkış sınıf sayısı (varsayılan: 7)
        in_channels (int): Giriş kanal sayısı (varsayılan: 1, gri tonlama)
    """

    def __init__(self, num_classes=6, in_channels=1):
        super(MiniXception, self).__init__()

        # ---- Giriş Katmanları ----
        # İlk convolution: Temel kenar ve doku özelliklerini çıkarır
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )

        # ---- Residual Bloklar ----
        # Her blok: boyutu yarıya düşürür, kanal sayısını artırır
        self.block1 = ResidualBlock(8, 16)     # 48x48 → 24x24
        self.block2 = ResidualBlock(16, 32)    # 24x24 → 12x12
        self.block3 = ResidualBlock(32, 64)    # 12x12 → 6x6
        self.block4 = ResidualBlock(64, 128)   # 6x6   → 3x3

        # ---- Son Katmanlar ----
        self.conv_final = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # Global Average Pooling: Her kanal için tek bir değer
        # Bu, FC katman sayısını azaltır ve overfitting'i önler
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # Dropout: Eğitimde rastgele nöronları kapatır
        self.dropout = nn.Dropout(p=0.5)

        # Sınıflandırma katmanı
        self.fc = nn.Linear(256, num_classes)

        # Ağırlıkları initialize et
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Kaiming (He) initialization uygular.
        ReLU aktivasyonlu ağlar için önerilir.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                       nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        İleri yönlü geçiş (forward pass).

        Parametreler:
            x (Tensor): [batch_size, 1, 48, 48] boyutunda giriş

        Returns:
            Tensor: [batch_size, num_classes] boyutunda logit çıkışı
        """
        # Giriş katmanları
        x = self.conv1(x)
        x = self.conv2(x)

        # Residual bloklar
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        # Son katmanlar
        x = self.conv_final(x)
        x = self.global_avg_pool(x)   # [batch, 256, 1, 1]
        x = x.view(x.size(0), -1)     # [batch, 256] - flatten
        x = self.dropout(x)
        x = self.fc(x)                # [batch, num_classes]

        return x

    def get_feature_vector(self, x):
        """
        Son FC katmanı öncesindeki özellik vektörünü döndürür.
        Transfer learning veya t-SNE görselleştirme için kullanılır.

        Returns:
            Tensor: [batch_size, 256] boyutunda özellik vektörü
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.conv_final(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        return x


def get_model(num_classes=None, in_channels=None, pretrained_path=None):
    """
    Model factory fonksiyonu.
    Farklı konfigürasyonlar veya pretrained model yüklemek için kullanılır.

    Parametreler:
        num_classes (int): Sınıf sayısı (varsayılan: config.NUM_CLASSES)
        in_channels (int): Giriş kanalı (varsayılan: config.NUM_CHANNELS)
        pretrained_path (str): Pretrained model yolu (opsiyonel)

    Returns:
        MiniXception: Model instance
    """
    if num_classes is None:
        num_classes = config.NUM_CLASSES
    if in_channels is None:
        in_channels = config.NUM_CHANNELS

    model = MiniXception(num_classes=num_classes, in_channels=in_channels)

    # Pretrained ağırlıkları yükle (varsa)
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"[INFO] Pretrained model yükleniyor: {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location=config.DEVICE)

        # checkpoint dict ise state_dict'i çıkar
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        print("[INFO] Pretrained model başarıyla yüklendi.")

    # Parametre sayısını yazdır
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[MODEL] Mini-Xception")
    print(f"  Toplam parametre:     {total_params:,}")
    print(f"  Eğitilebilir param.:  {trainable_params:,}")
    print(f"  Sınıf sayısı:        {num_classes}")
    print(f"  Giriş kanalı:        {in_channels}")

    return model
