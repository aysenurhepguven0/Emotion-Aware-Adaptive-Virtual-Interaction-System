"""
models/mini_xception.py - Mini-Xception CNN Model
===================================================
Lightweight CNN architecture based on Depthwise Separable Convolutions.
Inspired by the original Xception, optimized for FER2013.

Reference:
- Arriaga et al., "Real-time Convolutional Neural Networks for
  Emotion and Gender Classification" (2017)
- Chollet, "Xception: Deep Learning with Depthwise Separable
  Convolutions" (2017)

Architecture Features:
- Depthwise Separable Convolution: ~8-9x fewer parameters than standard conv
- Residual (Skip) Connections: Improved gradient flow
- Global Average Pooling: Reduces FC layers, prevents overfitting
- Batch Normalization + Dropout: Regularization
- ~60K parameters: Runs comfortably on i5 + 4GB RAM
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
    Depthwise Separable Convolution block.

    Consists of two stages:
    1. Depthwise Conv: Each channel is filtered separately
    2. Pointwise Conv (1x1): Channel information is combined

    This approach uses significantly fewer parameters than standard conv.
    Example: 3x3 conv, 64->128 channels:
      - Standard: 64 x 128 x 3 x 3 = 73,728 parameters
      - Separable: (64 x 1 x 3 x 3) + (64 x 128 x 1 x 1) = 8,768 parameters
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(SeparableConv2d, self).__init__()

        # Depthwise: Each input channel is processed with its own filter
        self.depthwise = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,  # groups=in_channels -> depthwise
            bias=False
        )

        # Pointwise: 1x1 conv combines channel information
        self.pointwise = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1,
            bias=False
        )

        # Batch Normalization: Stabilizes training
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return x


class ResidualBlock(nn.Module):
    """
    Residual (skip) connection block.

    Structure:
        input -> SepConv -> ReLU -> SepConv -> + -> ReLU -> MaxPool -> output
          |                                    ^
          +---------- 1x1 Conv ----------------+  (skip connection)

    Benefits of skip connections:
    - Gradients flow more easily (mitigates vanishing gradient problem)
    - Network can learn identity mappings
    """

    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        # Main path: 2 Separable Convolutions
        self.sep_conv1 = SeparableConv2d(in_channels, out_channels)
        self.sep_conv2 = SeparableConv2d(out_channels, out_channels)

        # Max Pooling: Halves spatial dimensions
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Skip connection: Apply 1x1 conv if channel count changes
        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        # Main path
        residual = x
        x = F.relu(self.sep_conv1(x))
        x = self.sep_conv2(x)
        x = self.pool(x)

        # Skip connection
        residual = self.skip(residual)

        # Element-wise addition
        x = x + residual
        x = F.relu(x)
        return x


class MiniXception(nn.Module):
    """
    Mini-Xception: Lightweight CNN model for facial expression recognition.

    Architecture:
        Input [1, 48, 48]
          |
        Conv2d (5x5, 8 filters) -> BN -> ReLU          [8, 48, 48]
          |
        Conv2d (5x5, 8 filters) -> BN -> ReLU          [8, 48, 48]
          |
        ResidualBlock (8 -> 16)                          [16, 24, 24]
          |
        ResidualBlock (16 -> 32)                         [32, 12, 12]
          |
        ResidualBlock (32 -> 64)                         [64, 6, 6]
          |
        ResidualBlock (64 -> 128)                        [128, 3, 3]
          |
        Conv2d (3x3, 256) -> BN -> ReLU                 [256, 3, 3]
          |
        Global Average Pooling                            [256]
          |
        Dropout (0.5)                                     [256]
          |
        Fully Connected -> 6 classes                      [6]

    Args:
        num_classes (int): Number of output classes (default: 6)
        in_channels (int): Number of input channels (default: 1, grayscale)
    """

    def __init__(self, num_classes=6, in_channels=1):
        super(MiniXception, self).__init__()

        # ---- Input Layers ----
        # First convolution: Extracts basic edge and texture features
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

        # ---- Residual Blocks ----
        # Each block: halves spatial size, increases channel count
        self.block1 = ResidualBlock(8, 16)     # 48x48 -> 24x24
        self.block2 = ResidualBlock(16, 32)    # 24x24 -> 12x12
        self.block3 = ResidualBlock(32, 64)    # 12x12 -> 6x6
        self.block4 = ResidualBlock(64, 128)   # 6x6   -> 3x3

        # ---- Final Layers ----
        self.conv_final = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # Global Average Pooling: Single value per channel
        # Reduces FC layer count and prevents overfitting
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # Dropout: Randomly disables neurons during training
        self.dropout = nn.Dropout(p=0.5)

        # Classification layer
        self.fc = nn.Linear(256, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Applies Kaiming (He) initialization.
        Recommended for networks with ReLU activation.
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
        Forward pass.

        Args:
            x (Tensor): Input of shape [batch_size, 1, 48, 48]

        Returns:
            Tensor: Logit output of shape [batch_size, num_classes]
        """
        # Input layers
        x = self.conv1(x)
        x = self.conv2(x)

        # Residual blocks
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        # Final layers
        x = self.conv_final(x)
        x = self.global_avg_pool(x)   # [batch, 256, 1, 1]
        x = x.view(x.size(0), -1)     # [batch, 256] - flatten
        x = self.dropout(x)
        x = self.fc(x)                # [batch, num_classes]

        return x

    def get_feature_vector(self, x):
        """
        Returns the feature vector before the final FC layer.
        Useful for transfer learning or t-SNE visualization.

        Returns:
            Tensor: Feature vector of shape [batch_size, 256]
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
    Model factory function.
    Used for creating models with different configurations or loading
    pretrained weights.

    Args:
        num_classes (int): Number of classes (default: config.NUM_CLASSES)
        in_channels (int): Input channels (default: config.NUM_CHANNELS)
        pretrained_path (str): Path to pretrained model (optional)

    Returns:
        MiniXception: Model instance
    """
    if num_classes is None:
        num_classes = config.NUM_CLASSES
    if in_channels is None:
        in_channels = config.NUM_CHANNELS

    model = MiniXception(num_classes=num_classes, in_channels=in_channels)

    # Load pretrained weights if available
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"[INFO] Loading pretrained model: {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location=config.DEVICE)

        # Extract state_dict if checkpoint is a dict
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        print("[INFO] Pretrained model loaded successfully.")

    # Print parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"\n[MODEL] Mini-Xception")
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Number of classes:    {num_classes}")
    print(f"  Input channels:       {in_channels}")

    return model
