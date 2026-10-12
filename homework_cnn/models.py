# AI assistance: OpenAI ChatGPT (GPT-6).
# Used to assist with code development, debugging, and explanations.
# https://chatgpt.com/
import torch
import torch.nn as nn
from torchvision import models


# --------------------------------------------------
# 1. Regular CNN: binary cat/dog classification
# --------------------------------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 128 -> 64

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64 -> 32

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32 -> 16
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x


# --------------------------------------------------
# 2. ResNet-18: binary cat/dog classification
# --------------------------------------------------
class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        # weights=None means training from scratch.
        self.network = models.resnet18(weights=None)

        # Replace the original 1000-class output layer.
        in_features = self.network.fc.in_features
        self.network.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.network(x)


# --------------------------------------------------
# 3. U-Net: pixel-level segmentation
# --------------------------------------------------
class DoubleConv(nn.Module):
    """Two convolutional layers with ReLU activations."""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=3, padding=1
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels, out_channels,
                kernel_size=3, padding=1
            ),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        # Encoder: extract features while reducing spatial dimensions.
        self.enc1 = DoubleConv(3, 32)
        self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck: deepest feature representation.
        self.bottleneck = DoubleConv(128, 256)

        # Decoder: increase spatial dimensions and combine skip connections.
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(64, 32)

        # One output channel per segmentation class.
        self.output = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)               # [B, 32, 128, 128]
        e2 = self.enc2(self.pool(e1))   # [B, 64, 64, 64]
        e3 = self.enc3(self.pool(e2))   # [B, 128, 32, 32]

        # Bottleneck
        b = self.bottleneck(self.pool(e3))  # [B, 256, 16, 16]

        # Decoder + skip connections
        d3 = self.up3(b)                    # [B, 128, 32, 32]
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)                   # [B, 64, 64, 64]
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)                   # [B, 32, 128, 128]
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.output(d1)
