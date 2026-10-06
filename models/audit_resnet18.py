"""Tiny ImageNet ResNet-18 pair: small-image stem, optional late self-attention."""
import torch
from torch import nn
from torchvision.models import resnet18


class LateSelfAttention(nn.Module):
    """Pre-LN spatial MHSA residual only: no extra FFN, conv, gate or dropout."""
    def __init__(self, channels=512, heads=4, residual_scale=0.1):
        super().__init__()
        if heads < 1 or channels % heads or residual_scale <= 0:
            raise ValueError('Invalid attention heads or residual scale')
        self.norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(channels, heads, batch_first=True, dropout=0.)
        self.residual_scale = float(residual_scale)

    def forward(self, x):
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        normalized = self.norm(tokens)
        correction = self.attention(normalized, normalized, normalized, need_weights=False)[0]
        return (tokens + self.residual_scale * correction).transpose(1, 2).reshape(b, c, h, w)


class AuditResNet18(nn.Module):
    def __init__(self, variant='cnn', seed=42, classes=200, heads=4, residual_scale=0.1):
        super().__init__()
        if variant not in ('cnn', 'attention'):
            raise ValueError('ResNet comparison supports cnn and attention only')
        self.variant = variant
        # The extra module never consumes randomness used to initialize the trunk/head.
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(seed)
            self.backbone = resnet18(weights=None, num_classes=classes)
            self.backbone.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
            nn.init.kaiming_normal_(self.backbone.conv1.weight, mode='fan_out', nonlinearity='relu')
            self.backbone.maxpool = nn.Identity()
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(seed + 1000)
            self.addon = (LateSelfAttention(512, heads, residual_scale)
                          if variant == 'attention' else nn.Identity())

    def forward(self, x):
        b = self.backbone
        x = b.relu(b.bn1(b.conv1(x)))
        x = b.maxpool(x)
        x = b.layer4(b.layer3(b.layer2(b.layer1(x))))
        x = self.addon(x)  # [B, 512, 8, 8] for native 64x64 images.
        return b.fc(b.avgpool(x).flatten(1))
