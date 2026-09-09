"""Controlled RepViT experiment; no pretrained weights or distillation."""
import torch
from torch import nn
import timm


class AuditMixer(nn.Module):
    """Identical pre-norm/residual/FFN shell, different spatial mixer only."""
    def __init__(self, channels, variant):
        super().__init__()
        self.variant = variant
        self.position = nn.Conv2d(channels, channels, 3, padding=1,
                                  groups=channels, bias=False)
        self.norm1 = nn.LayerNorm(channels)
        if variant == 'attention':
            # Standard full-channel single-head MHSA, NOT SHViT's partial SHSA.
            self.mixer = nn.MultiheadAttention(channels, 1, batch_first=True)
        elif variant == 'conv':
            self.mixer = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
                nn.Conv2d(channels, channels * 2, 1), nn.GELU(),
                nn.Conv2d(channels * 2, channels, 1))
        else:
            raise ValueError(variant)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(nn.Linear(channels, 2 * channels), nn.GELU(),
                                 nn.Linear(2 * channels, channels))

    def forward(self, x):
        b, c, h, w = x.shape
        tokens = (x + self.position(x)).flatten(2).transpose(1, 2)
        z = self.norm1(tokens)
        if self.variant == 'attention':
            z = self.mixer(z, z, z, need_weights=False)[0]
        else:
            z = self.mixer(z.transpose(1, 2).reshape(b, c, h, w))
            z = z.flatten(2).transpose(1, 2)
        tokens = tokens + z
        tokens = tokens + self.ffn(self.norm2(tokens))
        return tokens.transpose(1, 2).reshape(b, c, h, w)


class AuditRepViT(nn.Module):
    def __init__(self, variant='cnn', seed=42, classes=200):
        super().__init__()
        if variant not in ('cnn', 'conv', 'attention'):
            raise ValueError(variant)
        self.variant = variant
        # Isolate trunk/head initialization from add-on construction and training RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.backbone = timm.create_model('repvit_m0_9', pretrained=False,
                                             num_classes=classes, distillation=False)
        channels = self.backbone.embed_dim[2]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 1000)
            self.addon = nn.Identity() if variant == 'cnn' else AuditMixer(channels, variant)
        # Initialize common addon parameters identically across B1/B2 too.
        if variant != 'cnn':
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed + 2000)
                for part in (self.addon.position, self.addon.norm1,
                             self.addon.norm2, self.addon.ffn):
                    for m in part.modules():
                        if hasattr(m, 'reset_parameters'):
                            m.reset_parameters()

    def forward(self, x):
        x = self.backbone.stem(x)
        for i, stage in enumerate(self.backbone.stages):
            x = stage(x)
            if i == 2:
                x = self.addon(x)
        return self.backbone.forward_head(x)
