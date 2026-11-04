"""A bounded 8->16->32 feature-fusion classifier, initialized as its CNN parent.

This is a mechanism screen, not a novelty claim. The encoder and original
phase-00 classifier remain trainable. A zero-initialized classification
correction preserves the parent's initial logits while the new branch learns.
"""
import torch
from torch import nn
from torch.nn import functional as F

from models.spatial_preact import build_spatial_model


FUSION_VARIANTS = ('cnn8_control', 'unet32', 'unet32_noise')


class SpatialFusionClassifier(nn.Module):
    def __init__(self, encoder, width=128, classes=200):
        super().__init__()
        if width < 8 or width % 8:
            raise ValueError('Fusion width must be a positive multiple of eight')
        self.encoder = encoder
        self.deep = nn.Conv2d(2048, width, 1, bias=False)
        self.lateral16 = nn.Conv2d(512, width, 1, bias=False)
        self.lateral32 = nn.Conv2d(256, width, 1, bias=False)
        self.refine16 = nn.Sequential(nn.Conv2d(width, width, 3, padding=1, bias=False),
                                      nn.GroupNorm(8, width), nn.ReLU())
        self.refine32 = nn.Sequential(nn.Conv2d(width, width, 3, padding=1, bias=False),
                                      nn.GroupNorm(8, width), nn.ReLU())
        self.correction = nn.Linear(width, classes)
        nn.init.zeros_(self.correction.weight)
        nn.init.zeros_(self.correction.bias)

    def forward(self, x):
        x = self.encoder.stem(x)
        shallow = self.encoder.stages[0](x)
        middle = self.encoder.stages[1](shallow)
        deep = self.encoder.stages[3](self.encoder.stages[2](middle))
        original = self.encoder.head(self.encoder.final_reduction(deep))
        decoded = F.interpolate(self.deep(deep), size=middle.shape[-2:],
                                mode='bilinear', align_corners=False)
        decoded = self.refine16(decoded + self.lateral16(middle))
        decoded = F.interpolate(decoded, size=shallow.shape[-2:],
                                mode='bilinear', align_corners=False)
        decoded = self.refine32(decoded + self.lateral32(shallow))
        correction = self.correction(decoded.mean(dim=(-2, -1)))
        return original + correction


def build_fusion_model(variant, parent_state, seed=42):
    if variant not in FUSION_VARIANTS:
        raise ValueError(f'Unknown fusion variant: {variant}')
    # Construction consumes no training/model RNG, and both U-shaped jobs have
    # exactly the same added weights, regardless of execution order.
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        encoder = build_spatial_model('cnn8', seed)
        encoder.load_state_dict(parent_state, strict=True)
        model = encoder if variant == 'cnn8_control' else SpatialFusionClassifier(encoder)
    return model.requires_grad_(True)


def fusion_description(variant):
    if variant not in FUSION_VARIANTS:
        raise ValueError(f'Unknown fusion variant: {variant}')
    return dict(variant=variant, encoder='trained cnn8 3/4/6/3',
        encoder_spatial_sizes=[32, 16, 8, 8], encoder_frozen=False,
        parent_classifier='unchanged phase00: BN-ReLU-GAP-linear',
        decoder_spatial_sizes=[] if variant == 'cnn8_control' else [8, 16, 32],
        decoder_width=0 if variant == 'cnn8_control' else 128,
        skip_channels=[] if variant == 'cnn8_control' else [512, 256],
        correction='none' if variant == 'cnn8_control' else 'zero-initialized additive class logits',
        interpretation='Warm-start mechanism screen; does not isolate skip connections from added capacity')
