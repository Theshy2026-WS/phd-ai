"""CoAtNet-0 via timm 1.0.25, adapted to native 64x64 Tiny ImageNet.

Third-party paper-derived implementation, NOT an official author port.
Keep the original C-C-T-T layout and stride-32 pyramid (16/8/4/2 stages).
Only input size, class count and attention precision differ from timm defaults.
"""
import torch
from torch import nn
import timm
from timm.models.maxxvit import Attention2d


class CoAtAttentionFP32(nn.Module):
    """Preserve timm attention math; protect QKV, softmax and AV under AMP."""
    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        # Explicit attention matches the other literature baselines' backend.
        self.inner.fused_attn = False

    def forward(self, x, shared_rel_pos=None):
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
            return self.inner(x, shared_rel_pos)


def build_coatnet():
    if timm.__version__ != '1.0.25':
        raise RuntimeError('CoAtNet baseline requires timm==1.0.25; install requirements-literature.txt')
    model = timm.create_model('coatnet_0_224', pretrained=False, img_size=64,
                              num_classes=200, drop_rate=0., drop_path_rate=0.)
    def protect(module):
        for name, child in list(module.named_children()):
            if isinstance(child, Attention2d):
                setattr(module, name, CoAtAttentionFP32(child))
            else:
                protect(child)
    protect(model)
    return model


def coatnet_description():
    return dict(name='coatnet0', input_size=[3, 64, 64], classes=200, pretrained=False,
                source='timm==1.0.25:coatnet_0_224 (third-party paper-derived)',
                adaptation=True, stem='timm two-convolution stem, stride 2',
                stage_types=['MBConv', 'MBConv', 'Transformer', 'Transformer'],
                stage_spatial_sizes=[16, 8, 4, 2], output_stride=32,
                depths=[2, 3, 5, 2], channels=[96, 192, 384, 768],
                attention_blocks=7, attention_heads=[12, 24], dim_head=32,
                attention_type='global spatial attention with relative position bias',
                attention_precision='fp32 island', attention_backend='explicit matmul',
                control=None, comparison_role='whole-model reference, not a paired ResNet ablation')
