"""timm 1.0.25 structure baselines adapted explicitly to native Tiny ImageNet.

No pretrained weights, no FCMAE pretraining, and no claim to reproduce paper
scores. Original block depths/widths are retained; only entrance sampling and
class count change. CAFormer attention uses the project's FP32 policy.
"""
import torch
from torch import nn
import timm
from timm.layers import trunc_normal_
from timm.models.convmixer import ConvMixer
from timm.models.metaformer import Attention


MIXER_VARIANTS = ('convmixer768_tiny64', 'convnextv2_tiny64',
                  'convformer_s18_tiny64', 'caformer_s18_tiny64')


class MetaFormerAttentionFP32(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.inner.fused_attn = False

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            return self.inner(x.float() if x.dtype in (torch.float16, torch.bfloat16) else x)


def protect_attention(module):
    for name, child in list(module.named_children()):
        if isinstance(child, Attention):
            setattr(module, name, MetaFormerAttentionFP32(child))
        else:
            protect_attention(child)


def build_mixer(variant):
    if timm.__version__ != '1.0.25':
        raise RuntimeError('Mixer baselines require timm==1.0.25; install requirements-literature.txt')
    if variant == 'convmixer768_tiny64':
        # Author's released 768/32 variant uses ReLU. Patch7 -> patch4 tiles
        # all 64 pixels and maintains a constant 16x16 grid throughout.
        return ConvMixer(dim=768, depth=32, kernel_size=7, patch_size=4,
                         act_layer=nn.ReLU, num_classes=200, drop_rate=0.)
    names = dict(convnextv2_tiny64='convnextv2_tiny',
                 convformer_s18_tiny64='convformer_s18', caformer_s18_tiny64='caformer_s18')
    if variant not in names:
        raise ValueError(f'Unknown mixer variant: {variant}')
    model = timm.create_model(names[variant], pretrained=False, num_classes=200,
                              drop_rate=0., drop_path_rate=0.)
    if variant == 'convnextv2_tiny64':
        # Construct native model first, keeping stage0 stride1. Passing patch_size2
        # to timm's constructor would add a stage0 downsample, defeating this.
        original = model.stem[0]
        model.stem[0] = nn.Conv2d(3, original.out_channels, 2, stride=2,
                                  bias=original.bias is not None)
        trunc_normal_(model.stem[0].weight, std=.02)
        if model.stem[0].bias is not None:
            nn.init.zeros_(model.stem[0].bias)
    else:
        # Retain 7x7 overlapping stem; stride4/pad2 -> stride2/pad3.
        model.stem.conv.stride = (2, 2)
        model.stem.conv.padding = (3, 3)
        if variant == 'caformer_s18_tiny64':
            protect_attention(model)
    for i, info in enumerate(model.feature_info):
        info['reduction'] = 2**(i+1)
    return model


def mixer_description(variant):
    mixer = variant == 'convmixer768_tiny64'
    v2 = variant == 'convnextv2_tiny64'
    ca = variant == 'caformer_s18_tiny64'
    return dict(name=variant, input_size=[3, 64, 64], classes=200,
                pretrained=False, pretraining='none; supervised from scratch, no FCMAE',
                adaptation=True, source='timm==1.0.25; author-derived implementations',
                comparison_role='Tiny-adapted structure baseline under common recipe, not paper-score reproduction',
                control='convformer_s18_tiny64' if ca else None,
                stem=('4x4 nonoverlapping patch stride4; author variant patch7' if mixer else
                      '2x2 nonoverlapping patch stride2; original 4x4 stride4' if v2 else
                      '7x7 stride2 padding3; original stride4 padding2'),
                depths=[32] if mixer else [3, 3, 9, 3],
                channels=[768] if mixer else [96, 192, 384, 768] if v2 else [64, 128, 320, 512],
                stage_spatial_sizes=[16] if mixer else [32, 16, 8, 4],
                stage_types=(['depthwise7x7 + pointwise1x1, ReLU/BN'] if mixer else
                             ['ConvNeXt V2 with GRN']*4 if v2 else
                             ['SepConv', 'SepConv', 'self-attention', 'self-attention'] if ca else ['SepConv']*4),
                attention_blocks=12 if ca else 0,
                attention_spatial_sizes=[8]*9+[4]*3 if ca else [],
                attention_heads=[10, 16] if ca else [],
                attention_precision='fp32 island' if ca else 'outer autocast',
                attention_backend='explicit matmul' if ca else 'none',
                output_stride=4 if mixer else 16)
