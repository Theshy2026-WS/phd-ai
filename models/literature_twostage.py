"""Explicit Tiny ImageNet architecture experiments, NOT paper reproductions."""
from dataclasses import replace
from functools import partial

import torch
from torch import nn
from torchvision.models.resnet import Bottleneck
import timm
from timm.models.maxxvit import MaxxVit, MaxxVitStage, Attention2d, model_cfgs
from timm.models._manipulate import named_apply

from models.literature_alternet import PreBottleneck, BNGAPBlock
from models.literature_coatnet import CoAtAttentionFP32


class PreActResNet50(nn.Module):
    """Full preactivation, 3/4/6/3 bottlenecks; native Tiny stride-16 stem."""
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False)
        stages, incoming = [], 64
        for i, (width, depth) in enumerate(zip((64, 128, 256, 512), (3, 4, 6, 3))):
            blocks = []
            for j in range(depth):
                blocks.append(PreBottleneck(incoming, width, stride=2 if i and j == 0 else 1))
                incoming = width * 4
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.Sequential(*stages)
        self.head = BNGAPBlock(2048, 200)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        return self.head(self.stages(self.stem(x)))


class BoTTwoStage(nn.Module):
    """C3 at 32x32 -> average pool -> B3 at 16x16, all attention before no further pooling."""
    def __init__(self):
        super().__init__()
        from models.literature_resnet import BoTAttention
        self.stem = nn.Sequential(nn.Conv2d(3, 64, 3, 2, 1, bias=False),
                                  nn.BatchNorm2d(64), nn.ReLU())
        stages, incoming = [], 64
        for width, attention in ((64, False), (128, True)):
            blocks = []
            for _ in range(3):
                out = width * 4
                shortcut = (nn.Sequential(nn.Conv2d(incoming, out, 1, bias=False), nn.BatchNorm2d(out))
                            if incoming != out else None)
                block = Bottleneck(incoming, width, downsample=shortcut)
                if attention:
                    block.conv2 = BoTAttention(width, size=16, heads=4)
                blocks.append(block)
                incoming = out
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.ModuleList(stages)
        self.transition = nn.AvgPool2d(2)
        self.head = nn.Linear(512, 200)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.stages[0](self.stem(x))
        x = self.stages[1](self.transition(x))
        return self.head(x.mean((2, 3)))


class CoAtTwoStage(MaxxVit):
    """Reuse timm CoAtNet blocks: C2 stride1 -> T2 stride2 after stride2 stem."""
    def __init__(self):
        if timm.__version__ != '1.0.25':
            raise RuntimeError('Requires timm==1.0.25')
        cfg = replace(model_cfgs['coatnet_0'], embed_dim=(96, 192), depths=(2, 2),
                      block_type=('C', 'T'), stem_width=96, head_hidden_size=192)
        super().__init__(cfg, img_size=64, num_classes=200, drop_rate=0., drop_path_rate=0.)
        # Build with explicit strides so attention really receives 16x16, not 8x8.
        self.stages = nn.Sequential(*[
            MaxxVitStage(cin, cout, stride=stride, depth=2, feat_size=(size, size),
                         block_types=kind, conv_cfg=cfg.conv_cfg,
                         transformer_cfg=cfg.transformer_cfg, drop_path=[0., 0.])
            for cin, cout, stride, size, kind in ((96, 96, 1, 32, 'C'), (96, 192, 2, 16, 'T'))
        ])
        named_apply(partial(self._init_weights, scheme=cfg.weight_init), self.stages)
        self.feature_info = [dict(num_chs=96, reduction=2, module='stem'),
                             dict(num_chs=96, reduction=2, module='stages.0'),
                             dict(num_chs=192, reduction=4, module='stages.1')]
        def protect(module):
            for name, child in list(module.named_children()):
                if isinstance(child, Attention2d):
                    setattr(module, name, CoAtAttentionFP32(child))
                else:
                    protect(child)
        protect(self)


EXPERIMENT_VARIANTS = ('resnet50_preact', 'botnet_twostage', 'coatnet_twostage')


def build_twostage(variant):
    return dict(resnet50_preact=PreActResNet50, botnet_twostage=BoTTwoStage,
                coatnet_twostage=CoAtTwoStage)[variant]()


def experiment_description(variant):
    pre = variant == 'resnet50_preact'
    bot = variant == 'botnet_twostage'
    return dict(name=variant, input_size=[3, 64, 64], classes=200, pretrained=False,
                adaptation=True, comparison_role='architecture experiment; not a paper reproduction or size-matched ablation',
                source='existing PreBottleneck / BoT equations / timm 1.0.25 CoAtNet blocks',
                depths=[3, 4, 6, 3] if pre else [3, 3] if bot else [2, 2],
                stage_types=['preactivation CNN']*4 if pre else ['CNN', 'BoT'] if bot else ['MBConv', 'Transformer'],
                stage_spatial_sizes=[32, 16, 8, 4] if pre else [32, 16],
                channels=[256, 512, 1024, 2048] if pre else [256, 512] if bot else [96, 192],
                attention_blocks=0 if pre else 3 if bot else 2,
                attention_spatial_sizes=[] if pre else [16]* (3 if bot else 2),
                attention_precision='outer autocast' if pre else 'fp32 island',
                output_stride=16 if pre else 4, control=None)
