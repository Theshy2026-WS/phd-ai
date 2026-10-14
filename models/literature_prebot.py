"""BoT control with only CNN stages 1--3 changed to full preactivation.

The stem, attention stage 4 and classifier remain exactly the original BoTNet
implementation, including initialization. This is a mixed pre/postactivation
experiment, NOT an all-preactivation BoTNet or a new paper reproduction.
"""
from copy import deepcopy
from torch import nn
from models.literature_alternet import PreBottleneck


def replace_cnn_with_preact(model):
    for name in ('layer1', 'layer2', 'layer3'):
        blocks = []
        for old in getattr(model, name):
            block = PreBottleneck(old.conv1.in_channels, old.conv1.out_channels,
                                  stride=old.conv2.stride[0])
            # Same random Conv weights as the control, not pretrained weights.
            # BN now operates before each conv; keep fresh default BN stats.
            block.conv1 = deepcopy(old.conv1)
            block.conv2[2] = deepcopy(old.conv2)
            block.conv3[2] = deepcopy(old.conv3)
            if old.downsample is not None:
                block.shortcut[0] = deepcopy(old.downsample[0])
            blocks.append(block)
        setattr(model, name, nn.Sequential(*blocks))
    return model


def prebot_description():
    return dict(name='botnet50_preact_cnn', input_size=[3, 64, 64], classes=200,
                pretrained=False, adaptation=True, control='botnet50',
                source='existing BoTNet equation implementation + PreBottleneck',
                stem='unchanged BoT stem: 3x3 stride2, BN/ReLU, no maxpool',
                depths=[3, 4, 6, 3], channels=[256, 512, 1024, 2048],
                stage_types=['PreAct CNN', 'PreAct CNN', 'PreAct CNN', 'postactivation BoT'],
                stage_spatial_sizes=[32, 16, 8, 4], output_stride=16,
                attention_blocks=3, attention_spatial_sizes=[8, 4, 4],
                attention_heads=4, attention_precision='fp32 island',
                initialization='same Conv weights, stem, stage4 and classifier as seed-matched botnet50; fresh preactivation BN',
                comparison_role='CNN-stage preactivation change versus botnet50; stage4 remains unchanged')
