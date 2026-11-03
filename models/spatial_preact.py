"""Matched PreAct CNNs for a final-stage downsampling-timing diagnostic.

This is an architecture control, not a novel method. Both variants retain the
existing native64 3/4/6/3 model's parameter shapes, initialization and head.
``cnn8`` delays even-index spatial reduction until after the final stage. Its
larger intermediate maps change compute, receptive fields and training-time
BatchNorm statistics; improved accuracy alone would not isolate one cause.
"""
import torch
from torch import nn

from models.literature_twostage import PreActResNet50


SPATIAL_VARIANTS = ('cnn4', 'cnn8')


class EvenIndexDecimation(nn.Module):
    """Parameter-free stride-two point sampling, with no averaging/interpolation."""
    def forward(self, x):
        return x[..., ::2, ::2]


class SpatialPreActResNet50(PreActResNet50):
    def __init__(self, variant):
        if variant not in SPATIAL_VARIANTS:
            raise ValueError(f'Unknown spatial variant: {variant}')
        super().__init__()
        if variant == 'cnn8':
            transition = self.stages[3][0]
            spatial_conv, projection = transition.conv2[2], transition.shortcut[0]
            if spatial_conv.stride != (2, 2) or projection.stride != (2, 2):
                raise RuntimeError('Parent final-stage stride layout has changed')
            spatial_conv.stride = (1, 1)
            projection.stride = (1, 1)
        self.final_reduction = EvenIndexDecimation() if variant == 'cnn8' else nn.Identity()

    def forward(self, x):
        features = self.stages(self.stem(x))
        return self.head(self.final_reduction(features))


def build_spatial_model(variant, seed=42):
    """Use the original factory's isolated CPU RNG policy for paired weights."""
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        return SpatialPreActResNet50(variant)


def spatial_description(variant):
    if variant not in SPATIAL_VARIANTS:
        raise ValueError(f'Unknown spatial variant: {variant}')
    delayed = variant == 'cnn8'
    return dict(
        name=variant, parent_variant='resnet50_preact', input_size=[3, 64, 64],
        classes=200, pretrained=False, depths=[3, 4, 6, 3],
        channels=[256, 512, 1024, 2048],
        stage_types=['preactivation CNN'] * 4,
        stage_spatial_sizes=[32, 16, 8, 8 if delayed else 4],
        classifier_spatial_size=[4, 4], classifier_input_features=2048,
        stage4_first_block_stride=1 if delayed else 2,
        stage4_output_stride=8 if delayed else 16,
        output_stride_before_head=16, attention_blocks=0,
        final_reduction=('even-index point decimation after stage4' if delayed
                         else 'identity; original stride-two stage4 transition'),
        matched_weight_shapes=True,
        intervention=('move final-stage reduction after all three final-stage blocks'
                      if delayed else 'unchanged four-stage CNN reference'),
        interpretation=('Tests downsampling timing at fixed depth, width and head; '
                        'changes spatial compute, receptive fields and BatchNorm statistics. '
                        'Not a compute-matched comparison or proof of information loss.'),
    )
