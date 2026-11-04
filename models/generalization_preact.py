"""CNN controls with explicit readout, depth and normalization differences."""
import torch
from torch import nn
from torch.nn import functional as F

from models.spatial_preact import SpatialPreActResNet50


VARIANTS = ('warm_control', 'warm_full8', 'warm_phasebalanced', 'warm_gentle',
            'warm_fp32_control', 'warm_sam', 'fresh_control', 'fresh_full8',
            'fresh_phasebalanced', 'fresh_fp32_control', 'fresh_sam',
            'fresh_gentle', 'fresh_no_cutmix', 'fresh_no_randaugment',
            'fresh_sgd', 'fresh_shallow26', 'fresh_groupnorm',
            'fresh_wd_low', 'fresh_wd_high')


def _group_norm(module):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            groups = min(32, child.num_features)
            while child.num_features % groups:
                groups -= 1
            replacement = nn.GroupNorm(groups, child.num_features, eps=child.eps, affine=child.affine)
            if child.affine:
                with torch.no_grad():
                    replacement.weight.copy_(child.weight)
                    replacement.bias.copy_(child.bias)
            setattr(module, name, replacement)
        else:
            _group_norm(child)


class GeneralizationCNN(SpatialPreActResNet50):
    def __init__(self, readout='phase00'):
        super().__init__('cnn8')
        if readout not in ('phase00', 'full8', 'phasebalanced'):
            raise ValueError('Unknown generalization readout')
        self.readout = readout
        if readout != 'phase00':
            self.final_reduction = nn.Identity()

    def training_loss(self, images, targets):
        if self.readout != 'phasebalanced':
            return soft_loss(self(images), targets)
        features = self.stages(self.stem(images))
        # Normalize the full map ONCE: four head calls in train mode would
        # update its BN four times and leave statistics biased to the last grid.
        features = self.head.relu(self.head.bn(features))
        full = self.head.dense(self.head.gap(features).flatten(1))
        phases = [self.head.dense(self.head.gap(features[..., r::2, c::2]).flatten(1))
                  for r in range(2) for c in range(2)]
        return .5*soft_loss(full, targets) + .125*sum(soft_loss(x, targets) for x in phases)


def soft_loss(logits, targets):
    return -(targets*F.log_softmax(logits.float(), dim=1)).sum(1).mean()


def build_generalization_model(variant, parent_state=None, seed=42):
    if variant not in VARIANTS:
        raise ValueError('Unknown generalization variant')
    readout = ('phasebalanced' if 'phasebalanced' in variant else
               'full8' if variant.endswith('_full8') else 'phase00')
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        model = GeneralizationCNN(readout)
        if variant == 'fresh_shallow26':
            # Same bottleneck family, channels, stride layout and classifier;
            # retain two blocks/stage instead of3/4/6/3. Surviving initialized
            # weights remain identical to their full-depth counterparts.
            model.stages = nn.Sequential(*[
                nn.Sequential(*list(stage.children())[:2]) for stage in model.stages])
        if variant == 'fresh_groupnorm':
            _group_norm(model)
        if variant.startswith('warm_'):
            if parent_state is None:
                raise ValueError('Warm-start jobs require the original CNN8 state')
            model.load_state_dict(parent_state, strict=True)
    return model.requires_grad_(True)


def initialization_family(variant):
    """Only demand a full state digest match for identical parameter/buffer layouts."""
    if variant not in VARIANTS:
        raise ValueError('Unknown generalization variant')
    return ('shallow26_bn' if variant == 'fresh_shallow26' else
            'full_depth_gn' if variant == 'fresh_groupnorm' else 'full_depth_bn')


def description(variant):
    if variant not in VARIANTS:
        raise ValueError('Unknown generalization variant')
    return dict(variant=variant, warm_start=variant.startswith('warm_'),
        encoder_frozen=False, spatial_map=[8, 8], extra_parameters=0,
        attention_blocks=0, parameter_matched=variant != 'fresh_shallow26',
        depths=[2, 2, 2, 2] if variant == 'fresh_shallow26' else [3, 4, 6, 3],
        normalization='GroupNorm' if variant == 'fresh_groupnorm' else 'BatchNorm',
        initialization_family=initialization_family(variant),
        readout=('full8' if variant.endswith('_full8') or 'phasebalanced' in variant else 'phase00'),
        objective=('0.5 full8 CE + 0.125 sum of four phase CE; full-map head BN once'
                   if 'phasebalanced' in variant else 'single-readout soft-target cross entropy'),
        sam=variant.endswith('_sam'), gentle_augmentation=variant.endswith('_gentle'))
