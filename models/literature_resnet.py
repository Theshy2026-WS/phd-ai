"""ResNet-50 family adaptations for native 64x64 Tiny ImageNet.

AlterNet uses the vendored author's core. BoTNet is an implementation of
Srinivas et al. (CVPR 2021), Table 1 / Figure 4, NOT an official port.
All Tiny-specific deviations are documented in LITERATURE_BASELINES.md.
"""
from functools import partial
import torch
from torch import nn
from torchvision.models import resnet50
from models.literature_alternet import AlterNet, PreBottleneck, AttentionBlockB, StemB


DEFAULT_VARIANTS = ('resnet50', 'botnet50', 'preact50', 'alternet50')
VARIANTS = DEFAULT_VARIANTS + ('coatnet0', 'acmix50', 'conformer_tiny',
                             'resnet50_preact', 'botnet_twostage', 'coatnet_twostage')
CONTROLS = {'botnet50': 'resnet50', 'alternet50': 'preact50', 'acmix50': 'resnet50'}


class BoTAttention(nn.Module):
    """Global four-head attention with query-dependent relative H/W embeddings.

Unlike a scalar Swin relative bias, logits contain q dot r (Figure 4).
No output projection or FFN: the surrounding ResNet bottleneck supplies them.
"""
    def __init__(self, channels, size, heads=4):
        super().__init__()
        if channels % heads or size < 1:
            raise ValueError('Invalid BoT attention dimensions')
        self.heads, self.size = heads, size
        self.dim_head = channels // heads
        self.qkv = nn.Conv2d(channels, 3 * channels, 1, bias=False)
        self.rel_h = nn.Parameter(torch.randn(2*size-1, self.dim_head) * self.dim_head**-0.5)
        self.rel_w = nn.Parameter(torch.randn(2*size-1, self.dim_head) * self.dim_head**-0.5)
        coords = torch.stack(torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')).flatten(1)
        relative = coords[:, None, :] - coords[:, :, None] + size - 1
        self.register_buffer('relative_h', relative[0], persistent=False)
        self.register_buffer('relative_w', relative[1], persistent=False)

    def forward(self, x):
        # FP16 dot products can overflow BEFORE softmax. Casting logits after
        # QK is too late; keep projections, relative terms, QK and AV in FP32.
        # Preserve double precision for reference tests. Surrounding CNN stays AMP.
        with torch.autocast(device_type=x.device.type, enabled=False):
            return self._attention(x.float() if x.dtype in (torch.float16, torch.bfloat16) else x)

    def _attention(self, x):
        b, c, h, w = x.shape
        if (h, w) != (self.size, self.size):
            raise ValueError(f'BoTAttention expects {self.size}x{self.size}, got {h}x{w}')
        n = h*w
        q, k, v = self.qkv(x).reshape(b, 3, self.heads, self.dim_head, n).unbind(1)
        q = q.transpose(-2, -1) * self.dim_head**-0.5
        # Small intermediate tables avoid a [N,N,head_dim] relative tensor.
        rh = q @ self.rel_h.to(q.dtype).T
        rw = q @ self.rel_w.to(q.dtype).T
        ih = self.relative_h[None, None].expand(b, self.heads, -1, -1)
        iw = self.relative_w[None, None].expand(b, self.heads, -1, -1)
        logits = q @ k + rh.gather(-1, ih) + rw.gather(-1, iw)
        weights = logits.float().softmax(-1).to(v.dtype)
        return (weights @ v.transpose(-2, -1)).transpose(-2, -1).reshape(b, c, h, w)


def build_model(variant, seed=42):
    if variant not in VARIANTS:
        raise ValueError(f'Unknown variant: {variant}')
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        if variant in ('resnet50_preact', 'botnet_twostage', 'coatnet_twostage'):
            from models.literature_twostage import build_twostage
            model = build_twostage(variant)
        elif variant == 'acmix50':
            from models.literature_acmix import build_acmix
            model = build_acmix()
        elif variant == 'conformer_tiny':
            from models.literature_conformer import TinyConformer
            model = TinyConformer()
        elif variant == 'coatnet0':
            from models.literature_coatnet import build_coatnet
            model = build_coatnet()
        elif variant in ('resnet50', 'botnet50'):
            model = resnet50(weights=None, num_classes=200)
            model.conv1 = nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False)
            nn.init.kaiming_normal_(model.conv1.weight, mode='fan_out', nonlinearity='relu')
            model.maxpool = nn.Identity()
            if variant == 'botnet50':
                # Attention is before downsampling: 8x8 in first layer4 block,
                # then 4x4. Keep the original projection shortcuts and BN/ReLU.
                torch.default_generator.manual_seed(seed + 1000)
                for i, block in enumerate(model.layer4):
                    attention = BoTAttention(512, size=8 if i == 0 else 4)
                    nn.init.kaiming_normal_(attention.qkv.weight, mode='fan_out', nonlinearity='relu')
                    block.conv2 = nn.Sequential(attention, nn.AvgPool2d(2) if i == 0 else nn.Identity())
        else:
            # Author dnn_50 layout: one additional layer4 block, shared by its
            # CNN control. No invented attention block or timm Swin substitution.
            model = AlterNet(PreBottleneck, AttentionBlockB,
                             num_blocks=(3, 4, 6, 4),
                             num_blocks2=(0, 1, 3, 2) if variant == 'alternet50' else (0, 0, 0, 0),
                             heads=(3, 6, 12, 24), dim_head=64, window_size=4,
                             stem=partial(StemB, pool=False), num_classes=200)
            model.layer0.layer0[0].stride = (2, 2)
    return model


def architecture_description(variant):
    if variant in ('resnet50_preact', 'botnet_twostage', 'coatnet_twostage'):
        from models.literature_twostage import experiment_description
        return experiment_description(variant)
    if variant == 'acmix50':
        from models.literature_acmix import acmix_description
        return acmix_description()
    if variant == 'conformer_tiny':
        from models.literature_conformer import conformer_description
        return conformer_description()
    if variant == 'coatnet0':
        from models.literature_coatnet import coatnet_description
        return coatnet_description()
    return dict(
        name=variant, input_size=[3, 64, 64], classes=200, pretrained=False,
        stem='3x3 stride 2, no maxpool', stage_spatial_sizes=[32, 16, 8, 4],
        depths=[3, 4, 6, 4] if variant in ('preact50', 'alternet50') else [3, 4, 6, 3],
        control=CONTROLS.get(variant),
        source=('xxxnell/how-do-vits-work@9571ff1370f31af2f433558fbfa7bc87a97b0bab'
                if variant in ('preact50', 'alternet50') else 'torchvision ResNet-50 v1.5; BoTNet paper equations'),
        adaptation=True,
        attention_blocks=6 if variant == 'alternet50' else 3 if variant == 'botnet50' else 0,
        attention_precision='fp32 island' if variant in ('botnet50', 'alternet50') else 'outer autocast',
    )
