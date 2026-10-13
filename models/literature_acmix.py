"""Independent ACmix implementation, checked against the authors' ResNet code.

Reference: LeapLabTHU/ACmix@81dddb6dff98f5e238a7fb6ab174e256489c07fa.
See third_party/ACMIX_NOTICE.md for equations, initialization and adaptations.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torchvision.models import resnet50


class ACmix(nn.Module):
    def __init__(self, channels, stride=1, heads=4, window=7, checkpoint_attention=True):
        super().__init__()
        if channels % heads or window % 2 != 1 or stride not in (1, 2):
            raise ValueError('ACmix requires divisible channels, odd window and stride 1/2')
        self.channels, self.heads, self.window, self.stride = channels, heads, window, stride
        self.head_dim = channels // heads
        self.checkpoint_attention = checkpoint_attention
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.position = nn.Conv2d(2, self.head_dim, 1)
        self.aggregate = nn.Conv2d(3 * heads, 9, 1, bias=False)
        # Author operator's bias is actually None after its reset_parameters.
        self.shift = nn.Conv2d(9 * self.head_dim, channels, 3, stride=stride,
                               padding=1, groups=self.head_dim, bias=False)
        self.attention_weight = nn.Parameter(torch.full((1,), 0.5))
        self.convolution_weight = nn.Parameter(torch.full((1,), 0.5))

    def _local_attention(self, q, k, v, position):
        b, c, h, w = q.shape
        d, step, area = self.head_dim, self.stride, self.window**2
        oh, ow = h // step, w // step
        query = q.reshape(b * self.heads, d, h, w)[:, :, ::step, ::step] * d**-0.5
        def neighborhoods(value):
            padded = F.pad(value, (self.window//2,) * 4, mode='reflect')
            return F.unfold(padded, self.window, stride=step).reshape(-1, d, area, oh, ow)
        keys = neighborhoods(k.reshape(b * self.heads, d, h, w))
        relative = position[:, :, ::step, ::step].unsqueeze(2) - neighborhoods(position)
        scores = (query.unsqueeze(2) * (keys + relative)).sum(dim=1)
        weights = scores.softmax(dim=1)
        values = neighborhoods(v.reshape(b * self.heads, d, h, w))
        return (weights.unsqueeze(1) * values).sum(dim=2).reshape(b, c, oh, ow)

    def forward(self, x):
        # Includes projections: casting after an overflowing FP16 QK is too late.
        with torch.autocast(x.device.type, enabled=False):
            x = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
            b, c, h, w = x.shape
            if min(h, w) <= self.window//2 or h % self.stride or w % self.stride:
                raise ValueError('ACmix reflection window/stride incompatible with spatial size')
            q, k, v = self.q(x), self.k(x), self.v(x)
            yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype),
                                    torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype), indexing='ij')
            position = self.position(torch.stack((xx, yy)).unsqueeze(0))
            if self.training and torch.is_grad_enabled() and self.checkpoint_attention:
                attention = checkpoint(self._local_attention, q, k, v, position, use_reentrant=False)
            else:
                attention = self._local_attention(q, k, v, position)
            projected = torch.cat([t.reshape(b, self.heads, self.head_dim, h*w) for t in (q, k, v)], dim=1)
            mixed = self.aggregate(projected).permute(0, 2, 1, 3).reshape(b, -1, h, w)
            convolution = self.shift(mixed)
            return self.attention_weight * attention + self.convolution_weight * convolution


def build_acmix():
    # Same Tiny stem and 3/4/6/3 layout as our ResNet control.
    model = resnet50(weights=None, num_classes=200)
    model.conv1 = nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False)
    model.maxpool = nn.Identity()
    for stage in (model.layer1, model.layer2, model.layer3, model.layer4):
        for block in stage:
            block.conv2 = ACmix(block.conv2.in_channels, stride=block.stride)
    # Match the official ResNet constructor's final initialization pass, which
    # overwrites the standalone ACmix shift-kernel initialization with Kaiming.
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    return model


def acmix_description():
    return dict(name='acmix50', input_size=[3, 64, 64], classes=200, pretrained=False,
                source='Independent implementation checked against LeapLabTHU/ACmix@81dddb6dff98f5e238a7fb6ab174e256489c07fa',
                adaptation=True, stem='3x3 stride 2, no maxpool',
                stage_spatial_sizes=[32, 16, 8, 4], depths=[3, 4, 6, 3],
                control='resnet50', attention_blocks=16, attention_heads=4,
                attention_window=7, convolution_kernel=3,
                fusion='two learned static scalars per block, initialized to 0.5; not input-dependent gates',
                attention_precision='whole ACmix operator FP32 island',
                attention_activation_checkpointing=True,
                comparison_role='ResNet-family operator replacement; author-style initialization')
