"""AlterNet core extracted from the authors' Apache-2.0 implementation.

Upstream: xxxnell/how-do-vits-work @ 9571ff1370f31af2f433558fbfa7bc87a97b0bab
Sources: models/{alternet,attentions,preresnet_dnn_block,classifier_block}.py
Modified for this project: self-contained imports; omit unused ablation/dropout
variants; register relative indices as buffers; configurable Tiny stem stride.
Attention2d upstream credits https://github.com/lucidrains/vit-pytorch.
See third_party/ALTER_NET_NOTICE.md and Apache-2.0.txt.
"""
from functools import partial
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn, einsum
from einops import rearrange
from timm.layers import DropPath


def conv1x1(a, b, stride=1):
    return nn.Conv2d(a, b, 1, stride=stride, bias=False)


def conv3x3(a, b, stride=1, groups=1):
    return nn.Conv2d(a, b, 3, stride=stride, padding=1, groups=groups, bias=False)


layers = SimpleNamespace(conv1x1=conv1x1, conv3x3=conv3x3,
                         bn=nn.BatchNorm2d, relu=nn.ReLU, dense=nn.Linear,
                         DropPath=DropPath,
                         convnxn=lambda a, b, **kw: nn.Conv2d(a, b, bias=False, **kw))

class Attention2d(nn.Module):

    def __init__(self, dim_in, dim_out=None, *,
                 heads=8, dim_head=64, dropout=0.0, k=1):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5

        inner_dim = dim_head * heads
        dim_out = dim_in if dim_out is None else dim_out

        self.to_q = nn.Conv2d(dim_in, inner_dim * 1, 1, bias=False)
        self.to_kv = nn.Conv2d(dim_in, inner_dim * 2, k, stride=k, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim_out, 1),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        )

    def forward(self, x, mask=None):
        b, n, _, y = x.shape
        qkv = (self.to_q(x), *self.to_kv(x).chunk(2, dim=1))
        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> b h (x y) d', h=self.heads), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        dots = dots + mask if mask is not None else dots
        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', y=y)

        out = self.to_out(out)

        return out, attn

class LocalAttention(nn.Module):

    def __init__(self, dim_in, dim_out=None, *,
                 window_size=7, k=1,
                 heads=8, dim_head=32, dropout=0.0):
        super().__init__()
        self.attn = Attention2d(dim_in, dim_out,
                                heads=heads, dim_head=dim_head, dropout=dropout, k=k)
        self.window_size = window_size

        self.register_buffer("rel_index", self.rel_distance(window_size) + window_size - 1, persistent=False)
        self.pos_embedding = nn.Parameter(torch.randn(2 * window_size - 1, 2 * window_size - 1) * 0.02)

    def forward(self, x, mask=None):
        b, c, h, w = x.shape
        p = self.window_size
        n1 = h // p
        n2 = w // p

        mask = torch.zeros(p ** 2, p ** 2, device=x.device) if mask is None else mask
        mask = mask + self.pos_embedding[self.rel_index[:, :, 0], self.rel_index[:, :, 1]]

        x = rearrange(x, "b c (n1 p1) (n2 p2) -> (b n1 n2) c p1 p2", p1=p, p2=p)
        x, attn = self.attn(x, mask)
        x = rearrange(x, "(b n1 n2) c p1 p2 -> b c (n1 p1) (n2 p2)", n1=n1, n2=n2, p1=p, p2=p)

        return x, attn

    @staticmethod
    def rel_distance(window_size):
        i = torch.tensor(np.array([[x, y] for x in range(window_size) for y in range(window_size)]))
        d = i[None, :, :] - i[:, None, :]

        return d

class PreBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, channels,
                 stride=1, groups=1, width_per_group=64, sd=0.0,
                 **block_kwargs):
        super(PreBottleneck, self).__init__()

        width = int(channels * (width_per_group / 64.)) * groups

        self.shortcut = []
        if stride != 1 or in_channels != channels * self.expansion:
            self.shortcut.append(layers.conv1x1(in_channels, channels * self.expansion, stride=stride))
        self.shortcut = nn.Sequential(*self.shortcut)
        self.bn = layers.bn(in_channels)
        self.relu = layers.relu()

        self.conv1 = layers.conv1x1(in_channels, width)
        self.conv2 = nn.Sequential(
            layers.bn(width),
            layers.relu(),
            layers.conv3x3(width, width, stride=stride, groups=groups),
        )
        self.conv3 = nn.Sequential(
            layers.bn(width),
            layers.relu(),
            layers.conv1x1(width, channels * self.expansion),
        )

        self.sd = layers.DropPath(sd) if sd > 0.0 else nn.Identity()

    def forward(self, x):
        if len(self.shortcut) > 0:
            x = self.bn(x)
            x = self.relu(x)
            skip = self.shortcut(x)
        else:
            skip = self.shortcut(x)
            x = self.bn(x)
            x = self.relu(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.sd(x) + skip

        return x

class AttentionBlockB(nn.Module):
    # Attention block with pre-activation.
    # We use this block by default.
    expansion = 4

    def __init__(self, dim_in, dim_out=None, *,
                 heads=8, dim_head=64, dropout=0.0, sd=0.0,
                 stride=1, window_size=7, k=1, norm=nn.BatchNorm2d, activation=nn.GELU,
                 **block_kwargs):
        super().__init__()
        dim_out = dim_in if dim_out is None else dim_out
        attn = partial(LocalAttention, window_size=window_size, k=k)
        width = dim_in // self.expansion

        self.shortcut = []
        if stride != 1 or dim_in != dim_out * self.expansion:
            self.shortcut.append(layers.conv1x1(dim_in, dim_out * self.expansion, stride=stride))
        self.shortcut = nn.Sequential(*self.shortcut)
        self.norm1 = norm(dim_in)
        self.relu = activation()

        self.conv = nn.Conv2d(dim_in, width, kernel_size=1, bias=False)
        self.norm2 = norm(width)
        self.attn = attn(width, dim_out * self.expansion, heads=heads, dim_head=dim_head, dropout=dropout)
        self.sd = DropPath(sd) if sd > 0.0 else nn.Identity()

    def forward(self, x):
        if len(self.shortcut) > 0:
            x = self.norm1(x)
            x = self.relu(x)
            skip = self.shortcut(x)
        else:
            skip = self.shortcut(x)
            x = self.norm1(x)
            x = self.relu(x)

        x = self.conv(x)
        x = self.norm2(x)
        x, attn = self.attn(x)

        x = self.sd(x) + skip

        return x

class StemB(nn.Module):
    # Stem stage for pre-activation pattern based on pre-activation ResNet.
    # We use this block by default.

    def __init__(self, dim_in, dim_out, pool=True):
        super().__init__()

        self.layer0 = []
        if pool:
            self.layer0.append(layers.convnxn(dim_in, dim_out, kernel_size=7, stride=2, padding=3))
            self.layer0.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        else:
            self.layer0.append(layers.conv3x3(dim_in, dim_out, stride=1))
        self.layer0 = nn.Sequential(*self.layer0)

    def forward(self, x):
        x = self.layer0(x)
        return x

class BNGAPBlock(nn.Module):

    def __init__(self, in_features, num_classes, **kwargs):
        super(BNGAPBlock, self).__init__()

        self.bn = layers.bn(in_features)
        self.relu = layers.relu()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dense = layers.dense(in_features, num_classes)

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.gap(x)
        x = x.view(x.size()[0], -1)
        x = self.dense(x)

        return x

classifier = SimpleNamespace(BNGAPBlock=BNGAPBlock, MLPBlock=None)

class AlterNet(nn.Module):

    def __init__(self, block1, block2, *,
                 num_blocks, num_blocks2, heads,
                 cblock=classifier.BNGAPBlock,
                 sd=0.0, num_classes=10, stem=StemB, name="alternet", **block_kwargs):
        super().__init__()
        self.name = name
        idxs = [[j for j in range(sum(num_blocks[:i]), sum(num_blocks[:i + 1]))] for i in range(len(num_blocks))]
        sds = [[sd * j / (sum(num_blocks) - 1) for j in js] for js in idxs]

        self.layer0 = stem(3, 64)
        self.layer1 = self._make_layer(block1, block2, 64, 64,
                                       num_blocks[0], num_blocks2[0], stride=1, heads=heads[0], sds=sds[0], **block_kwargs)
        self.layer2 = self._make_layer(block1, block2, 64 * block2.expansion, 128,
                                       num_blocks[1], num_blocks2[1], stride=2, heads=heads[1], sds=sds[1], **block_kwargs)
        self.layer3 = self._make_layer(block1, block2, 128 * block2.expansion, 256,
                                       num_blocks[2], num_blocks2[2], stride=2, heads=heads[2], sds=sds[2], **block_kwargs)
        self.layer4 = self._make_layer(block1, block2, 256 * block2.expansion, 512,
                                       num_blocks[3], num_blocks2[3], stride=2, heads=heads[3], sds=sds[3], **block_kwargs)

        self.classifier = []
        if cblock is classifier.MLPBlock:
            self.classifier.append(nn.AdaptiveAvgPool2d((7, 7)))
            self.classifier.append(cblock(7 * 7 * 512 * block2.expansion, num_classes, **block_kwargs))
        else:
            self.classifier.append(cblock(512 * block2.expansion, num_classes, **block_kwargs))
        self.classifier = nn.Sequential(*self.classifier)

    @staticmethod
    def _make_layer(block1, block2, in_channels, out_channels, num_block1, num_block2, stride, heads, sds, **block_kwargs):
        alt_seq = [False] * (num_block1 - num_block2 * 2) + [False, True] * num_block2
        stride_seq = [stride] + [1] * (num_block1 - 1)

        seq, channels = [], in_channels
        for alt, stride, sd in zip(alt_seq, stride_seq, sds):
            block = block1 if not alt else block2
            seq.append(block(channels, out_channels, stride=stride, sd=sd, heads=heads, **block_kwargs))
            channels = out_channels * block.expansion

        return nn.Sequential(*seq)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.classifier(x)

        return x
