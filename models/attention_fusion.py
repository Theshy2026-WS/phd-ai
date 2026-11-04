"""Feature-level attention screens on the same trainable native64 CNN8.

These are explicit architecture experiments, not novelty or paper reproduction
claims. Every model retains the original phase00 classification head. Attention
is evaluated in FP32 even when the shared encoder uses mixed precision.
"""
from contextlib import contextmanager
import math

import torch
from torch import nn
from torch.nn import functional as F

from models.spatial_preact import build_spatial_model


VARIANTS = ('control', 'conv16', 'conv_cross', 'self8', 'self16',
            'cross16', 'cross32', 'bidir16', 'gated_cross16')
ATTENTION_VARIANTS = ('self8', 'self16', 'cross16', 'cross32', 'bidir16', 'gated_cross16')
ANALYSIS_MODES = ('learned', 'uniform', 'removed')


class ChannelNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, features):
        return self.norm(features.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class TokenAttention(nn.Module):
    """Pre-normalized Q/K/V routing; optional compact inference statistics."""
    def __init__(self, query_channels, source_channels, output_channels, width, heads):
        super().__init__()
        self.width, self.heads = width, heads
        self.query_norm = nn.LayerNorm(query_channels)
        self.source_norm = nn.LayerNorm(source_channels)
        self.query = nn.Linear(query_channels, width)
        self.key = nn.Linear(source_channels, width)
        self.value = nn.Linear(source_channels, width)
        self.output = nn.Linear(width, output_channels)

    def forward(self, query_features, source_features, *, uniform=False, collect=False, capture=False):
        batch, _, height, width = query_features.shape
        key_height, key_width = source_features.shape[-2:]
        with torch.autocast(query_features.device.type, enabled=False):
            query_tokens = query_features.float().flatten(2).transpose(1, 2)
            source_tokens = source_features.float().flatten(2).transpose(1, 2)
            query_tokens = self.query_norm(query_tokens)
            source_tokens = self.source_norm(source_tokens)
            q = self.query(query_tokens)
            k, v = self.key(source_tokens), self.value(source_tokens)
            q = q.reshape(batch, -1, self.heads, self.width//self.heads).transpose(1, 2)
            k = k.reshape(batch, -1, self.heads, self.width//self.heads).transpose(1, 2)
            v = v.reshape(batch, -1, self.heads, self.width//self.heads).transpose(1, 2)
            if uniform:
                weights = q.new_full((batch, self.heads, q.shape[2], k.shape[2]), 1./k.shape[2])
            else:
                weights = ((q @ k.transpose(-2, -1))/math.sqrt(q.shape[-1])).softmax(-1)
            context = (weights @ v).transpose(1, 2).reshape(batch, -1, self.width)
            result = self.output(context).transpose(1, 2).reshape(batch, -1, height, width)
            statistics = {}
            if collect:
                detached = weights.detach()
                entropy = -(detached*detached.clamp_min(1e-12).log()).sum(-1)
                statistics['attention_entropy'] = entropy.mean((1, 2))/max(1., math.log(k.shape[2]))
                statistics['attention_peak'] = detached.amax(-1).mean((1, 2))
            if capture:
                # Averaged over heads and queries: a retrieval-mass display,
                # not evidence that the highlighted pixels caused a decision.
                statistics['retrieval_map'] = weights.detach().mean((1, 2)).reshape(batch, key_height, key_width)
        return result, statistics, query_tokens, context


class ConvLocal(nn.Module):
    def __init__(self, channels, width):
        super().__init__()
        self.norm = ChannelNorm(channels)
        self.input = nn.Conv2d(channels, width, 1)
        self.local = nn.Conv2d(width, width, 3, padding=1)
        self.output = nn.Conv2d(width, channels, 1)

    def forward(self, features):
        with torch.autocast(features.device.type, enabled=False):
            return self.output(F.gelu(self.local(F.gelu(self.input(self.norm(features.float()))))))


class ConvCross(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.query_norm = ChannelNorm(2048)
        self.source_norm = ChannelNorm(512)
        self.query = nn.Conv2d(2048, width, 1)
        self.source = nn.Conv2d(512, width, 1)
        self.local = nn.Conv2d(width, width, 3, padding=1)
        self.output = nn.Conv2d(width, 2048, 1)

    def forward(self, deep, middle):
        with torch.autocast(deep.device.type, enabled=False):
            query = self.query(self.query_norm(deep.float()))
            source = self.source(self.source_norm(middle.float()))
            source = F.adaptive_avg_pool2d(source, query.shape[-2:])
            return self.output(F.gelu(self.local(query+source)))


class AttentionFusionClassifier(nn.Module):
    def __init__(self, encoder, variant, width=128, heads=4):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f'Unknown attention-fusion variant: {variant}')
        if width < 4 or heads < 1 or width % heads:
            raise ValueError('Need positive heads and a width divisible by heads')
        self.encoder, self.variant = encoder, variant
        self.width, self.heads = width, heads
        self.analysis_mode = 'learned'
        self.capture_attention = False
        self.register_parameter('gamma', None)
        self.fusion = None
        self.reverse = None
        self.reverse_output = None
        self.gate = None
        if variant != 'control':
            self.gamma = nn.Parameter(torch.tensor(.1))
        if variant == 'conv16':
            self.fusion = ConvLocal(512, width)
        elif variant == 'conv_cross':
            self.fusion = ConvCross(width)
        elif variant in ATTENTION_VARIANTS:
            query_channels = 512 if variant == 'self16' else 2048
            source_channels = (2048 if variant == 'self8' else 256 if variant == 'cross32' else 512)
            self.fusion = TokenAttention(query_channels, source_channels, query_channels, width, heads)
            if variant == 'bidir16':
                self.reverse = TokenAttention(512, 2048, width, width, heads)
                self.reverse_output = nn.Conv2d(width, 2048, 1)
            if variant == 'gated_cross16':
                self.gate = nn.Linear(2048+width, 1)
                nn.init.zeros_(self.gate.weight)
                nn.init.zeros_(self.gate.bias)

    @contextmanager
    def analysis(self, mode):
        """Temporarily ablate routing or the complete insertion; restore on failure."""
        if mode not in ANALYSIS_MODES:
            raise ValueError('Unknown attention analysis mode')
        previous = self.analysis_mode
        self.analysis_mode = mode
        try:
            yield self
        finally:
            self.analysis_mode = previous

    def _inject(self, before, delta):
        with torch.autocast(before.device.type, enabled=False):
            after = before.float()+self.gamma.float()*delta.float()
        return after.to(before.dtype)

    def _run(self, images, collect):
        if self.analysis_mode not in ANALYSIS_MODES:
            raise ValueError('Unknown attention analysis mode')
        shallow = self.encoder.stages[0](self.encoder.stem(images))
        middle = self.encoder.stages[1](shallow)
        removed = self.analysis_mode == 'removed'
        statistics, maps, gate, delta = {}, {}, None, None
        if self.variant in ('conv16', 'self16'):
            before = middle
            if removed:
                after = before
            elif self.variant == 'conv16':
                delta = self.fusion(middle)
                after = self._inject(middle, delta)
            else:
                delta, statistics, _, _ = self.fusion(middle, middle,
                    uniform=self.analysis_mode == 'uniform', collect=collect,
                    capture=collect and self.capture_attention)
                after = self._inject(middle, delta)
                if 'retrieval_map' in statistics:
                    maps['middle_self'] = statistics.pop('retrieval_map')
            deep = self.encoder.stages[3](self.encoder.stages[2](after))
        else:
            deep = self.encoder.stages[3](self.encoder.stages[2](middle))
            before = deep
            if self.variant == 'control' or removed:
                after = deep
            elif self.variant == 'conv_cross':
                delta = self.fusion(deep, middle)
                after = self._inject(deep, delta)
                deep = after
            else:
                source = deep if self.variant == 'self8' else shallow if self.variant == 'cross32' else middle
                delta, statistics, query_tokens, context = self.fusion(deep, source,
                    uniform=self.analysis_mode == 'uniform', collect=collect,
                    capture=collect and self.capture_attention)
                if 'retrieval_map' in statistics:
                    maps['deep_reads_source'] = statistics.pop('retrieval_map')
                if self.variant == 'gated_cross16':
                    with torch.autocast(deep.device.type, enabled=False):
                        gate = 2*self.gate(torch.cat((query_tokens, context), -1)).sigmoid()
                        gate = gate.transpose(1, 2).reshape(deep.shape[0], 1, *deep.shape[-2:])
                        delta = delta*gate
                if self.variant == 'bidir16':
                    reverse, reverse_stats, _, _ = self.reverse(middle, before,
                        uniform=self.analysis_mode == 'uniform', collect=collect,
                        capture=collect and self.capture_attention)
                    with torch.autocast(deep.device.type, enabled=False):
                        reverse = self.reverse_output(F.adaptive_avg_pool2d(reverse, before.shape[-2:]))
                        delta = .5*(delta+reverse)
                    if collect:
                        for name in ('attention_entropy', 'attention_peak'):
                            statistics[name] = .5*(statistics[name]+reverse_stats[name])
                    if 'retrieval_map' in reverse_stats:
                        maps['middle_reads_deep'] = reverse_stats['retrieval_map']
                after = self._inject(before, delta)
                deep = after
        logits = self.encoder.head(self.encoder.final_reduction(deep))
        if not collect:
            return logits
        return dict(logits=logits, features_before=before, features_after=after,
            delta=torch.zeros_like(before) if delta is None else delta,
            gamma=before.new_zeros(()) if self.gamma is None else self.gamma.detach(),
            gate=gate, attention_entropy=statistics.get('attention_entropy'),
            attention_peak=statistics.get('attention_peak'), retrieval_maps=maps,
            key_spatial={name: list(value.shape[-2:]) for name, value in maps.items()},
            insertion_stage='stage2' if self.variant in ('conv16', 'self16') else 'stage4')

    def forward(self, images):
        if self.variant == 'control' or self.analysis_mode == 'removed':
            # Direct parent path makes removal an exact inference intervention,
            # and keeps the control's latency free of diagnostic bookkeeping.
            if self.analysis_mode not in ANALYSIS_MODES:
                raise ValueError('Unknown attention analysis mode')
            return self.encoder(images)
        return self._run(images, collect=False)

    def forward_decomposed(self, images):
        return self._run(images, collect=True)

    @property
    def attention_flops_per_image(self):
        return attention_matmul_flops(self.variant, self.width)


def attention_matmul_flops(variant, width=128, batch=1):
    """QK and AV only; existing profiling already counts learned Linear layers."""
    if variant not in VARIANTS or width < 1 or batch < 1:
        raise ValueError('Invalid attention FLOPs request')
    products = dict(self8=64*64, self16=256*256, cross16=64*256,
                    cross32=64*1024, bidir16=2*64*256, gated_cross16=64*256)
    return 4*products.get(variant, 0)*width*batch


def build_attention_model(variant, parent_state=None, seed=42, width=128, heads=4):
    if variant not in VARIANTS:
        raise ValueError(f'Unknown attention-fusion variant: {variant}')
    with torch.random.fork_rng(devices=[]):
        encoder = build_spatial_model('cnn8', seed)
        if parent_state is not None:
            encoder.load_state_dict(parent_state, strict=True)
        torch.default_generator.manual_seed(seed+404001)
        model = AttentionFusionClassifier(encoder, variant, width, heads)
    return model.requires_grad_(True)


def description(variant, width=128, heads=4):
    if variant not in VARIANTS:
        raise ValueError(f'Unknown attention-fusion variant: {variant}')
    spatial = dict(self8=[8, 8], self16=[16, 16], cross16=[8, 16],
                   cross32=[8, 32], bidir16=[[8, 16], [16, 8]], gated_cross16=[8, 16])
    return dict(variant=variant, encoder='same preactivation CNN8, depths3/4/6/3',
        encoder_frozen=False, stage_spatial_sizes=[32, 16, 8, 8],
        classifier='unchanged phase00 BN-ReLU-GAP-linear',
        insertion_stage='stage2 before later CNN stages' if variant in ('conv16', 'self16') else 'stage4 before classifier',
        attention_blocks=2 if variant == 'bidir16' else int(variant in ATTENTION_VARIANTS),
        attention_query_source_spatial_sizes=spatial.get(variant), width=width, heads=heads,
        feature_fusion='residual feature injection; gamma initialized0.1' if variant != 'control' else 'none',
        gate='2*sigmoid(content linear), initialized1' if variant == 'gated_cross16' else None,
        bidirectional_fusion='average of deep-read-middle and pooled middle-read-deep updates' if variant == 'bidir16' else None,
        attention_precision='FP32 Q/K/V, softmax and aggregation; encoder follows training AMP policy',
        positional_encoding='none; geometry is carried by CNN features',
        analysis_modes=list(ANALYSIS_MODES), uniform_mode_for_controls='no change',
        attention_matmul_flops=attention_matmul_flops(variant, width),
        comparison_limit='Conv controls approximately compare capacity/compute; they are not exactly parameter/FLOP matched',
        interpretation='Architecture screening, not a novelty or paper reproduction claim')
