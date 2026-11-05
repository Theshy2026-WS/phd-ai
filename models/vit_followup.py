"""Explicit full-ViT, dual-image fusion and spatial-attention follow-ups.

These models preserve the old CNN8 path and do not modify historical factories.
The 64x64 experiment uses all 4096 shallow positions as Q, K and V. Query
chunking/checkpointing changes memory usage, not the global attention operator.
"""
from contextlib import contextmanager
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from models.spatial_preact import build_spatial_model
from models.vit import ViT


VARIANTS = ('vit', 'dual_fusion', 'shallow64', 'cross16_mlp')
VIT_CONFIGURATION = dict(input_channels=3, num_classes=200, image_size=64,
    patch_size=4, embed_dim=224, depth=5, num_heads=4, mlp_ratio=4.,
    dropout=.1, attention_dropout=0.)


class AnalysisMixin:
    @contextmanager
    def analysis(self, mode):
        if mode not in self.analysis_modes:
            raise ValueError(f'Unsupported analysis mode {mode!r} for {self.variant}')
        old = self.analysis_mode
        self.analysis_mode = mode
        try:
            yield self
        finally:
            self.analysis_mode = old


def _entropy_peak(weights):
    detached = weights.detach()
    entropy = -(detached * detached.clamp_min(1e-12).log()).sum(-1)
    return (entropy.mean((1, 2)) / max(1., math.log(weights.shape[-1])),
            detached.amax(-1).mean((1, 2)))


class FP32MultiheadAttention(nn.MultiheadAttention):
    """State-compatible self-attention with an explicit FP32 arithmetic island.

    Subclassing MultiheadAttention also preserves existing FLOPs hook accounting
    for the packed projections and the attention matmuls.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.analysis_mode = 'learned'
        self.collect_statistics = False
        self.last_statistics = {}

    def forward(self, query, key, value, need_weights=False, **kwargs):
        if kwargs.get('attn_mask') is not None or kwargs.get('key_padding_mask') is not None:
            raise ValueError('This native64 ViT does not use attention masks')
        if query is not key or query is not value or not self.batch_first:
            raise ValueError('The follow-up ViT expects batch-first self-attention')
        with torch.autocast(query.device.type, enabled=False):
            x = query.float()
            batch, count, width = x.shape
            qkv = F.linear(x, self.in_proj_weight, self.in_proj_bias)
            q, k, v = [t.reshape(batch, count, self.num_heads,
                width // self.num_heads).transpose(1, 2) for t in qkv.chunk(3, -1)]
            if self.analysis_mode == 'uniform':
                weights = q.new_full((batch, self.num_heads, count, count), 1. / count)
            elif self.analysis_mode == 'learned':
                weights = ((q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])).softmax(-1)
            else:
                raise ValueError('Unknown ViT attention mode')
            if self.collect_statistics:
                entropy, peak = _entropy_peak(weights)
                self.last_statistics = dict(attention_entropy=entropy, attention_peak=peak)
            else:
                self.last_statistics = {}
            used_weights = F.dropout(weights, self.dropout, self.training)
            context = (used_weights @ v).transpose(1, 2).reshape(batch, count, width)
            result = self.out_proj(context)
        returned_weights = weights.mean(1) if need_weights else None
        return result, returned_weights


class FollowupViT(ViT, AnalysisMixin):
    """Complete five-block ViT; keys match models.vit.ViT for checkpoint loading."""
    variant = 'vit'
    analysis_modes = ('learned', 'uniform', 'removed')

    def __init__(self):
        super().__init__(**VIT_CONFIGURATION)
        self.encoder = nn.Identity()
        self.analysis_mode = 'learned'
        for block in self.blocks:
            attention = FP32MultiheadAttention(224, 4, dropout=0., batch_first=True)
            attention.load_state_dict(block.attention.state_dict(), strict=True)
            block.attention = attention

    def _tokens(self, images, collect=False):
        with torch.autocast(images.device.type, enabled=False):
            patches = self.patch_embedding(images.float()).flatten(2).transpose(1, 2)
            if patches.shape[1] != self.num_patches:
                raise ValueError(f'Expected {self.num_patches} image tokens')
            cls = self.class_token.expand(images.shape[0], -1, -1)
            tokens = self.embedding_dropout(torch.cat((cls, patches), 1) + self.position_embedding)
            before = tokens[:, 0]
            for block in self.blocks:
                normalized = block.norm1(tokens)
                if self.analysis_mode == 'removed':
                    attended = torch.zeros_like(tokens)
                    block.attention.last_statistics = {}
                else:
                    block.attention.analysis_mode = self.analysis_mode
                    block.attention.collect_statistics = collect
                    attended, _ = block.attention(normalized, normalized, normalized, need_weights=False)
                tokens = tokens + block.attention_dropout(attended)
                tokens = tokens + block.mlp(block.norm2(tokens))
            features = self.norm(tokens)[:, 0]
            stats = self.blocks[-1].attention.last_statistics if collect else {}
        return before, features, stats

    def forward_features(self, images):
        return self._tokens(images, False)[1]

    def forward(self, images):
        with torch.autocast(images.device.type, enabled=False):
            return self.classifier(self.forward_features(images))

    def forward_decomposed(self, images):
        before, after, stats = self._tokens(images, True)
        with torch.autocast(images.device.type, enabled=False):
            logits = self.classifier(after)
        return dict(logits=logits, features_before=before, features_after=after,
            gamma=after.new_zeros(()), gate=None, retrieval_maps={}, **stats)

    @property
    def attention_flops_per_image(self):
        # nn.MultiheadAttention hooks already include these matmuls/projections.
        return 0


def _cnn_features(encoder, images):
    shallow = encoder.stages[0](encoder.stem(images))
    middle = encoder.stages[1](shallow)
    deep = encoder.stages[3](encoder.stages[2](middle))
    return middle, deep


def _cnn_pool(encoder, deep):
    selected = encoder.final_reduction(deep)
    return encoder.head.gap(encoder.head.relu(encoder.head.bn(selected))).flatten(1)


class DualFeatureFusion(nn.Module, AnalysisMixin):
    """Independent, jointly trainable CNN and full ViT with final feature concat."""
    variant = 'dual_fusion'
    analysis_modes = ('learned', 'cnn_only', 'vit_only')

    def __init__(self, encoder, vit):
        super().__init__()
        self.encoder, self.vit = encoder, vit
        self.analysis_mode = 'learned'
        old_classifier = self.encoder.head.dense
        self.fusion_classifier = nn.Linear(2048 + 224, 200)
        with torch.no_grad():
            self.fusion_classifier.weight[:, :2048].copy_(old_classifier.weight)
            nn.init.normal_(self.fusion_classifier.weight[:, 2048:], std=.001)
            self.fusion_classifier.bias.copy_(old_classifier.bias)
        # Do not keep unused trainable classification heads in either branch.
        self.encoder.head.dense = nn.Identity()
        self.vit.classifier = nn.Identity()

    def _run(self, images, collect):
        _, deep = _cnn_features(self.encoder, images)
        cnn = _cnn_pool(self.encoder, deep)
        vit = self.vit.forward_features(images)
        before = torch.cat((cnn.float(), torch.zeros_like(vit)), 1)
        if self.analysis_mode == 'cnn_only':
            after = before
        elif self.analysis_mode == 'vit_only':
            after = torch.cat((torch.zeros_like(cnn.float()), vit), 1)
        elif self.analysis_mode == 'learned':
            after = torch.cat((cnn.float(), vit), 1)
        else:
            raise ValueError('Unknown dual-feature analysis mode')
        with torch.autocast(images.device.type, enabled=False):
            logits = self.fusion_classifier(after)
        if not collect:
            return logits
        return dict(logits=logits, features_before=before, features_after=after,
            cnn_features=cnn, vit_features=vit, gamma=after.new_zeros(()),
            gate=None, retrieval_maps={})

    def forward(self, images):
        return self._run(images, False)

    def forward_decomposed(self, images):
        return self._run(images, True)

    @property
    def attention_flops_per_image(self):
        return 0


class ChunkedGlobalAttention(nn.Module):
    """Exact global attention with recomputed query chunks during training."""
    def __init__(self, channels=64, width=128, heads=4, query_chunk_size=64):
        super().__init__()
        if width < 1 or heads < 1 or width % heads or query_chunk_size < 1:
            raise ValueError('Invalid chunked attention dimensions')
        self.width, self.heads, self.query_chunk_size = width, heads, query_chunk_size
        self.norm = nn.LayerNorm(channels)
        self.query = nn.Linear(channels, width)
        self.key = nn.Linear(channels, width)
        self.value = nn.Linear(channels, width)
        self.output = nn.Linear(width, width)
        self.last_token_count = None

    def _attend(self, q, k, v):
        weights = ((q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])).softmax(-1)
        return weights @ v

    def forward(self, features, uniform=False, collect=False):
        with torch.autocast(features.device.type, enabled=False):
            batch, _, height, width = features.shape
            tokens = self.norm(features.float().flatten(2).transpose(1, 2))
            count = tokens.shape[1]
            self.last_token_count = count
            q, k, v = [projection(tokens).reshape(batch, count, self.heads,
                self.width // self.heads).transpose(1, 2)
                for projection in (self.query, self.key, self.value)]
            if uniform:
                context = v.mean(2, keepdim=True).expand(-1, -1, count, -1)
                entropy, peak = q.new_ones(batch), q.new_full((batch,), 1. / count)
            else:
                parts = []
                entropy, peak = q.new_zeros(batch), q.new_zeros(batch)
                for start in range(0, count, self.query_chunk_size):
                    piece = q[:, :, start:start+self.query_chunk_size]
                    if self.training and torch.is_grad_enabled():
                        attended = checkpoint(self._attend, piece, k, v,
                            use_reentrant=False, preserve_rng_state=False)
                    else:
                        attended = self._attend(piece, k, v)
                    parts.append(attended)
                    if collect:
                        with torch.no_grad():
                            weights = ((piece.detach() @ k.detach().transpose(-2, -1)) /
                                math.sqrt(piece.shape[-1])).softmax(-1)
                            e, p = _entropy_peak(weights)
                            entropy += e * piece.shape[2] / count
                            peak += p * piece.shape[2] / count
                context = torch.cat(parts, 2)
            merged = context.transpose(1, 2).reshape(batch, count, self.width)
            result = self.output(merged).transpose(1, 2).reshape(batch, self.width, height, width)
            stats = dict(attention_entropy=entropy, attention_peak=peak) if collect else {}
        return result, stats


class CrossTransformerBlock(nn.Module):
    """A complete pre-LN cross-attention residual plus 4x GELU FFN residual."""
    def __init__(self, width=128, heads=4):
        super().__init__()
        if width % heads:
            raise ValueError('Transformer width must be divisible by heads')
        self.width, self.heads = width, heads
        self.query_norm = nn.LayerNorm(width)
        self.source_norm = nn.LayerNorm(width)
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.output = nn.Linear(width, width)
        self.mlp_norm = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, 4*width), nn.GELU(), nn.Linear(4*width, width))

    def forward(self, tokens, source, uniform=False, no_mlp=False, collect=False):
        batch, count, _ = tokens.shape
        query, evidence = self.query_norm(tokens), self.source_norm(source)
        q = self.query(query).reshape(batch, count, self.heads, self.width//self.heads).transpose(1, 2)
        k, v = [projection(evidence).reshape(batch, source.shape[1], self.heads,
            self.width//self.heads).transpose(1, 2) for projection in (self.key, self.value)]
        if uniform:
            weights = q.new_full((batch, self.heads, count, source.shape[1]), 1./source.shape[1])
        else:
            weights = ((q @ k.transpose(-2, -1))/math.sqrt(q.shape[-1])).softmax(-1)
        context = (weights @ v).transpose(1, 2).reshape(batch, count, self.width)
        updated = tokens + self.output(context)
        if not no_mlp:
            updated = updated + self.mlp(self.mlp_norm(updated))
        stats = {}
        if collect:
            entropy, peak = _entropy_peak(weights)
            stats = dict(attention_entropy=entropy, attention_peak=peak)
        return updated, stats


class SpatialFollowup(nn.Module, AnalysisMixin):
    def __init__(self, encoder, variant, width=128, heads=4, query_chunk_size=64):
        super().__init__()
        if variant not in ('shallow64', 'cross16_mlp'):
            raise ValueError('Unknown spatial follow-up')
        self.encoder, self.variant = encoder, variant
        self.width, self.heads = width, heads
        self.analysis_mode = 'learned'
        self.analysis_modes = ('learned', 'uniform', 'removed') + (('no_mlp',) if variant == 'cross16_mlp' else ())
        self.gamma = nn.Parameter(torch.tensor(.1))
        if variant == 'shallow64':
            self.attention = ChunkedGlobalAttention(64, width, heads, query_chunk_size)
            self.output = nn.Conv2d(width, 2048, 1)
        else:
            self.query_norm = nn.LayerNorm(2048)
            self.source_norm = nn.LayerNorm(512)
            self.query_input = nn.Linear(2048, width)
            self.source_input = nn.Linear(512, width)
            self.blocks = nn.ModuleList([CrossTransformerBlock(width, heads) for _ in range(2)])
            self.output = nn.Linear(width, 2048)

    def _run(self, images, collect):
        middle, before = _cnn_features(self.encoder, images)
        stats = {}
        if self.analysis_mode == 'removed':
            after = before
        elif self.variant == 'shallow64':
            stem = self.encoder.stem
            # Reuse the very same learned CNN kernel, at stride1, without changing
            # the parent stride2 stem or introducing an independent shallow CNN.
            shallow = F.conv2d(images, stem.weight, stem.bias, stride=1,
                padding=stem.padding, dilation=stem.dilation, groups=stem.groups)
            if shallow.shape[-2:] != (64, 64):
                raise ValueError('Shallow all-QKV experiment requires true64x64 features')
            latent, stats = self.attention(shallow,
                uniform=self.analysis_mode == 'uniform', collect=collect)
            with torch.autocast(images.device.type, enabled=False):
                delta = self.output(F.adaptive_avg_pool2d(latent, before.shape[-2:]))
                after = (before.float()+self.gamma.float()*delta).to(before.dtype)
        else:
            with torch.autocast(images.device.type, enabled=False):
                initial = self.query_input(self.query_norm(before.float().flatten(2).transpose(1, 2)))
                source = self.source_input(self.source_norm(middle.float().flatten(2).transpose(1, 2)))
                tokens = initial
                layer_stats = []
                for block in self.blocks:
                    tokens, current = block(tokens, source,
                        uniform=self.analysis_mode == 'uniform',
                        no_mlp=self.analysis_mode == 'no_mlp', collect=collect)
                    if collect:
                        layer_stats.append(current)
                delta = self.output(tokens-initial).transpose(1, 2).reshape_as(before)
                after = (before.float()+self.gamma.float()*delta).to(before.dtype)
                if collect:
                    stats = {name: torch.stack([s[name] for s in layer_stats]).mean(0)
                        for name in ('attention_entropy', 'attention_peak')}
        logits = self.encoder.head(self.encoder.final_reduction(after))
        if not collect:
            return logits
        return dict(logits=logits, features_before=before, features_after=after,
            gamma=self.gamma.detach(), gate=None, retrieval_maps={}, **stats)

    def forward(self, images):
        if self.analysis_mode == 'removed':
            return self.encoder(images)
        return self._run(images, False)

    def forward_decomposed(self, images):
        return self._run(images, True)

    @property
    def attention_flops_per_image(self):
        if self.variant == 'cross16_mlp':
            return 2 * 4 * 64 * 256 * self.width
        # The shared-stem stride1 invocation uses F.conv2d rather than a module
        # call, so its additional multiply/adds must be included explicitly.
        stem = self.encoder.stem
        stem_flops = 2 * 64 * 64 * stem.out_channels * stem.in_channels * 3 * 3
        return 4 * 4096 * 4096 * self.width + stem_flops


def build_followup_model(variant, parent_state=None, vit_state=None, seed=42,
                         width=128, heads=4, query_chunk_size=64):
    if variant not in VARIANTS:
        raise ValueError(f'Unknown ViT follow-up variant: {variant}')
    if width < 4 or heads < 1 or width % heads:
        raise ValueError('Attention width must be divisible by positive heads')
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        if variant == 'vit':
            model = FollowupViT()
            if vit_state is not None:
                model.load_state_dict(vit_state, strict=True)
            return model
        encoder = build_spatial_model('cnn8', seed)
        if parent_state is not None:
            encoder.load_state_dict(parent_state, strict=True)
        if variant == 'dual_fusion':
            vit = FollowupViT()
            if vit_state is not None:
                vit.load_state_dict(vit_state, strict=True)
            model = DualFeatureFusion(encoder, vit)
            if {id(p) for p in model.encoder.parameters()} & {id(p) for p in model.vit.parameters()}:
                raise RuntimeError('CNN and ViT must not share parameters')
            return model
        return SpatialFollowup(encoder, variant, width, heads, query_chunk_size)


def description(variant):
    if variant not in VARIANTS:
        raise ValueError('Unknown ViT follow-up variant')
    common = dict(name=variant, input_size=[3, 64, 64], classes=200,
        all_parameters_trainable=True, external_pretraining=False,
        pretrained_CNN_parent=variant != 'vit', explicit_position_embedding=variant in ('vit', 'dual_fusion'))
    if variant in ('vit', 'dual_fusion'):
        common.update(vit=VIT_CONFIGURATION, vit_tokens=257, transformer_depth=5,
            transformer_mlp_ratio=4., vit_precision='full FP32',
            classifier=('CLS linear' if variant == 'vit' else 'concat CNN2048 GAP and ViT224 CLS -> Linear200'),
            independent_branch_parameters=variant == 'dual_fusion',
            removed_old_classification_heads=variant == 'dual_fusion',
            auxiliary_classification_loss=False)
    elif variant == 'shallow64':
        common.update(query_grid=[64, 64], key_value_grid=[64, 64],
            query_tokens=4096, key_value_tokens=4096, attention_depth=1,
            query_source='CNN first learned convolution kernel applied at stride1',
            key_value_source='same true64x64 shallow features as queries',
            original_CNN_stride2_path_preserved=True,
            attention_scope='exact global all-to-all, no token reduction before attention',
            memory_policy='query chunks with nonreentrant backward recomputation',
            pool_after_attention=[8, 8], classifier='original phase00 CNN head')
    else:
        common.update(query_grid=[8, 8], key_value_grid=[16, 16],
            attention_depth=2, transformer_width=128, transformer_heads=4,
            transformer_mlp_ratio=4., transformer_mlp_hidden_width=512,
            structure='two pre-LN cross-attention+MLP residual blocks in128D; project delta back to2048D',
            classifier='original phase00 CNN head',
            comparison_limit='relative to previous cross16, latent parameterization and depth also change; not an MLP-only ablation')
    return common
