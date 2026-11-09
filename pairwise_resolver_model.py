"""Candidate-pair evidence retrieval on a fixed, complete CNN--ViT anchor.

These are controlled experimental models, not claims of architectural novelty.
The anchor's encoders *and* existing joint classifier stay fixed. Only a small
resolver is trained. The default source is the six-block intermediate ViT patch
representation; the original final CLS and CNN paths are preserved exactly.
"""
from contextlib import contextmanager
import math

import torch
from torch import nn
from torch.nn import functional as F

from models.vit_followup import _cnn_features, _cnn_pool


RESOLVER_VARIANTS = ('pairwise', 'static_query', 'mlp', 'mean_tokens')


def stable_candidates(logits, topk):
    """Descending logit order, resolving ties by ascending class index."""
    if logits.ndim != 2 or not 2 <= topk <= logits.shape[1]:
        raise ValueError('Need BxC logits and 2 <= topk <= C')
    return logits.argsort(dim=-1, descending=True, stable=True)[:, :topk]


class PairwiseResolverHead(nn.Module):
    """Shared antisymmetric comparisons followed by bounded candidate updates.

    Queries use a bias-free projection, split into heads, then per-head unit RMS
    normalization (L2 norm sqrt(head_dim)). The usual sqrt(head_dim) attention
    scaling therefore does not shrink an already unit-L2 query a second time.
    Keys/values are bias-free projections of raw block-six patch tokens. No new
    token positional embedding or LayerNorm is introduced.
    """
    def __init__(self, feature_dim=2432, token_dim=384, token_count=64,
                 variant='pairwise', topk=5, width=128, heads=4,
                 initial_eta=.1):
        super().__init__()
        if variant not in RESOLVER_VARIANTS:
            raise ValueError(f'Unknown resolver variant: {variant}')
        if min(feature_dim, token_dim, token_count, width, heads) < 1 or width % heads:
            raise ValueError('Positive dimensions and divisible width/heads required')
        if topk < 2 or not math.isfinite(initial_eta) or initial_eta <= 0:
            raise ValueError('Need topk >= 2 and a finite positive initial eta')
        self.variant, self.topk = variant, int(topk)
        self.last_statistics = {}
        self.feature_dim, self.token_dim, self.token_count = feature_dim, token_dim, token_count
        self.width, self.heads = width, heads
        self.register_buffer('train_patch_mean', torch.zeros(token_count, token_dim))
        self.register_buffer('train_patch_mean_ready', torch.tensor(False))
        self.raw_eta = nn.Parameter(torch.tensor(math.log(math.expm1(initial_eta))))
        left, right = torch.triu_indices(topk, topk, offset=1)
        self.register_buffer('pair_left', left, persistent=False)
        self.register_buffer('pair_right', right, persistent=False)
        if variant == 'mlp':
            attention_parameters = feature_dim*width + 2*token_dim*width + width + 1
            hidden = max(1, round((attention_parameters-1)/(feature_dim+token_dim+2)))
            self.mlp_hidden_width = hidden
            self.mlp = nn.Sequential(nn.Linear(feature_dim+token_dim, hidden),
                                     nn.GELU(), nn.Linear(hidden, 1, bias=False))
            nn.init.normal_(self.mlp[-1].weight, std=.02)
        else:
            self.query = nn.Linear(feature_dim, width, bias=False)
            self.key = nn.Linear(token_dim, width, bias=False)
            self.value = nn.Linear(token_dim, width, bias=False)
            self.output = nn.Linear(width, 1, bias=False)
            nn.init.normal_(self.output.weight, std=.02)

    @property
    def eta(self):
        return F.softplus(self.raw_eta)

    @torch.no_grad()
    def set_train_patch_mean(self, value):
        value = torch.as_tensor(value, device=self.train_patch_mean.device,
                                dtype=self.train_patch_mean.dtype)
        if value.shape != self.train_patch_mean.shape or not torch.isfinite(value).all():
            raise ValueError('Training patch mean must be finite with the exact token shape')
        self.train_patch_mean.copy_(value)
        self.train_patch_mean_ready.fill_(True)

    def signed_attention(self, evidence, patch_tokens, collect=False):
        """Return odd pair scores; reversing evidence reverses each score."""
        b, pairs, _ = evidence.shape
        n, per_head = patch_tokens.shape[1], self.width//self.heads
        q = self.query(evidence).reshape(b, pairs, self.heads, per_head).transpose(1, 2)
        q = F.normalize(q, dim=-1, eps=1e-12)*math.sqrt(per_head)
        k, v = [projection(patch_tokens).reshape(b, n, self.heads, per_head)
                .transpose(1, 2) for projection in (self.key, self.value)]
        logits = (q @ k.transpose(-2, -1))/math.sqrt(per_head)
        positive, negative = logits.softmax(-1), (-logits).softmax(-1)
        difference = positive-negative
        context = (difference @ v).transpose(1, 2).reshape(b, pairs, self.width)
        scores = self.output(context).squeeze(-1)
        maps = dict(positive=positive, negative=negative, difference=difference) if collect else {}
        if collect:
            with torch.no_grad():
                entropy = -.5*((positive*positive.clamp_min(1e-12).log()).sum(-1)
                              +(negative*negative.clamp_min(1e-12).log()).sum(-1))
                self.last_statistics = dict(
                    attention_entropy=entropy.mean((1, 2))/max(1., math.log(n)),
                    attention_peak=.5*(positive.amax(-1)+negative.amax(-1)).mean((1, 2)),
                    attention_logit_std=logits.std(-1, unbiased=False).mean((1, 2)),
                    signed_weight_l1=difference.abs().sum(-1).mean((1, 2)),
                    query_rms=q.square().mean((1, 2, 3)).sqrt(),
                    key_rms=k.square().mean((1, 2, 3)).sqrt(),
                    value_rms=v.square().mean((1, 2, 3)).sqrt())
        else:
            self.last_statistics = {}
        return scores, maps

    def pair_scores(self, evidence, patch_tokens, collect=False):
        if self.variant != 'mlp':
            return self.signed_attention(evidence, patch_tokens, collect)
        self.last_statistics = {}
        pooled = patch_tokens.mean(1, keepdim=True).expand(-1, evidence.shape[1], -1)
        forward = self.mlp(torch.cat((evidence, pooled), -1)).squeeze(-1)
        reverse = self.mlp(torch.cat((-evidence, pooled), -1)).squeeze(-1)
        return .5*(forward-reverse), {}

    def forward(self, features, patch_tokens, anchor_logits, classifier_weight, collect=False):
        expected = (features.shape[0], self.token_count, self.token_dim)
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError('Unexpected anchor feature dimensions')
        if patch_tokens.shape != expected or anchor_logits.ndim != 2:
            raise ValueError('Unexpected intermediate token or logit dimensions')
        if classifier_weight.shape != (anchor_logits.shape[1], self.feature_dim):
            raise ValueError('Unexpected anchor classifier dimensions')
        if not self.topk <= anchor_logits.shape[1]:
            raise ValueError('topk exceeds the number of classes')
        with torch.autocast(features.device.type, enabled=False):
            features, patch_tokens = features.float(), patch_tokens.float()
            anchor_logits, classifier_weight = anchor_logits.float(), classifier_weight.float()
            candidates = stable_candidates(anchor_logits, self.topk)
            weights = classifier_weight[candidates]
            difference = weights[:, self.pair_left]-weights[:, self.pair_right]
            evidence = difference if self.variant == 'static_query' else features[:, None]*difference
            used_tokens = patch_tokens
            if self.variant == 'mean_tokens':
                if not bool(self.train_patch_mean_ready):
                    raise RuntimeError('Fit the training-only patch mean before using mean_tokens')
                used_tokens = self.train_patch_mean[None].expand(features.shape[0], -1, -1)
            scores, maps = self.pair_scores(evidence, used_tokens, collect)
            pair_updates = self.eta/(self.topk-1)*scores.tanh()
            local = torch.zeros_like(anchor_logits[:, :self.topk])
            local.scatter_add_(1, self.pair_left[None].expand_as(pair_updates), pair_updates)
            local.scatter_add_(1, self.pair_right[None].expand_as(pair_updates), -pair_updates)
            delta = torch.zeros_like(anchor_logits).scatter(1, candidates, local)
            logits = anchor_logits+delta
            if not collect:
                return logits
            return dict(logits=logits, anchor_logits=anchor_logits, candidates=candidates,
                delta=delta, eta=self.eta, pair_scores=scores,
                pair_updates=pair_updates, pair_indices=torch.stack((self.pair_left,self.pair_right)),
                features=features, patch_tokens=patch_tokens, used_patch_tokens=used_tokens,
                correction_rms=delta.square().mean(-1).sqrt(),
                pair_score_rms=scores.square().mean(-1).sqrt(),
                retrieval_maps=maps, **self.last_statistics)


class PairwiseResolver(nn.Module):
    """Frozen complete concat anchor plus one trainable comparison resolver."""
    frozen_backbones = True
    analysis_modes = ('learned', 'anchor', 'mean_tokens', 'uniform')

    def __init__(self, anchor, variant='pairwise', topk=5, width=128, heads=4,
                 intermediate_block=6, initial_eta=.1):
        super().__init__()
        required = ('encoder', 'vit', 'fusion_classifier')
        if any(not hasattr(anchor, field) for field in required):
            raise ValueError('Need a complete CNN--ViT concat anchor')
        if getattr(anchor, 'variant', 'concat') != 'concat':
            raise ValueError('Only the existing unnormalized concat anchor is supported')
        classifier, vit = anchor.fusion_classifier, anchor.vit
        if not isinstance(classifier, nn.Linear) or not 1 <= intermediate_block <= len(vit.blocks):
            raise ValueError('Need a linear anchor classifier and valid one-based ViT block')
        self.anchor = anchor.requires_grad_(False)
        self.variant, self.intermediate_block = variant, intermediate_block
        self.analysis_mode = 'learned'
        self.resolver = PairwiseResolverHead(classifier.in_features, vit.embed_dim,
            vit.num_patches, variant, topk, width, heads, initial_eta)
        self.train()

    @property
    def eta(self):
        return self.resolver.eta

    @property
    def topk(self):
        return self.resolver.topk

    @property
    def train_patch_mean(self):
        return self.resolver.train_patch_mean

    def set_train_patch_mean(self, value):
        self.resolver.set_train_patch_mean(value)

    def train(self, mode=True):
        super().train(mode)
        self.anchor.eval()
        return self

    @contextmanager
    def analysis(self, mode):
        if mode not in self.analysis_modes:
            raise ValueError(f'Unknown analysis mode: {mode}')
        old = self.analysis_mode
        self.analysis_mode = mode
        try:
            yield self
        finally:
            self.analysis_mode = old

    @torch.no_grad()
    def extract_anchor(self, images):
        self.anchor.eval()
        _, deep = _cnn_features(self.anchor.encoder, images)
        cnn = _cnn_pool(self.anchor.encoder, deep)
        vit = self.anchor.vit
        expected = (vit.config['input_channels'], vit.config['image_size'], vit.config['image_size'])
        if images.ndim != 4 or tuple(images.shape[1:]) != expected:
            raise ValueError(f'Expected images shaped B,{expected}')
        with torch.autocast(images.device.type, enabled=False):
            patches = vit.patch_embedding(images.float()).flatten(2).transpose(1, 2)
            cls = vit.class_token.expand(images.shape[0], -1, -1)
            tokens = vit.embedding_dropout(torch.cat((cls, patches), 1)+vit.position_embedding)
            intermediate = None
            for i, block in enumerate(vit.blocks, 1):
                tokens = block(tokens)
                if i == self.intermediate_block:
                    intermediate = tokens[:, 1:].clone()
            final_cls = vit.norm(tokens)[:, 0]
            features = torch.cat((cnn.float(), final_cls), -1)
            logits = self.anchor.fusion_classifier(features)
        return dict(logits=logits, features=features, patch_tokens=intermediate)

    def forward_cached(self, features, patch_tokens, anchor_logits=None, collect=False):
        with torch.autocast(features.device.type, enabled=False):
            if anchor_logits is None:
                with torch.no_grad():
                    anchor_logits = self.anchor.fusion_classifier(features.float())
            if self.analysis_mode == 'anchor':
                if not collect:
                    return anchor_logits
                return dict(logits=anchor_logits, anchor_logits=anchor_logits,
                    candidates=stable_candidates(anchor_logits, self.topk),
                    delta=torch.zeros_like(anchor_logits), eta=self.eta,
                    features=features, patch_tokens=patch_tokens, retrieval_maps={})
            used_tokens = patch_tokens
            if self.analysis_mode == 'mean_tokens':
                if not bool(self.resolver.train_patch_mean_ready):
                    raise RuntimeError('Fit the training-only patch mean first')
                used_tokens = self.train_patch_mean[None].expand(features.shape[0], -1, -1)
            if self.analysis_mode == 'uniform':
                # Both signed weights become 1/N, so their difference is zero.
                if not collect:
                    return anchor_logits
                return dict(logits=anchor_logits, anchor_logits=anchor_logits,
                    candidates=stable_candidates(anchor_logits, self.topk),
                    delta=torch.zeros_like(anchor_logits), eta=self.eta,
                    features=features, patch_tokens=patch_tokens, retrieval_maps={})
            return self.resolver(features, used_tokens, anchor_logits,
                                 self.anchor.fusion_classifier.weight, collect)

    def forward(self, images, return_details=False):
        cached = self.extract_anchor(images)
        return self.forward_cached(cached['features'], cached['patch_tokens'],
                                   cached['logits'], collect=return_details)

    def forward_decomposed(self, images):
        result = self.forward(images, return_details=True)
        result.update(features_before=result['features'], features_after=result['features'],
                      gamma=self.eta, gate=None)
        return result

    @property
    def attention_flops_per_image(self):
        vit = self.anchor.vit
        base = 4*len(vit.blocks)*(vit.num_patches+1)**2*vit.embed_dim
        if self.variant == 'mlp':
            return base
        pairs = self.topk*(self.topk-1)//2
        # One QK product and one (A_plus-A_minus)V product. The negative
        # attention logits are reused by negation, not another QK matmul.
        return base+4*pairs*vit.num_patches*self.resolver.width
