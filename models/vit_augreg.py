"""Native64, from-scratch AugReg ViTs and controlled CNN/ViT fusion models.

The profile names describe experimental candidates, not pretrained weights or
claimed new architectures. Attention arithmetic stays in FP32; SDPA avoids
retaining an explicitly materialized attention matrix during ordinary forwards.
"""
from contextlib import contextmanager
import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from models.vit_followup import _cnn_features, _cnn_pool


_BASE = dict(input_channels=3, num_classes=200, image_size=64, mlp_ratio=4.,
             attention_dropout=0., norm_eps=1e-6)
PROFILES = {
    'vit_tiny_p4_aug': dict(_BASE, patch_size=4, embed_dim=192, depth=12,
                           num_heads=3, dropout=0., drop_path=0.),
    'vit_small_p8_aug': dict(_BASE, patch_size=8, embed_dim=384, depth=12,
                            num_heads=6, dropout=0., drop_path=0.),
    'vit_small_p8_augreg': dict(_BASE, patch_size=8, embed_dim=384, depth=12,
                               num_heads=6, dropout=.1, drop_path=.1),
}
FUSION_VARIANTS = ('concat', 'normalized', 'residual', 'residual_linear',
                   'cross_patch', 'mlp_patch')


class AnalysisMixin:
    @contextmanager
    def analysis(self, mode):
        if mode not in self.analysis_modes:
            raise ValueError(f'Unsupported analysis mode: {mode}')
        previous = self.analysis_mode
        self.analysis_mode = mode
        try:
            yield self
        finally:
            self.analysis_mode = previous


class DropPath(nn.Module):
    """Per-example stochastic depth; expectation is unchanged during training."""
    def __init__(self, probability=0.):
        super().__init__()
        if not 0 <= probability < 1:
            raise ValueError('Drop-path probability must be in [0,1)')
        self.probability = float(probability)

    def forward(self, x):
        if not self.training or self.probability == 0:
            return x
        keep = 1-self.probability
        mask = (torch.rand((x.shape[0],)+(1,)*(x.ndim-1),
                           device=x.device) < keep).to(x.dtype)
        return x*mask/keep


class FP32SelfAttention(nn.Module):
    def __init__(self, width, heads, attention_dropout=0., projection_dropout=0.):
        super().__init__()
        if width % heads:
            raise ValueError('Attention width must be divisible by heads')
        self.width, self.heads = width, heads
        self.attention_dropout = float(attention_dropout)
        self.qkv = nn.Linear(width, 3*width)
        self.output = nn.Linear(width, width)
        self.dropout = nn.Dropout(projection_dropout)
        self.last_statistics = {}

    def forward(self, x, collect=False):
        with torch.autocast(x.device.type, enabled=False):
            x = x.float()
            b, n, d = x.shape
            q, k, v = [part.reshape(b, n, self.heads, d//self.heads).transpose(1, 2)
                       for part in self.qkv(x).chunk(3, dim=-1)]
            context = F.scaled_dot_product_attention(q, k, v,
                dropout_p=self.attention_dropout if self.training else 0.)
            if collect:
                with torch.no_grad():
                    weights = (q.detach() @ k.detach().transpose(-2, -1)
                               / math.sqrt(q.shape[-1])).softmax(-1)
                    entropy = -(weights*weights.clamp_min(1e-12).log()).sum(-1)
                    self.last_statistics = dict(
                        attention_entropy=entropy.mean((1, 2))/max(1., math.log(n)),
                        attention_peak=weights.amax(-1).mean((1, 2)))
            else:
                self.last_statistics = {}
            return self.dropout(self.output(context.transpose(1, 2).reshape(b, n, d)))


class AugRegBlock(nn.Module):
    def __init__(self, width, heads, mlp_ratio=4., dropout=0.,
                 attention_dropout=0., drop_path=0., norm_eps=1e-6):
        super().__init__()
        self.norm1 = nn.LayerNorm(width, eps=norm_eps)
        self.attention = FP32SelfAttention(width, heads, attention_dropout, dropout)
        self.norm2 = nn.LayerNorm(width, eps=norm_eps)
        hidden = int(width*mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(width, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, width), nn.Dropout(dropout))
        self.drop_path1, self.drop_path2 = DropPath(drop_path), DropPath(drop_path)

    def forward(self, tokens, collect=False):
        tokens = tokens+self.drop_path1(self.attention(self.norm1(tokens), collect))
        return tokens+self.drop_path2(self.mlp(self.norm2(tokens)))


class ViTAugReg(nn.Module, AnalysisMixin):
    variant = 'vit_augreg'
    analysis_modes = ('learned',)

    def __init__(self, config):
        super().__init__()
        self.config = copy.deepcopy(dict(config))
        c = self.config
        for field in ('image_size', 'patch_size', 'embed_dim', 'depth',
                      'num_heads', 'num_classes', 'input_channels'):
            if not isinstance(c[field], int) or c[field] <= 0:
                raise ValueError(f'Invalid positive integer: {field}')
        if c['image_size'] % c['patch_size'] or c['embed_dim'] % c['num_heads']:
            raise ValueError('Image/patch and width/head dimensions must divide exactly')
        if any(not 0 <= c.get(key, 0.) < 1 for key in
               ('dropout', 'attention_dropout', 'drop_path')):
            raise ValueError('Dropout probabilities must be in [0,1)')
        if c.get('mlp_ratio', 4.) <= 0 or c.get('norm_eps', 1e-6) <= 0:
            raise ValueError('MLP ratio and norm epsilon must be positive')
        self.embed_dim = c['embed_dim']
        self.num_classes = c['num_classes']
        self.num_patches = (c['image_size']//c['patch_size'])**2
        self.analysis_mode = 'learned'
        self.patch_embedding = nn.Conv2d(c['input_channels'], self.embed_dim,
            kernel_size=c['patch_size'], stride=c['patch_size'])
        self.class_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.position_embedding = nn.Parameter(torch.zeros(1, self.num_patches+1, self.embed_dim))
        self.embedding_dropout = nn.Dropout(c.get('dropout', 0.))
        rates = [c.get('drop_path', 0.)*i/max(1, c['depth']-1)
                 for i in range(c['depth'])]
        self.blocks = nn.ModuleList([AugRegBlock(self.embed_dim, c['num_heads'],
            c.get('mlp_ratio', 4.), c.get('dropout', 0.),
            c.get('attention_dropout', 0.), rate, c.get('norm_eps', 1e-6))
            for rate in rates])
        self.norm = nn.LayerNorm(self.embed_dim, eps=c.get('norm_eps', 1e-6))
        self.classifier = nn.Linear(self.embed_dim, self.num_classes)
        self.apply(self._initialize)
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        nn.init.zeros_(self.patch_embedding.bias)
        nn.init.trunc_normal_(self.class_token, std=.02)
        nn.init.trunc_normal_(self.position_embedding, std=.02)

    @staticmethod
    def _initialize(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            if module.elementwise_affine:
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _forward_tokens(self, images, collect=False):
        expected = (self.config['input_channels'], self.config['image_size'],
                    self.config['image_size'])
        if images.ndim != 4 or tuple(images.shape[1:]) != expected:
            raise ValueError(f'Expected images shaped B,{expected}')
        with torch.autocast(images.device.type, enabled=False):
            patches = self.patch_embedding(images.float()).flatten(2).transpose(1, 2)
            cls = self.class_token.expand(images.shape[0], -1, -1)
            tokens = self.embedding_dropout(torch.cat((cls, patches), 1)+self.position_embedding)
            for i, block in enumerate(self.blocks):
                tokens = block(tokens, collect and i == len(self.blocks)-1)
            return self.norm(tokens)

    def forward_tokens(self, images):
        return self._forward_tokens(images)

    def forward_features(self, images):
        return self.forward_tokens(images)[:, 0]

    def forward(self, images):
        with torch.autocast(images.device.type, enabled=False):
            return self.classifier(self.forward_features(images))

    def forward_decomposed(self, images):
        tokens = self._forward_tokens(images, True)
        with torch.autocast(images.device.type, enabled=False):
            logits = self.classifier(tokens[:, 0])
        return dict(logits=logits, features_before=tokens[:, 0],
            features_after=tokens[:, 0], vit_features=tokens[:, 0],
            gamma=logits.new_zeros(()), gate=None, retrieval_maps={},
            **self.blocks[-1].attention.last_statistics)

    @property
    def attention_flops_per_image(self):
        # Projection Linear hooks count QKV/output; add only QK/AV matmuls.
        return 4*len(self.blocks)*(self.num_patches+1)**2*self.embed_dim


def build_vit(profile, seed=42, config=None):
    if profile not in PROFILES:
        raise ValueError(f'Unknown ViT AugReg profile: {profile}')
    architecture = copy.deepcopy(PROFILES[profile])
    if config is not None:
        architecture.update(config)
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        model = ViTAugReg(architecture)
    model.profile = profile
    return model


def description(profile):
    if profile not in PROFILES:
        raise ValueError(f'Unknown ViT AugReg profile: {profile}')
    return dict(name=profile, architecture=copy.deepcopy(PROFILES[profile]),
        input_size=[3, 64, 64], classes=200, external_pretraining=False,
        representation='learned CLS after full pre-LN attention+MLP stack',
        attention_precision='FP32 SDPA',
        patch_tokens=(64//PROFILES[profile]['patch_size'])**2,
        study_reference='https://arxiv.org/abs/2106.10270',
        interpretation='Native64 adaptation of AugReg principles; not an exact reproduction of ImageNet experiments')


class AugRegFusion(nn.Module, AnalysisMixin):
    """Existing CNN representation plus a selected independently trained ViT.

    ``normalized`` changes only the concat inputs to parameterless LayerNorm;
    inherited head weights therefore do not preserve epoch-zero CNN predictions.
    Anchored variants retain a fixed original CNN classifier and learn a bounded
    correction. These are controlled candidate designs, not novelty claims.
    """
    def __init__(self, encoder, vit, variant='concat', frozen=True, width=128,
                 heads=4, residual_bound=1., initial_scale=.1):
        super().__init__()
        if variant not in FUSION_VARIANTS or width < 1 or width % heads:
            raise ValueError('Invalid fusion variant or latent attention dimensions')
        if not 0 < initial_scale < residual_bound:
            raise ValueError('Need 0 < initial residual scale < its bound')
        old = encoder.head.dense
        if not isinstance(old, nn.Linear) or old.in_features != 2048:
            raise ValueError('Expected the original 2048D CNN classifier')
        self.encoder, self.vit, self.variant = encoder, vit, variant
        self.vit_dim = vit.embed_dim
        self.num_classes = old.out_features
        self.frozen_backbones = bool(frozen)
        self.analysis_mode = 'learned'
        self.width, self.heads = width, heads
        self.residual_bound = float(residual_bound)
        self.initial_correction_scale = float(initial_scale)
        self.cnn_norm = nn.LayerNorm(2048, eps=1e-6, elementwise_affine=False)
        self.vit_norm = nn.LayerNorm(self.vit_dim, eps=1e-6, elementwise_affine=False)
        self.encoder.head.dense = nn.Identity()
        self.vit.classifier = nn.Identity()
        modes = ['learned', 'cnn_only', 'vit_train_mean', 'vit_mismatch']
        if variant in ('concat', 'normalized'):
            self.fusion_classifier = nn.Linear(2048+self.vit_dim, self.num_classes)
            with torch.no_grad():
                self.fusion_classifier.weight[:, :2048].copy_(old.weight)
                nn.init.normal_(self.fusion_classifier.weight[:, 2048:], std=.001)
                self.fusion_classifier.bias.copy_(old.bias)
            modes.append('vit_only')
        else:
            self.cnn_classifier = old
            self.cnn_classifier.requires_grad_(False)
            fraction = initial_scale/residual_bound
            self.scale_logit = nn.Parameter(torch.tensor(math.log(fraction/(1-fraction))))
            if variant == 'residual':
                self.correction = nn.Sequential(nn.Linear(2048+self.vit_dim, width),
                    nn.GELU(), nn.Linear(width, self.num_classes))
            elif variant == 'residual_linear':
                self.correction = nn.Linear(2048+self.vit_dim, self.num_classes)
            elif variant == 'cross_patch':
                self.query = nn.Linear(2048, width)
                self.key = nn.Linear(self.vit_dim, width)
                self.value = nn.Linear(self.vit_dim, width)
                self.output = nn.Linear(width, width)
                self.correction = nn.Linear(width, self.num_classes)
                modes.append('uniform')
            else:
                # Match the cross-attention branch's trainable parameter budget
                # approximately with an actual pooled-feature MLP. No dormant
                # projections or padding parameters inflate the reported count.
                target = width*(2048+2*self.vit_dim+width+self.num_classes)+4*width+self.num_classes
                hidden = max(1, round((target-self.num_classes)/
                    (2048+self.vit_dim+self.num_classes+1)))
                self.mlp_hidden_width = hidden
                self.cross_reference_parameters = target
                self.correction = nn.Sequential(nn.Linear(2048+self.vit_dim, hidden),
                    nn.GELU(), nn.Linear(hidden, self.num_classes))
            # Small but nonzero outputs keep gradients flowing into new modules.
            if isinstance(self.correction, nn.Sequential):
                self.correction.apply(ViTAugReg._initialize)
                nn.init.normal_(self.correction[-1].weight, std=.001)
            else:
                nn.init.normal_(self.correction.weight, std=.001)
                nn.init.zeros_(self.correction.bias)
        self.analysis_modes = tuple(modes)
        self.last_attention_weights = None
        self.set_backbones_frozen(frozen)

    def set_backbones_frozen(self, frozen):
        self.frozen_backbones = bool(frozen)
        self.encoder.requires_grad_(not frozen)
        self.vit.requires_grad_(not frozen)
        if hasattr(self, 'cnn_classifier'):
            self.cnn_classifier.requires_grad_(False)
        self.train(self.training)
        return self

    def train(self, mode=True):
        super().train(mode)
        if self.frozen_backbones:
            self.encoder.eval()
            self.vit.eval()
        if hasattr(self, 'cnn_classifier'):
            self.cnn_classifier.eval()
        return self

    @property
    def correction_scale(self):
        if not hasattr(self, 'scale_logit'):
            return None
        return self.residual_bound*self.scale_logit.sigmoid()

    def extract_features(self, images):
        _, deep = _cnn_features(self.encoder, images)
        cnn = _cnn_pool(self.encoder, deep).float()
        tokens = self.vit.forward_tokens(images)
        return cnn, tokens[:, 0], tokens[:, 1:]

    def _inputs(self, cnn, cls, patches, mode, mean_feature, mean_patches):
        if mode not in self.analysis_modes:
            raise ValueError(f'Unsupported fusion mode: {mode}')
        if cnn.ndim != 2 or cnn.shape[1] != 2048 or cls.shape != (cnn.shape[0], self.vit_dim):
            raise ValueError('CNN/ViT feature layouts differ')
        if patches.ndim != 3 or patches.shape[0] != cnn.shape[0] or patches.shape[-1] != self.vit_dim:
            raise ValueError('Invalid ViT patch-feature layout')
        if mode == 'vit_train_mean':
            if mean_feature is None or tuple(mean_feature.shape) != (self.vit_dim,):
                raise ValueError('A training-only CLS mean is required')
            cls = mean_feature.to(cls).expand_as(cls)
            if self.variant in ('cross_patch', 'mlp_patch'):
                if mean_patches is None or tuple(mean_patches.shape) != tuple(patches.shape[1:]):
                    raise ValueError('A training-only positional patch mean is required')
                patches = mean_patches.to(patches).unsqueeze(0).expand_as(patches)
        elif mode == 'vit_mismatch':
            if cnn.shape[0] < 2:
                raise ValueError('Cross-image mismatch needs at least two examples')
            cls, patches = cls.roll(1, 0), patches.roll(1, 0)
        elif mode == 'cnn_only' and self.variant in ('concat', 'normalized'):
            cls = torch.zeros_like(cls)
        elif mode == 'vit_only':
            cnn = torch.zeros_like(cnn)
        return cnn.float(), cls.float(), patches.float()

    def forward_from_features(self, cnn, cls, patches, mode='learned', *,
                              mean_feature=None, mean_patches=None):
        cnn, cls, patches = self._inputs(cnn, cls, patches, mode, mean_feature, mean_patches)
        with torch.autocast(cnn.device.type, enabled=False):
            if self.variant in ('concat', 'normalized'):
                c, v = (self.cnn_norm(cnn), self.vit_norm(cls)) if self.variant == 'normalized' else (cnn, cls)
                return self.fusion_classifier(torch.cat((c, v), -1))
            base = self.cnn_classifier(cnn)
            if mode == 'cnn_only':
                return base
            if self.variant in ('residual', 'residual_linear'):
                delta = self.correction(torch.cat((self.cnn_norm(cnn), self.vit_norm(cls)), -1))
            elif self.variant == 'mlp_patch':
                source = self.vit_norm(patches).mean(1)
                delta = self.correction(torch.cat((self.cnn_norm(cnn), source), -1))
            else:
                b, n, _ = patches.shape
                q = self.query(self.cnn_norm(cnn)).reshape(b, 1, self.heads, self.width//self.heads).transpose(1, 2)
                source = self.vit_norm(patches)
                k = self.key(source).reshape(b, n, self.heads, self.width//self.heads).transpose(1, 2)
                v = self.value(source).reshape(b, n, self.heads, self.width//self.heads).transpose(1, 2)
                if mode == 'uniform':
                    context = v.mean(2, keepdim=True)
                else:
                    context = F.scaled_dot_product_attention(q, k, v)
                delta = self.correction(self.output(context.transpose(1, 2).reshape(b, self.width)))
            return base+self.correction_scale*delta.tanh()

    def forward(self, images):
        return self.forward_from_features(*self.extract_features(images), mode=self.analysis_mode)

    def forward_decomposed(self, images):
        cnn, cls, patches = self.extract_features(images)
        logits = self.forward_from_features(cnn, cls, patches, mode=self.analysis_mode)
        result = dict(logits=logits, features_before=cnn, features_after=torch.cat((cnn, cls), -1),
            cnn_features=cnn, vit_features=cls, gamma=logits.new_zeros(()), gate=None, retrieval_maps={})
        if self.correction_scale is not None:
            result['gamma'] = self.correction_scale
        return result

    @property
    def attention_flops_per_image(self):
        count = self.vit.attention_flops_per_image
        if self.variant == 'cross_patch':
            count += 4*self.vit.num_patches*self.width
        return count

    def fusion_description(self):
        return dict(variant=self.variant, frozen_backbones=self.frozen_backbones,
            cnn_features=2048, vit_features=self.vit_dim, vit_patch_tokens=self.vit.num_patches,
            latent_width=self.width, heads=self.heads,
            original_cnn_anchor=self.variant not in ('concat', 'normalized'),
            normalized_initial_prediction_preserved=False if self.variant == 'normalized' else None,
            residual_logit_bound=self.residual_bound if hasattr(self, 'scale_logit') else None,
            initial_correction_scale=self.initial_correction_scale if hasattr(self, 'scale_logit') else None,
            mlp_hidden_width=getattr(self, 'mlp_hidden_width', None),
            cross_reference_parameters=getattr(self, 'cross_reference_parameters', None),
            mismatch_policy='roll complete ViT representations by one image within each deterministic batch',
            patch_control='MLP of CNN feature and mean-pooled ViT patches; approximately matched trainable parameters, not FLOPs'
                if self.variant == 'mlp_patch' else None,
            interpretation='Experimental fusion candidate; architectural or method novelty is not asserted')
