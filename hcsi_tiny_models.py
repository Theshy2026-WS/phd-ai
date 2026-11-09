"""Controlled native64 adaptation of HCSI-Net's cross-stream gating mechanism.

This implements the published equations on the existing CNN8/plain-ViT pair,
not the author's EfficientNet/Swin/SBI benchmark. Four CNN stages align with
four equal groups of ViT blocks. Final patch mean pooling makes the last
Transformer-side gate participate in classification. Both encoders are trainable.

Cross-gate defaults follow the official implementation at commit
85944b39ee2e509f5b8d10b03eb95a82e9c17752: d64, beta_pos0.2, tau1,
bias-free projections, epsilon-added L2 normalization, and shared non-affine LN.
Attribution and the original MIT notice are in third_party/HCSI-Net-LICENSE.txt.
The plain ViT final LayerNorm precedes its final spatial gate; another LayerNorm
after that scalar gate would cancel its contribution to the mean-token readout.
"""
from contextlib import contextmanager
import math

import torch
from torch import nn
from torch.nn import functional as F

from models.vit_followup import _cnn_features, _cnn_pool


PAPER = 'https://lmi.fe.uni-lj.si/wp-content/uploads/2026/03/IWBF_2026_paper.pdf'
AUTHOR_COMMIT = '85944b39ee2e509f5b8d10b03eb95a82e9c17752'
AUTHOR_SOURCE = ('https://github.com/markobrodaric/HCSI-Net/blob/'
    +AUTHOR_COMMIT+'/models/architecture/HCSINet.py')


def _head(inputs, hidden, classes):
    if len(hidden) != 2 or min(inputs, *hidden, classes) < 1:
        raise ValueError('Need positive input, two hidden widths and classes')
    result = nn.Sequential(nn.Linear(inputs, hidden[0]), nn.BatchNorm1d(hidden[0]),
        nn.ReLU(), nn.Dropout(.5), nn.Linear(hidden[0], hidden[1]),
        nn.BatchNorm1d(hidden[1]), nn.ReLU(), nn.Dropout(.5),
        nn.Linear(hidden[1], classes))
    return result


def _l2_normalize(value):
    return value/(value.norm(dim=-1, keepdim=True)+1e-6)


def _coordinates(height, width, reference):
    yy, xx = torch.meshgrid(torch.linspace(-1., 1., height, device=reference.device,
        dtype=reference.dtype), torch.linspace(-1., 1., width, device=reference.device,
        dtype=reference.dtype), indexing='ij')
    return torch.stack((xx, yy), -1).reshape(1, height*width, 2)


class HCSICrossGate(nn.Module):
    """Equations3--14: descriptor alignment, bidirectional attention, sigmoid gates.

    CNN and Transformer node counts may differ. Only the learned descriptors
    exchange information; the original feature streams are multiplied by scalar
    spatial gates. Values are projected descriptors, not a third learned V map.
    """
    def __init__(self, cnn_channels, vit_channels, descriptor_dim=64,
                 beta_pos=.2, temperature=1.):
        super().__init__()
        if min(cnn_channels, vit_channels, descriptor_dim) < 1:
            raise ValueError('Descriptor and stream dimensions must be positive')
        if (not math.isfinite(beta_pos) or beta_pos < 0 or
                not math.isfinite(temperature) or temperature <= 0):
            raise ValueError('Need finite nonnegative positional weight and positive temperature')
        self.descriptor_dim = int(descriptor_dim)
        self.beta_pos, self.temperature = float(beta_pos), float(temperature)
        self.cnn_projection = nn.Conv2d(cnn_channels, descriptor_dim, 1, bias=False)
        self.vit_projection = nn.Linear(vit_channels, descriptor_dim, bias=False)
        self.cnn_position = nn.Linear(2, descriptor_dim, bias=False)
        self.vit_position = nn.Linear(2, descriptor_dim, bias=False)
        self.context_norm = nn.LayerNorm(descriptor_dim, eps=1e-5, elementwise_affine=False)

    def forward(self, cnn, patches, *, mode='learned', collect=False):
        if mode not in ('learned', 'uniform'):
            raise ValueError('Cross-gating mode must be learned or uniform')
        if cnn.ndim != 4 or patches.ndim != 3 or cnn.shape[0] != patches.shape[0]:
            raise ValueError('Expected paired CNN maps and patch token sequences')
        batch, _, height, width = cnn.shape
        side = math.isqrt(patches.shape[1])
        if side*side != patches.shape[1]:
            raise ValueError('The ViT patch layout must be a square grid, excluding CLS')
        original_dtype = cnn.dtype
        with torch.autocast(cnn.device.type, enabled=False):
            cnn_values, patch_values = cnn.float(), patches.float()
            dc = _l2_normalize(self.cnn_projection(cnn_values).flatten(2).transpose(1, 2))
            dt = _l2_normalize(self.vit_projection(patch_values))
            pc = _l2_normalize(self.cnn_position(_coordinates(height, width, cnn_values)))
            pt = _l2_normalize(self.vit_position(_coordinates(side, side, patch_values)))
            qc = _l2_normalize(dc+self.beta_pos*pc)
            qt = _l2_normalize(dt+self.beta_pos*pt)
            similarity = (qc @ qt.transpose(-1, -2))/(max(self.temperature, 1e-6)*math.sqrt(self.descriptor_dim))
            if mode == 'uniform':
                ct = torch.full_like(similarity, 1/patch_values.shape[1])
                tc = torch.full_like(similarity.transpose(-1, -2), 1/(height*width))
            else:
                ct = similarity.softmax(-1)
                tc = similarity.transpose(-1, -2).softmax(-1)
            vc = self.context_norm(ct @ qt)
            vt = self.context_norm(tc @ qc)
            gc = (qc*vc).sum(-1).sigmoid()
            gt = (qt*vt).sum(-1).sigmoid()
            # These are the article's raw multiplicative sigmoid gates. There is
            # intentionally no2*g rescaling, additive residual or identity blend.
            out_cnn = cnn_values*gc.reshape(batch, 1, height, width)
            out_patches = patch_values*gt.unsqueeze(-1)
            details = {}
            if collect:
                with torch.no_grad():
                    for prefix, gate in (('cnn', gc), ('vit', gt)):
                        details[prefix+'_gate_mean'] = gate.mean(-1).detach()
                        details[prefix+'_gate_min'] = gate.amin(-1).detach()
                        details[prefix+'_gate_max'] = gate.amax(-1).detach()
                    for prefix, weights in (('cnn', ct), ('vit', tc)):
                        entropy = -(weights*weights.clamp_min(1e-12).log()).sum(-1)
                        details[prefix+'_attention_entropy'] = (
                            entropy.mean(-1)/max(1., math.log(weights.shape[-1]))).detach()
                        details[prefix+'_attention_peak'] = weights.amax(-1).mean(-1).detach()
                    details['retrieval_maps'] = dict(cnn_to_vit=ct.detach(), vit_to_cnn=tc.detach(),
                        cnn_gate=gc.reshape(batch, height, width).detach(),
                        vit_gate=gt.reshape(batch, side, side).detach())
            return out_cnn.to(original_dtype), out_patches, details


class HCSITinyFusion(nn.Module):
    """Existing encoders with matched patch-mean/MLP heads, optionally cross-gated."""
    frozen_backbones = False
    analysis_modes = ('learned', 'no_interaction', 'uniform')

    def __init__(self, encoder, vit, variant='hcsi', descriptor_dim=64,
                 beta_pos=.2, temperature=1., hidden=(1024, 512),
                 cnn_channels=(256, 512, 1024, 2048)):
        super().__init__()
        if variant not in ('hcsi', 'late_mlp'):
            raise ValueError('Unknown HCSI Tiny comparison arm')
        old = encoder.head.dense
        if (not isinstance(old, nn.Linear) or len(encoder.stages) != 4 or
                len(cnn_channels) != 4 or cnn_channels[-1] != old.in_features):
            raise ValueError('Expected a four-stage CNN and its original linear classifier')
        if len(vit.blocks) < 4 or len(vit.blocks) % 4:
            raise ValueError('ViT depth must divide into four equal block groups')
        if old.out_features != vit.num_classes:
            raise ValueError('CNN and ViT class counts must agree')
        self.encoder, self.vit, self.variant = encoder, vit, variant
        self.cnn_channels, self.hidden = tuple(cnn_channels), tuple(hidden)
        self.descriptor_dim, self.beta_pos, self.temperature = descriptor_dim, beta_pos, temperature
        self.num_classes, self.vit_dim = old.out_features, vit.embed_dim
        self.encoder.head.dense = nn.Identity()
        self.vit.classifier = nn.Identity()
        # Construct and initialize the head BEFORE the optional gates. Paired
        # variants built under the same seed therefore get identical MLP weights.
        self.classifier = _head(old.in_features+vit.embed_dim, self.hidden, self.num_classes)
        self.interactions = nn.ModuleList([HCSICrossGate(channels, vit.embed_dim,
            descriptor_dim, beta_pos, temperature) for channels in cnn_channels]
            if variant == 'hcsi' else [])
        self.analysis_mode = 'learned'
        self.last_statistics = {}
        self.last_retrieval_maps = {}
        self.requires_grad_(True)

    @contextmanager
    def analysis(self, mode):
        if mode not in self.analysis_modes:
            raise ValueError('Unknown HCSI analysis mode')
        old = self.analysis_mode
        self.analysis_mode = mode
        try:
            yield self
        finally:
            self.analysis_mode = old

    def extract_features(self, images, collect=False):
        expected = (self.vit.config['input_channels'], self.vit.config['image_size'],
                    self.vit.config['image_size'])
        if images.ndim != 4 or tuple(images.shape[1:]) != expected:
            raise ValueError(f'Expected input B,{expected}')
        cnn = self.encoder.stem(images)
        self.last_statistics, self.last_retrieval_maps = {}, {}
        with torch.autocast(images.device.type, enabled=False):
            patches = self.vit.patch_embedding(images.float()).flatten(2).transpose(1, 2)
            cls = self.vit.class_token.expand(images.shape[0], -1, -1)
            tokens = self.vit.embedding_dropout(torch.cat((cls, patches), 1)+self.vit.position_embedding)
        group = len(self.vit.blocks)//4
        for stage, cnn_block in enumerate(self.encoder.stages):
            cnn = cnn_block(cnn)
            with torch.autocast(images.device.type, enabled=False):
                for block in self.vit.blocks[stage*group:(stage+1)*group]:
                    tokens = block(tokens)
                if stage == 3:
                    # The pretrained final normalization belongs to the ViT
                    # backbone and precedes its final spatial gate. Normalizing
                    # each token AFTER a scalar gate would cancel that gate.
                    tokens = self.vit.norm(tokens)
            if self.variant == 'hcsi' and self.analysis_mode != 'no_interaction':
                cnn, patches, details = self.interactions[stage](cnn, tokens[:, 1:],
                    mode='uniform' if self.analysis_mode == 'uniform' else 'learned', collect=collect)
                # CLS is retained unchanged at the intervention point; it still
                # participates in subsequent full Transformer blocks as usual.
                tokens = torch.cat((tokens[:, :1], patches), 1)
                if collect:
                    self.last_retrieval_maps[f'stage{stage+1}'] = details.pop('retrieval_maps')
                    self.last_statistics.update({f'stage{stage+1}_{key}': value
                        for key, value in details.items()})
        cnn_vector = _cnn_pool(self.encoder, cnn).float()
        with torch.autocast(images.device.type, enabled=False):
            # Mean of final normalized-and-then-gated patches. The last gate
            # affects the readout and is not canceled by another LayerNorm.
            vit_vector = tokens[:, 1:].mean(1)
        return cnn_vector, vit_vector

    def forward(self, images):
        cnn, vit = self.extract_features(images)
        with torch.autocast(images.device.type, enabled=False):
            return self.classifier(torch.cat((cnn, vit), -1).float())

    def forward_decomposed(self, images):
        cnn, vit = self.extract_features(images, collect=True)
        features = torch.cat((cnn, vit), -1)
        with torch.autocast(images.device.type, enabled=False):
            logits = self.classifier(features.float())
        gate = (torch.stack([v for k, v in self.last_statistics.items() if k.endswith('_gate_mean')]).mean(0)
            if self.last_statistics else None)
        return dict(logits=logits, cnn_features=cnn, vit_features=vit,
            features_before=cnn, features_after=features, gamma=logits.new_zeros(()),
            gate=gate, retrieval_maps=self.last_retrieval_maps, **self.last_statistics)

    @property
    def attention_flops_per_image(self):
        count = self.vit.attention_flops_per_image
        if self.variant == 'hcsi':
            image = self.vit.config['image_size']
            grids = (image//2, image//4, image//8, image//8)
            count += sum(8*grid*grid*self.vit.num_patches*self.descriptor_dim for grid in grids)
        return count

    def fusion_description(self):
        return dict(variant=self.variant, study_reference=PAPER, author_source=AUTHOR_SOURCE,
            author_commit=AUTHOR_COMMIT,
            scope='HCSI mechanism adaptation; not author EfficientNet/Swin/SBI benchmark reproduction',
            frozen_backbones=False, cnn_channels=list(self.cnn_channels),
            vit_block_groups=[len(self.vit.blocks)//4]*4, vit_tokens_exclude_cls_for_gating=True,
            pooling='CNN original preactivation/phase00/GAP; ViT final norm before last gate, then mean patch tokens',
            head=[self.cnn_channels[-1]+self.vit_dim, *self.hidden, self.num_classes],
            descriptor_dim=self.descriptor_dim, beta_pos=self.beta_pos, temperature=self.temperature,
            raw_sigmoid_multiplicative_gates=True, identity_residual=False,
            author_gate_defaults=dict(beta_pos=self.beta_pos, temperature=self.temperature,
                projection_bias=False, layer_norm_eps=1e-5, layer_norm_affine=False,
                l2_normalization='x/(norm(x)+1e-6)'),
            original_linear_classifier_inherited=False,
            head_initialization='PyTorch Linear defaults as author implementation; same seed/head in both fusion arms')


class CNNMLPControl(nn.Module):
    """Trainable source CNN with the same two hidden widths, without a ViT path."""
    variant = 'cnn_mlp'
    frozen_backbones = False
    analysis_modes = ('learned',)

    def __init__(self, encoder, hidden=(1024, 512)):
        super().__init__()
        old = encoder.head.dense
        if not isinstance(old, nn.Linear):
            raise ValueError('Need the source CNN original linear classifier')
        self.encoder = encoder
        self.num_classes, self.cnn_dim, self.hidden = old.out_features, old.in_features, tuple(hidden)
        self.encoder.head.dense = nn.Identity()
        self.classifier = _head(self.cnn_dim, self.hidden, self.num_classes)
        self.requires_grad_(True)

    def extract_features(self, images, collect=False):
        _, deep = _cnn_features(self.encoder, images)
        return _cnn_pool(self.encoder, deep).float()

    def forward(self, images):
        features = self.extract_features(images)
        with torch.autocast(images.device.type, enabled=False):
            return self.classifier(features.float())

    def forward_decomposed(self, images):
        features = self.extract_features(images)
        with torch.autocast(images.device.type, enabled=False):
            logits = self.classifier(features.float())
        return dict(logits=logits, cnn_features=features, features_before=features,
            features_after=features, gamma=logits.new_zeros(()), gate=None, retrieval_maps={})

    @property
    def attention_flops_per_image(self):
        return 0

    def fusion_description(self):
        return dict(variant=self.variant, frozen_backbones=False,
            pooling='original CNN preactivation/phase00/GAP',
            head=[self.cnn_dim, *self.hidden, self.num_classes],
            original_linear_classifier_inherited=False,
            head_initialization='PyTorch Linear defaults as author implementation')
