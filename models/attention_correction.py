"""Prediction-conditioned attention correction; an experimental side network.

Not a claimed novel/SOTA method. The original classifier is preserved exactly.
Attention retrieves spatial evidence using a query conditioned on base predictions.
"""
import hashlib

import torch
from torch import nn

from models.vit import ViT, TransformerBlock


def state_digest(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class FrozenAttentionCorrection(nn.Module):
    def __init__(self, base, width=192, heads=4, mode='attention'):
        super().__init__()
        if mode not in ('attention', 'pooled'):
            raise ValueError(mode)
        self.base = base.requires_grad_(False).eval()
        self.base_digest = state_digest(base)
        self.mode = mode
        self.projection = nn.Conv2d(1024, width, 1)
        self.position_embedding = nn.Parameter(torch.zeros(1, 64, width))
        nn.init.trunc_normal_(self.position_embedding, std=.02)
        self.norm = nn.LayerNorm(width)
        self.query = nn.Linear(200, width)
        self.query_embedding = nn.Parameter(torch.zeros(1, 1, width))
        if mode == 'attention':
            self.spatial = TransformerBlock(width, heads, mlp_ratio=2., dropout=0.)
            self.readout = nn.MultiheadAttention(width, heads, batch_first=True)
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width),
                                    nn.GELU(), nn.Linear(width, 200))
        # Identity at initialization. Earlier layers are nonzero so gradients
        # start reaching them after the first update of this output projection.
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()  # Freeze running means/variances as well as parameters.
        return self

    def verify_frozen(self):
        if state_digest(self.base) != self.base_digest:
            raise RuntimeError('Frozen CNN parameters or buffers changed')

    def forward(self, x):
        # FP32 base reproduces the saved classifier; no backbone autograd graph.
        with torch.no_grad(), torch.autocast(x.device.type, enabled=False):
            f = self.base.stem(x.float())
            for stage in self.base.stages[:3]:
                f = stage(f)
            z0 = self.base.head(self.base.stages[3](f))
        # Stable FP32 attention including QKV, not just softmax.
        with torch.autocast(x.device.type, enabled=False):
            tokens = self.projection(f).flatten(2).transpose(1, 2)
            if tokens.shape[1] != 64:
                raise ValueError('Expected native Tiny stage3 features: 8x8')
            tokens = self.norm(tokens + self.position_embedding)
            q = self.query(z0.softmax(-1)).unsqueeze(1) + self.query_embedding
            if self.mode == 'attention':
                tokens = self.spatial(tokens)
                evidence, _ = self.readout(q, tokens, tokens, need_weights=False)
                evidence = evidence[:, 0] + q[:, 0]
            else:
                evidence = tokens.mean(1) + q[:, 0]
            return z0 + self.output(evidence)


class TinyViTFP32(ViT):
    """Existing compact ViT structure, FP32 to avoid fp16 attention overflow."""
    def __init__(self):
        super().__init__(num_classes=200, image_size=64, patch_size=4,
                         embed_dim=224, depth=5, num_heads=4,
                         mlp_ratio=4., dropout=.1, attention_dropout=0.)

    def forward(self, x):
        with torch.autocast(x.device.type, enabled=False):
            return super().forward(x.float())
