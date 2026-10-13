"""Conformer-Ti/16 author-derived baseline at native 64x64 resolution."""
import torch
from torch import nn
from models.conformer_author import Conformer, Attention


class ConformerAttentionFP32(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        with torch.autocast(x.device.type, enabled=False):
            return self.inner(x.float() if x.dtype in (torch.float16, torch.bfloat16) else x)


class TinyConformer(Conformer):
    def __init__(self):
        # Author models.py Conformer_tiny_patch16 parameters. Keep native stem,
        # FCUs, CLS token and both heads; only class count/input resolution change.
        super().__init__(patch_size=16, channel_ratio=1, embed_dim=384, depth=12,
                         num_heads=6, mlp_ratio=4, qkv_bias=True, num_classes=200,
                         drop_rate=0., attn_drop_rate=0., drop_path_rate=0.)
        def protect(module):
            for name, child in list(module.named_children()):
                if isinstance(child, Attention):
                    setattr(module, name, ConformerAttentionFP32(child))
                else:
                    protect(child)
        protect(self)

    def forward(self, x):
        heads = super().forward(x)
        # Author engine.py: mean of two losses during training, summed logits
        # for inference Top-1. Do NOT train using loss(sum(heads)).
        return heads if self.training else heads[0] + heads[1]


def conformer_description():
    return dict(name='conformer_tiny', input_size=[3, 64, 64], classes=200, pretrained=False,
                source='pengzhiliang/Conformer@815aaad3ef5dbdfcf1e11368891416c2d7478cb1, Conformer_tiny_patch16',
                adaptation=True, stem='7x7 stride 2 + maxpool stride 2',
                cnn_spatial_sizes=[16, 8, 4, 2], cnn_channels=[64, 128, 256, 256],
                transformer_grid=[4, 4], transformer_tokens=17, includes_cls_token=True,
                transformer_depth=12, transformer_width=384, attention_heads=6,
                attention_blocks=12, coupling_units=11,
                attention_precision='fp32 island', control=None,
                training_objective='mean soft-target cross entropy of CNN and Transformer heads',
                prediction='sum of both head logits; NLL computed on summed logits',
                comparison_role='whole-model dual-branch reference, not a paired ResNet ablation')
