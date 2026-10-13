"""Explicit attention MAC accounting on top of Conv/Linear estimates."""
from models.literature_alternet import Attention2d
from models.literature_resnet import BoTAttention
from models.literature_coatnet import CoAtAttentionFP32
from models.literature_acmix import ACmix
from models.literature_conformer import ConformerAttentionFP32
from utils.profiling import estimate_model_flops


def estimate_flops(model, device):
    extra = [0]

    def attention_macs(module, inputs, output):
        x = inputs[0]
        if isinstance(module, ConformerAttentionFP32):
            b, n, c = x.shape
            extra[0] += 2*b*n*n*c
            return
        if isinstance(module, ACmix):
            b, c, h, w = x.shape
            n = (h // module.stride) * (w // module.stride)
            # q*(k + relative position) and attention*values. Position projection,
            # shared QKV, generated convolution features and grouped 3x3 are
            # Conv2d hooks. Relative subtractions, Unfold and softmax excluded.
            extra[0] += 2*b*c*n*module.window**2
            return
        if isinstance(module, CoAtAttentionFP32):
            # timm CoAtNet attention is NCHW; QKV/proj/MLP counted by
            # Conv/Linear hooks. Relative bias adds no dot products.
            b, _, h, w = x.shape
            n = h*w
            extra[0] += 2*b*n*n*module.inner.num_heads*module.inner.dim_head
            return
        b, _, h, w = x.shape
        n = h*w
        if isinstance(module, Attention2d):
            inner = module.to_q.out_channels
            # Author port fixes k=1; window partition already folded into b.
            if module.to_kv.stride != (1, 1):
                raise ValueError('FLOPs formula assumes AlterNet k=1')
            extra[0] += 2*b*n*n*inner  # QK and AV; projections counted as Conv2d.
        else:
            inner = module.heads * module.dim_head
            extra[0] += 2*b*n*n*inner
            extra[0] += 2*b*n*(2*module.size-1)*inner  # q*relative H/W tables

    handles = [m.register_forward_hook(attention_macs) for m in model.modules()
               if isinstance(m, (Attention2d, BoTAttention, CoAtAttentionFP32, ACmix, ConformerAttentionFP32))]
    try:
        base = estimate_model_flops(model, (3, 64, 64), device)
    finally:
        for handle in handles:
            handle.remove()
    return base + 2*extra[0]
