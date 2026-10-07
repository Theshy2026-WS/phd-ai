"""Explicit attention MAC accounting on top of Conv/Linear estimates."""
from models.literature_alternet import Attention2d
from models.literature_resnet import BoTAttention
from utils.profiling import estimate_model_flops


def estimate_flops(model, device):
    extra = [0]

    def attention_macs(module, inputs, output):
        x = inputs[0]
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
               if isinstance(m, (Attention2d, BoTAttention))]
    try:
        base = estimate_model_flops(model, (3, 64, 64), device)
    finally:
        for handle in handles:
            handle.remove()
    return base + 2*extra[0]
