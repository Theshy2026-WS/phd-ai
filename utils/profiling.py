import torch
import torch.nn as nn


def estimate_model_flops(model, input_shape, device):
    """Estimate one-sample forward FLOPs, using the convention 1 MAC = 2 FLOPs."""

    totals = {"macs": 0}
    handles = []
    mha_output_projections = {
        id(module.out_proj)
        for module in model.modules()
        if isinstance(module, nn.MultiheadAttention)
    }

    def convolution_hook(module, inputs, output):
        kernel_operations = 1
        for kernel_size in module.kernel_size:
            kernel_operations *= kernel_size
        kernel_operations *= module.in_channels // module.groups
        totals["macs"] += output.numel() * kernel_operations

    def linear_hook(module, inputs, output):
        totals["macs"] += output.numel() * module.in_features

    def attention_hook(module, inputs, output):
        query, key, value = inputs[:3]
        if module.batch_first:
            batch_size, query_length, embed_dim = query.shape
            key_length = key.shape[1]
            value_length = value.shape[1]
        else:
            query_length, batch_size, embed_dim = query.shape
            key_length = key.shape[0]
            value_length = value.shape[0]

        projection_macs = batch_size * embed_dim * embed_dim * (
            query_length + key_length + value_length + query_length
        )
        attention_macs = (
            2 * batch_size * query_length * key_length * embed_dim
        )
        totals["macs"] += projection_macs + attention_macs

    for module in model.modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d)):
            handles.append(module.register_forward_hook(convolution_hook))
        elif (
            isinstance(module, nn.Linear)
            and id(module) not in mha_output_projections
        ):
            handles.append(module.register_forward_hook(linear_hook))
        elif isinstance(module, nn.MultiheadAttention):
            handles.append(module.register_forward_hook(attention_hook))

    was_training = model.training
    model.eval()
    dummy_input = torch.zeros((1, *input_shape), device=device)
    try:
        with torch.no_grad():
            model(dummy_input)
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)

    return 2 * totals["macs"]


def format_large_number(value):
    for suffix, divisor in (("T", 1e12), ("G", 1e9), ("M", 1e6), ("K", 1e3)):
        if value >= divisor:
            return f"{value / divisor:.3f} {suffix}"
    return str(value)
