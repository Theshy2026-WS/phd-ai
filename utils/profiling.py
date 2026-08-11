import time

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


def benchmark_inference(
    model,
    input_shape,
    device,
    batch_size,
    warmup_iterations=20,
    benchmark_iterations=100,
    amp_enabled=False,
    amp_dtype=torch.float16,
):
    """Benchmark forward-only latency and throughput with synthetic inputs."""

    if warmup_iterations < 0 or benchmark_iterations <= 0:
        raise ValueError("Benchmark iterations must be positive")

    was_training = model.training
    model.eval()
    dummy_input = torch.randn(
        (batch_size, *input_shape),
        device=device,
    )
    device_type = str(device).split(":")[0]

    with torch.inference_mode():
        for _ in range(warmup_iterations):
            with torch.autocast(
                device_type=device_type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                model(dummy_input)

        if device_type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        start = time.perf_counter()
        for _ in range(benchmark_iterations):
            with torch.autocast(
                device_type=device_type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                model(dummy_input)
        if device_type == "cuda":
            torch.cuda.synchronize()
        elapsed_seconds = time.perf_counter() - start

    total_images = batch_size * benchmark_iterations
    latency_per_batch_ms = (
        elapsed_seconds / benchmark_iterations * 1000.0
    )
    latency_per_image_ms = elapsed_seconds / total_images * 1000.0
    throughput = total_images / elapsed_seconds
    peak_memory_bytes = (
        torch.cuda.max_memory_allocated()
        if device_type == "cuda"
        else 0
    )

    model.train(was_training)
    return {
        "latency_per_batch_ms": latency_per_batch_ms,
        "latency_per_image_ms": latency_per_image_ms,
        "throughput_images_per_second": throughput,
        "peak_inference_memory_bytes": peak_memory_bytes,
    }
