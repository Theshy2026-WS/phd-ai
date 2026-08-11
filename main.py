import yaml
import torch
import argparse
import os
import pandas as pd
import time
import random
import platform
import numpy as np

from datasets.mnist import get_mnist_loaders
from datasets.cifar10 import get_cifar10_loaders
from datasets.cifar100 import get_cifar100_loaders
from datasets.tiny_imagenet import get_tiny_imagenet_loaders

from models.cnn import CNN
from models.hybrid import CNNViTHybrid
from models.vit import ViT
from trainer.train import train, test
from utils.profiling import (
    benchmark_inference,
    estimate_model_flops,
    format_large_number,
)


def load_config(config_name):
    project_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = config_name

    if not os.path.isabs(config_path) and not os.path.exists(config_path):
        config_path = os.path.join(project_dir, "configs", config_name)

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config file not found: {config_path}"
        )

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def get_device():
    return (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


def synchronize_device(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def configure_runtime(training_config):
    seed = training_config.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    deterministic = training_config.get("deterministic", False)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic

    enable_tf32 = training_config.get("enable_tf32", True)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = enable_tf32
        torch.backends.cudnn.allow_tf32 = enable_tf32
    torch.set_float32_matmul_precision("high" if enable_tf32 else "highest")
    return seed


def configure_mixed_precision(training_config, device):
    enabled = training_config.get("mixed_precision", True)
    enabled = enabled and str(device).startswith("cuda")
    dtype_name = training_config.get("amp_dtype", "float16").lower()
    dtype_options = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_name not in dtype_options:
        raise ValueError(f"Unsupported AMP dtype: {dtype_name}")
    amp_dtype = dtype_options[dtype_name]
    scaler_enabled = enabled and amp_dtype == torch.float16
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
    return enabled, amp_dtype, dtype_name, scaler


def build_scheduler(optimizer, training_config):
    scheduler_config = training_config.get("scheduler")
    if not scheduler_config:
        return None

    scheduler_name = scheduler_config.get("name", "").lower()
    if scheduler_name != "cosine":
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    total_epochs = training_config["epochs"]
    warmup_epochs = scheduler_config.get("warmup_epochs", 0)
    min_lr = scheduler_config.get("min_lr", 0.0)

    if warmup_epochs < 0 or warmup_epochs >= total_epochs:
        raise ValueError("warmup_epochs must be between 0 and epochs - 1")

    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_epochs - warmup_epochs,
        eta_min=min_lr,
    )
    if warmup_epochs == 0:
        return cosine_scheduler

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=scheduler_config.get("warmup_start_factor", 0.1),
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )


def save_checkpoint(
    model,
    optimizer,
    epoch,
    test_acc,
    checkpoint_path,
    scheduler=None,
    scaler=None,
    config=None,
    evaluation_top5_acc=None,
):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "test_acc": test_acc,
        "evaluation_acc": test_acc,
        "evaluation_top5_acc": evaluation_top5_acc,
        "config": config,
    }
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None and scaler.is_enabled():
        checkpoint["scaler_state_dict"] = scaler.state_dict()
    torch.save(checkpoint, checkpoint_path)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "config",
        type=str,
        help="YAML config file name, e.g. mnist.yaml or cifar10.yaml"
    )

    args = parser.parse_args()

    config = load_config(args.config)

    device = get_device()
    seed = configure_runtime(config["training"])
    amp_enabled, amp_dtype, amp_dtype_name, scaler = (
        configure_mixed_precision(config["training"], device)
    )
    experiment_name = config.get(
        "experiment_name",
        f"{config['dataset']['name']}_{config['model']['name']}"
    )

    print("Using config:", args.config)
    print("Experiment:", experiment_name)
    print("Device:", device)
    print("PyTorch:", torch.__version__)
    print("Python:", platform.python_version())
    print("Seed:", seed)
    print(
        "Mixed precision:",
        f"{amp_dtype_name} AMP" if amp_enabled else "disabled",
    )
    if str(device).startswith("cuda"):
        print("GPU:", torch.cuda.get_device_name(0))

    dataset_name = config["dataset"]["name"]
    evaluation_split = config["dataset"].get("evaluation_split", "test")
    print("Evaluation split:", evaluation_split)

    if dataset_name == "mnist":
        train_loader, test_loader = get_mnist_loaders(
            batch_size=config["dataset"]["batch_size"],
            data_dir=config["dataset"]["data_dir"]
        )

    elif dataset_name == "cifar10":
        train_loader, test_loader = get_cifar10_loaders(
            batch_size=config["dataset"]["batch_size"],
            data_dir=config["dataset"]["data_dir"]
        )

    elif dataset_name == "cifar100":
        train_loader, test_loader = get_cifar100_loaders(
            batch_size=config["dataset"]["batch_size"],
            data_dir=config["dataset"]["data_dir"],
            download=config["dataset"].get("download", False)
        )

    elif dataset_name == "tiny_imagenet":
        train_loader, test_loader = get_tiny_imagenet_loaders(
            batch_size=config["dataset"]["batch_size"],
            data_dir=config["dataset"]["data_dir"],
            download=config["dataset"].get("download", False),
            evaluation_split=config["dataset"].get(
                "evaluation_split",
                "development",
            ),
            validation_fraction=config["dataset"].get(
                "validation_fraction",
                0.1,
            ),
            split_seed=config["dataset"].get("split_seed", 42),
            num_workers=config["dataset"].get("num_workers", 4),
        )

    else:
        raise ValueError(
            f"Unsupported dataset: {dataset_name}"
        )

    if config["model"]["name"] == "cnn":
        model_args = {
            key: value
            for key, value in config["model"].items()
            if key != "name"
        }
        model = CNN(
            input_channels=config["dataset"]["input_channels"],
            num_classes=config["dataset"]["num_classes"],
            **model_args
        ).to(device)

    elif config["model"]["name"] == "vit":
        model_args = {
            key: value
            for key, value in config["model"].items()
            if key != "name"
        }
        model = ViT(
            input_channels=config["dataset"]["input_channels"],
            num_classes=config["dataset"]["num_classes"],
            **model_args
        ).to(device)

    elif config["model"]["name"] == "hybrid":
        model_args = {
            key: value
            for key, value in config["model"].items()
            if key != "name"
        }
        model = CNNViTHybrid(
            input_channels=config["dataset"]["input_channels"],
            num_classes=config["dataset"]["num_classes"],
            **model_args
        ).to(device)

    else:
        raise ValueError(
            f"Unsupported model: {config['model']['name']}"
        )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    print(f"Trainable parameters: {trainable_parameters:,}")
    image_size = config["dataset"].get(
        "image_size",
        28 if dataset_name == "mnist" else 32,
    )
    estimated_flops = estimate_model_flops(
        model=model,
        input_shape=(
            config["dataset"]["input_channels"],
            image_size,
            image_size,
        ),
        device=device,
    )
    print(
        "Estimated forward FLOPs (batch=1, 1 MAC=2 FLOPs): "
        f"{format_large_number(estimated_flops)}"
    )
    input_shape = (
        config["dataset"]["input_channels"],
        image_size,
        image_size,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"].get("weight_decay", 0.0)
    )
    scheduler = build_scheduler(optimizer, config["training"])

    loss_fn = torch.nn.CrossEntropyLoss(
        label_smoothing=config["training"].get("label_smoothing", 0.0)
    )

    os.makedirs("results", exist_ok=True)
    os.makedirs(config["checkpoint"]["save_dir"], exist_ok=True)

    history = []

    best_test_acc = 0.0
    best_test_top5_acc = 0.0
    best_epoch = 0
    total_train_seconds = 0.0
    total_test_seconds = 0.0
    experiment_start = time.perf_counter()
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(1, config["training"]["epochs"] + 1):
        current_lr = optimizer.param_groups[0]["lr"]

        synchronize_device(device)
        train_start = time.perf_counter()
        train_acc = train(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            epoch,
            scaler=scaler,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            gradient_clip_norm=config["training"].get(
                "gradient_clip_norm"
            ),
        )
        synchronize_device(device)
        train_seconds = time.perf_counter() - train_start

        synchronize_device(device)
        test_start = time.perf_counter()
        test_acc, test_top5_acc = test(
            model,
            test_loader,
            device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )
        synchronize_device(device)
        test_seconds = time.perf_counter() - test_start
        epoch_seconds = train_seconds + test_seconds
        total_train_seconds += train_seconds
        total_test_seconds += test_seconds

        print(
            f"Epoch {epoch}: "
            f"Train Acc = {train_acc:.4f}, "
            f"Evaluation Acc = {test_acc:.4f}, "
            f"Evaluation Top-5 Acc = {test_top5_acc:.4f}, "
            f"LR = {current_lr:.6f}, "
            f"Train Time = {train_seconds:.2f}s, "
            f"Test Time = {test_seconds:.2f}s"
        )

        history.append({
            "epoch": epoch,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "test_top5_acc": test_top5_acc,
            "evaluation_split": evaluation_split,
            "evaluation_acc": test_acc,
            "evaluation_top5_acc": test_top5_acc,
            "learning_rate": current_lr,
            "train_seconds": train_seconds,
            "test_seconds": test_seconds,
            "epoch_seconds": epoch_seconds,
        })

        if (
            config["checkpoint"]["save_best"]
            and test_acc > best_test_acc
        ):
            best_test_acc = test_acc
            best_test_top5_acc = test_top5_acc
            best_epoch = epoch

            checkpoint_path = os.path.join(
                config["checkpoint"]["save_dir"],
                f"{experiment_name}_best.pt"
            )

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                test_acc=test_acc,
                checkpoint_path=checkpoint_path,
                scheduler=scheduler,
                scaler=scaler,
                config=config,
                evaluation_top5_acc=test_top5_acc,
            )

            print(
                f"Best checkpoint saved: {checkpoint_path} "
                f"with evaluation_acc={test_acc:.4f}"
            )

        if scheduler is not None:
            scheduler.step()

    result_df = pd.DataFrame(history)

    result_path = os.path.join(
        "results",
        f"{experiment_name}_results.csv"
    )

    result_df.to_csv(
        result_path,
        index=False
    )

    peak_training_memory_bytes = (
        torch.cuda.max_memory_allocated()
        if str(device).startswith("cuda")
        else 0
    )
    profiling_config = config.get("profiling", {})
    inference_metrics = {
        "latency_per_batch_ms": None,
        "latency_per_image_ms": None,
        "throughput_images_per_second": None,
        "peak_inference_memory_bytes": 0,
    }
    if profiling_config.get("enabled", True):
        inference_metrics = benchmark_inference(
            model=model,
            input_shape=input_shape,
            device=device,
            batch_size=profiling_config.get(
                "inference_batch_size",
                config["dataset"]["batch_size"],
            ),
            warmup_iterations=profiling_config.get("warmup_iterations", 20),
            benchmark_iterations=profiling_config.get(
                "benchmark_iterations", 100
            ),
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )

    experiment_seconds = time.perf_counter() - experiment_start
    gpu_name = (
        torch.cuda.get_device_name(0)
        if str(device).startswith("cuda")
        else "CPU"
    )
    summary = {
        "experiment_name": experiment_name,
        "dataset": dataset_name,
        "model": config["model"]["name"],
        "evaluation_split": evaluation_split,
        "seed": seed,
        "epochs": config["training"]["epochs"],
        "batch_size": config["dataset"]["batch_size"],
        "image_size": image_size,
        "trainable_parameters": trainable_parameters,
        "estimated_forward_flops": estimated_flops,
        "best_epoch": best_epoch,
        "best_evaluation_acc": best_test_acc,
        "best_evaluation_top5_acc": best_test_top5_acc,
        "total_train_seconds": total_train_seconds,
        "total_evaluation_seconds": total_test_seconds,
        "total_experiment_seconds": experiment_seconds,
        "peak_training_memory_mb": peak_training_memory_bytes / 1024 ** 2,
        "inference_latency_batch_ms": inference_metrics[
            "latency_per_batch_ms"
        ],
        "inference_latency_image_ms": inference_metrics[
            "latency_per_image_ms"
        ],
        "inference_throughput_images_per_second": inference_metrics[
            "throughput_images_per_second"
        ],
        "peak_inference_memory_mb": inference_metrics[
            "peak_inference_memory_bytes"
        ] / 1024 ** 2,
        "mixed_precision": amp_enabled,
        "amp_dtype": amp_dtype_name if amp_enabled else "disabled",
        "pytorch_version": torch.__version__,
        "python_version": platform.python_version(),
        "device": gpu_name,
    }
    summary_path = os.path.join(
        "results",
        f"{experiment_name}_summary.csv",
    )
    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    print(f"Results saved to {result_path}")
    print(f"Summary saved to {summary_path}")
    print(f"Best Evaluation Acc: {best_test_acc:.4f}")
    print(f"Best Evaluation Top-5 Acc: {best_test_top5_acc:.4f}")
    print(f"Total Train Time: {total_train_seconds:.2f}s")
    print(f"Total Evaluation Time: {total_test_seconds:.2f}s")
    print(f"Total Experiment Time: {experiment_seconds:.2f}s")
    print(
        "Peak Training GPU Memory: "
        f"{peak_training_memory_bytes / 1024 ** 2:.2f} MB"
    )
    if inference_metrics["throughput_images_per_second"] is not None:
        print(
            "Inference: "
            f"{inference_metrics['throughput_images_per_second']:.2f} images/s, "
            f"{inference_metrics['latency_per_image_ms']:.4f} ms/image"
        )


if __name__ == "__main__":
    main()
