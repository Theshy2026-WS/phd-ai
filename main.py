import yaml
import torch
import argparse
import os
import pandas as pd

from datasets.mnist import get_mnist_loaders
from datasets.cifar10 import get_cifar10_loaders

from models.cnn import CNN
from models.hybrid import CNNViTHybrid
from models.vit import ViT
from trainer.train import train, test


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
    checkpoint_path
):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "test_acc": test_acc
        },
        checkpoint_path
    )


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
    experiment_name = config.get(
        "experiment_name",
        f"{config['dataset']['name']}_{config['model']['name']}"
    )

    print("Using config:", args.config)
    print("Experiment:", experiment_name)
    print("Device:", device)

    dataset_name = config["dataset"]["name"]

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

    for epoch in range(1, config["training"]["epochs"] + 1):
        current_lr = optimizer.param_groups[0]["lr"]

        train_acc = train(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            epoch
        )

        test_acc = test(
            model,
            test_loader,
            device
        )

        print(
            f"Epoch {epoch}: "
            f"Train Acc = {train_acc:.4f}, "
            f"Test Acc = {test_acc:.4f}, "
            f"LR = {current_lr:.6f}"
        )

        history.append({
            "epoch": epoch,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "learning_rate": current_lr
        })

        if (
            config["checkpoint"]["save_best"]
            and test_acc > best_test_acc
        ):
            best_test_acc = test_acc

            checkpoint_path = os.path.join(
                config["checkpoint"]["save_dir"],
                f"{experiment_name}_best.pt"
            )

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                test_acc=test_acc,
                checkpoint_path=checkpoint_path
            )

            print(
                f"Best checkpoint saved: {checkpoint_path} "
                f"with test_acc={test_acc:.4f}"
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

    print(f"Results saved to {result_path}")
    print(f"Best Test Acc: {best_test_acc:.4f}")


if __name__ == "__main__":
    main()
