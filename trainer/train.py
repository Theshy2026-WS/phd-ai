import torch
from tqdm import tqdm


def train(
    model,
    loader,
    optimizer,
    loss_fn,
    device,
    epoch,
    scaler=None,
    amp_enabled=False,
    amp_dtype=torch.float16,
    gradient_clip_norm=None,
):
    model.train()

    total_correct = 0
    total_seen = 0

    progress_bar = tqdm(
        loader,
        desc=f"Train Epoch {epoch}",
        unit="batch"
    )

    for x, y in progress_bar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=str(device).split(":")[0],
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            output = model(x)
            loss = loss_fn(output, y)

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            if gradient_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    gradient_clip_norm,
                )
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    gradient_clip_norm,
                )
            optimizer.step()

        pred = output.argmax(1)

        total_correct += (pred == y).sum().item()
        total_seen += x.size(0)

        accuracy = total_correct / total_seen

        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{accuracy:.4f}",
            "images": f"{total_seen}/{len(loader.dataset)}"
        })

    return total_correct / len(loader.dataset)


def test(
    model,
    loader,
    device,
    amp_enabled=False,
    amp_dtype=torch.float16,
):
    model.eval()

    correct = 0
    top5_correct = 0
    total_seen = 0

    progress_bar = tqdm(
        loader,
        desc="Test",
        unit="batch"
    )

    with torch.no_grad():
        for x, y in progress_bar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.autocast(
                device_type=str(device).split(":")[0],
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                output = model(x)
            pred = output.argmax(1)
            top5_pred = output.topk(
                min(5, output.size(1)),
                dim=1,
            ).indices

            correct += (pred == y).sum().item()
            top5_correct += (
                top5_pred == y.unsqueeze(1)
            ).any(dim=1).sum().item()
            total_seen += x.size(0)

            accuracy = correct / total_seen
            top5_accuracy = top5_correct / total_seen

            progress_bar.set_postfix({
                "acc": f"{accuracy:.4f}",
                "top5": f"{top5_accuracy:.4f}",
                "images": f"{total_seen}/{len(loader.dataset)}"
            })

    return (
        correct / len(loader.dataset),
        top5_correct / len(loader.dataset),
    )
