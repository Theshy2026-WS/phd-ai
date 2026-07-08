import torch
from tqdm import tqdm


def train(model, loader, optimizer, loss_fn, device, epoch):
    model.train()

    total_correct = 0
    total_seen = 0

    progress_bar = tqdm(
        loader,
        desc=f"Train Epoch {epoch}",
        unit="batch"
    )

    for x, y in progress_bar:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        output = model(x)
        loss = loss_fn(output, y)

        loss.backward()
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


def test(model, loader, device):
    model.eval()

    correct = 0
    total_seen = 0

    progress_bar = tqdm(
        loader,
        desc="Test",
        unit="batch"
    )

    with torch.no_grad():
        for x, y in progress_bar:
            x = x.to(device)
            y = y.to(device)

            output = model(x)
            pred = output.argmax(1)

            correct += (pred == y).sum().item()
            total_seen += x.size(0)

            accuracy = correct / total_seen

            progress_bar.set_postfix({
                "acc": f"{accuracy:.4f}",
                "images": f"{total_seen}/{len(loader.dataset)}"
            })

    return correct / len(loader.dataset)