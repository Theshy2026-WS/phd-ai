from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_mnist_loaders(batch_size):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.1307,),
            std=(0.3081,)
        )
    ])

    train_data = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )


    test_data = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )


    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True
    )


    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False
    )


    return train_loader, test_loader