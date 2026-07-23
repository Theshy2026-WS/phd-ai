from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_cifar100_loaders(batch_size, data_dir="./data"):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5071, 0.4867, 0.4408),
            std=(0.2675, 0.2565, 0.2761),
        ),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5071, 0.4867, 0.4408),
            std=(0.2675, 0.2565, 0.2761),
        ),
    ])

    train_data = datasets.CIFAR100(
        root=data_dir,
        train=True,
        download=False,
        transform=transform_train,
    )
    test_data = datasets.CIFAR100(
        root=data_dir,
        train=False,
        download=False,
        transform=transform_test,
    )

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )
    return train_loader, test_loader
