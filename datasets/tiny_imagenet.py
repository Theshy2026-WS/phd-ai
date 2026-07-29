import os
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_and_extract_archive


TINY_IMAGENET_URL = "https://cs231n.stanford.edu/tiny-imagenet-200.zip"
TINY_IMAGENET_MD5 = "90528d7ca1a48142e341f4ef8d21d0de"


class TinyImageNetOfficialValidation(Dataset):
    """Read the official validation images using val_annotations.txt."""

    def __init__(self, dataset_root, class_to_index, transform=None):
        self.transform = transform

        annotation_path = os.path.join(
            dataset_root,
            "val",
            "val_annotations.txt",
        )
        image_dir = os.path.join(dataset_root, "val", "images")
        self.samples = []
        with open(annotation_path, "r", encoding="utf-8") as file:
            for line in file:
                filename, class_id, *_ = line.strip().split("\t")
                self.samples.append(
                    (
                        os.path.join(image_dir, filename),
                        class_to_index[class_id],
                    )
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, target = self.samples[index]
        image = default_loader(image_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def _stratified_split_indices(targets, validation_fraction, seed):
    indices_by_class = defaultdict(list)
    for index, target in enumerate(targets):
        indices_by_class[target].append(index)

    generator = torch.Generator().manual_seed(seed)
    train_indices = []
    validation_indices = []
    for class_indices in indices_by_class.values():
        permutation = torch.randperm(
            len(class_indices),
            generator=generator,
        ).tolist()
        validation_count = max(
            1,
            int(round(len(class_indices) * validation_fraction)),
        )
        shuffled = [class_indices[position] for position in permutation]
        validation_indices.extend(shuffled[:validation_count])
        train_indices.extend(shuffled[validation_count:])

    return train_indices, validation_indices


def _ensure_tiny_imagenet(data_dir, download):
    dataset_root = os.path.join(data_dir, "tiny-imagenet-200")
    if os.path.isdir(dataset_root):
        return dataset_root
    if not download:
        raise RuntimeError(
            "Tiny ImageNet was not found. Set dataset.download: true "
            "or place tiny-imagenet-200 under the configured data_dir."
        )

    download_and_extract_archive(
        url=TINY_IMAGENET_URL,
        download_root=data_dir,
        filename="tiny-imagenet-200.zip",
        md5=TINY_IMAGENET_MD5,
    )
    return dataset_root


def get_tiny_imagenet_loaders(
    batch_size,
    data_dir="./data",
    download=False,
    evaluation_split="development",
    validation_fraction=0.1,
    split_seed=42,
    num_workers=4,
):
    dataset_root = _ensure_tiny_imagenet(data_dir, download)

    train_transform = transforms.Compose([
        transforms.RandomCrop(64, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])
    evaluation_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])

    train_root = os.path.join(dataset_root, "train")
    augmented_train_data = datasets.ImageFolder(
        train_root,
        transform=train_transform,
    )

    if evaluation_split == "development":
        evaluation_train_data = datasets.ImageFolder(
            train_root,
            transform=evaluation_transform,
        )
        train_indices, validation_indices = _stratified_split_indices(
            augmented_train_data.targets,
            validation_fraction=validation_fraction,
            seed=split_seed,
        )
        train_data = Subset(augmented_train_data, train_indices)
        evaluation_data = Subset(
            evaluation_train_data,
            validation_indices,
        )
    elif evaluation_split == "official":
        train_data = augmented_train_data
        evaluation_data = TinyImageNetOfficialValidation(
            dataset_root,
            class_to_index=augmented_train_data.class_to_idx,
            transform=evaluation_transform,
        )
    else:
        raise ValueError(
            "evaluation_split must be either 'development' or 'official'"
        )

    persistent_workers = num_workers > 0
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
    )
    evaluation_loader = DataLoader(
        evaluation_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
    )
    return train_loader, evaluation_loader
