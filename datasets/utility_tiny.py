"""Stratified 80/10/10 train/selection/audit split of Tiny ImageNet train."""
import hashlib
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import functional as TF
from datasets.tiny_imagenet import _ensure_tiny_imagenet


class UtilityDataset(Dataset):
    def __init__(self, folder, indices, split, seed):
        self.folder, self.indices, self.split, self.seed = folder, indices, split, seed
        self.epoch = 0

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        index = self.indices[position]
        path, target = self.folder.samples[index]
        x = TF.to_tensor(self.folder.loader(path))
        if self.split == 'train':
            # Augmentation determined by sample and epoch, not model RNG/worker order.
            g = torch.Generator().manual_seed(self.seed + self.epoch * 1000003 + index)
            x = TF.pad(x, [8, 8, 8, 8])
            top, left = torch.randint(0, 17, (2,), generator=g).tolist()
            x = x[:, top:top + 64, left:left + 64]
            if torch.rand((), generator=g).item() < 0.5:
                x = x.flip(-1)
        x = TF.normalize(x, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return x, target, Path(path).relative_to(self.folder.root).as_posix()


def make_datasets(config, data_seed):
    root = _ensure_tiny_imagenet(config['data_dir'], config.get('download', False))
    folder = ImageFolder(str(Path(root) / 'train'))
    targets = torch.tensor(folder.targets)
    if len(folder.classes) != 200 or len(folder) != 100000:
        raise ValueError('Expected complete Tiny ImageNet: 200 classes, 100000 training images')
    g = torch.Generator().manual_seed(config['split_seed'])
    splits = {k: [] for k in ('train', 'selection', 'audit')}
    for label in range(200):
        idx = (targets == label).nonzero().flatten()
        if len(idx) != 500:
            raise ValueError('Expected 500 images per class')
        idx = idx[torch.randperm(len(idx), generator=g)].tolist()
        for key, values in zip(splits, (idx[:400], idx[400:450], idx[450:])):
            splits[key].extend(values)
    manifest = {key: [Path(folder.samples[i][0]).relative_to(folder.root).as_posix()
                      for i in ids] for key, ids in splits.items()}
    fingerprint = hashlib.sha256(repr((folder.class_to_idx, manifest)).encode()).hexdigest()
    return ({k: UtilityDataset(folder, v, k, data_seed) for k, v in splits.items()},
            manifest, fingerprint)


def loader(dataset, config, seed, epoch=0):
    dataset.epoch = epoch
    return DataLoader(dataset, batch_size=config['batch_size'],
                      shuffle=dataset.split == 'train',
                      num_workers=config['num_workers'], pin_memory=True,
                      generator=torch.Generator().manual_seed(seed + epoch),
                      persistent_workers=False)
