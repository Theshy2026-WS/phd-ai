"""Same 80/10/10 split as the utility protocol, with isolated train augmentation."""
import numpy as np
import torch
from torchvision import transforms as T
from timm.data import Mixup
from datasets.utility_tiny import UtilityDataset, make_datasets


class AugmentedTrainingDataset(UtilityDataset):
    def __init__(self, base, augmentation):
        super().__init__(base.folder, base.indices, 'train', base.seed)
        self.transform = T.Compose([
            T.RandomResizedCrop(64, scale=tuple(augmentation['crop_scale']),
                                interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(),
            T.RandAugment(num_ops=augmentation['randaugment_ops'],
                          magnitude=augmentation['randaugment_magnitude'],
                          interpolation=T.InterpolationMode.BILINEAR, fill=[124,116,104]),
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
        ])

    def __getitem__(self, position):
        from pathlib import Path
        index = self.indices[position]
        path, label = self.folder.samples[index]
        # These torchvision PIL transforms use the CPU Torch RNG. Preserve its state.
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(self.seed + self.epoch*1000003 + index)
            x = self.transform(self.folder.loader(path))
        return x, label, Path(path).relative_to(self.folder.root).as_posix()


def prepare_datasets(config, seed):
    datasets, manifest, fingerprint = make_datasets(config['dataset'], seed)
    base = datasets['train']
    datasets['train'] = AugmentedTrainingDataset(base, config['augmentation'])
    # Small, fixed stratified clean training probe; never used to choose checkpoints.
    by_class = {}
    for i in base.indices:
        by_class.setdefault(base.folder.samples[i][1], []).append(i)
    g = torch.Generator().manual_seed(config['dataset']['split_seed'] + 90001)
    selected = []
    for label in sorted(by_class):
        candidates = by_class[label]
        order = torch.randperm(len(candidates), generator=g).tolist()
        selected.extend(candidates[j] for j in order[:config['training']['probe_per_class']])
    datasets['train_probe'] = UtilityDataset(base.folder, selected, 'train_probe', seed)
    return datasets, manifest, fingerprint


def mix_batch(x, y, training, seed, epoch, batch_index):
    """timm batch Mixup/CutMix with a model-independent, per-batch NumPy stream."""
    mixer = Mixup(mixup_alpha=training['mixup_alpha'], cutmix_alpha=training['cutmix_alpha'],
                  prob=training['mix_probability'], switch_prob=0.5, mode='batch',
                  label_smoothing=training['label_smoothing'], num_classes=200)
    # timm pairs samples with batch.flip(0) and samples mixing using NumPy only.
    previous = np.random.get_state()
    try:
        np.random.seed((seed + epoch*1000003 + batch_index*9176 + 70001) % (2**32))
        return mixer(x, y)
    finally:
        np.random.set_state(previous)
