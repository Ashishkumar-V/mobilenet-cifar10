"""
dataset.py — CIFAR-10 data loading for MobileNet-v2 training.
"""

import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)


class Cutout:
    def __init__(self, n_holes=1, length=16):
        self.n_holes = n_holes
        self.length  = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        for _ in range(self.n_holes):
            y, x = np.random.randint(h), np.random.randint(w)
            y1, y2 = max(0, y - self.length // 2), min(h, y + self.length // 2)
            x1, x2 = max(0, x - self.length // 2), min(w, x + self.length // 2)
            mask[y1:y2, x1:x2] = 0.
        return img * torch.from_numpy(mask).expand_as(img)


class CIFAR10NoCheck(datasets.CIFAR10):
    """Skip MD5 integrity check — works with partial/symlinked dataset."""
    def _check_integrity(self):
        return True

    def _load_meta(self):
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                        'dog', 'frog', 'horse', 'ship', 'truck']
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}


def get_loaders(batch_size=128, num_workers=2, data_dir='./data'):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        Cutout(n_holes=1, length=16),
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    full_train = CIFAR10NoCheck(data_dir, train=True,  download=False, transform=train_transform)
    test_set   = CIFAR10NoCheck(data_dir, train=False, download=False, transform=eval_transform)

    train_set, val_set = torch.utils.data.random_split(
        full_train, [45000, 5000],
        generator=torch.Generator().manual_seed(42),
    )
    val_set.dataset.transform = eval_transform

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
