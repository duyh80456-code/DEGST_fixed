"""CIFAR-10/100 and Tiny ImageNet loaders with an optional held-out validation split.

The experiment plan does no tuning; if you tune anything, select on the
validation split (``val_size > 0``), never on the test set.

Tiny ImageNet (200 classes, 64x64) has no public test labels, so its official
validation set (10k images) plays the role of the test set, as is usual.
"""
import os

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_and_extract_archive

_STATS = {
    'cifar10': ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    'cifar100': ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    'tinyimagenet': ((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)),
}
NUM_CLASSES = {'cifar10': 10, 'cifar100': 100, 'tinyimagenet': 200, 'fake': 10}
IMAGE_SIZE = {'cifar10': 32, 'cifar100': 32, 'tinyimagenet': 64, 'fake': 32}


class TinyImageNet(Dataset):
    """Tiny ImageNet-200 read from ``<root>/tiny-imagenet-200``; downloaded (~240 MB) if missing.

    ``train=False`` gives the official validation split, whose labels come from
    ``val/val_annotations.txt`` (the images are not in per-class folders).
    """
    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    base = 'tiny-imagenet-200'

    def __init__(self, root, train=True, download=False, transform=None):
        self.dir = os.path.join(root, self.base)
        self.transform = transform
        if download and not self._exists():
            download_and_extract_archive(self.url, root, filename=self.base + '.zip')
        if not self._exists():
            raise RuntimeError(f'Tiny ImageNet not found in {self.dir}; pass download=True')
        with open(os.path.join(self.dir, 'wnids.txt')) as f:
            self.classes = sorted(line.strip() for line in f if line.strip())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        if train:
            self.samples = []
            for c in self.classes:
                d = os.path.join(self.dir, 'train', c, 'images')
                self.samples += [(os.path.join(d, fn), self.class_to_idx[c]) for fn in sorted(os.listdir(d))]
        else:
            with open(os.path.join(self.dir, 'val', 'val_annotations.txt')) as f:
                rows = [line.split('\t') for line in f if line.strip()]
            self.samples = [(os.path.join(self.dir, 'val', 'images', r[0]), self.class_to_idx[r[1]])
                            for r in sorted(rows)]
        self.targets = [t for _, t in self.samples]
        # decoded images cached once as uint8 [N, 64, 64, 3] (identical pixels, no JPEG decoding per epoch)
        self.cache = os.path.join(self.dir, f'cache_{"train" if train else "val"}.npy')
        if not self._cache_ok():
            self._build_cache()
        self._arr = None            # opened lazily, so each data-loader worker maps the file itself

    def _exists(self):
        return os.path.isfile(os.path.join(self.dir, 'val', 'val_annotations.txt'))

    def _cache_ok(self):
        if not os.path.isfile(self.cache):
            return False
        return np.load(self.cache, mmap_mode='r').shape[0] == len(self.samples)

    def _build_cache(self):
        # some images are greyscale; default_loader converts to RGB
        arr = np.stack([np.asarray(default_loader(p), dtype=np.uint8) for p, _ in self.samples])
        tmp = self.cache + '.tmp.npy'
        np.save(tmp, arr)
        os.replace(tmp, self.cache)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        if self._arr is None:
            self._arr = np.load(self.cache, mmap_mode='r')
        img = Image.fromarray(np.array(self._arr[i]))
        return (self.transform(img) if self.transform else img), self.samples[i][1]


_DATASETS = {'cifar10': torchvision.datasets.CIFAR10, 'cifar100': torchvision.datasets.CIFAR100,
             'tinyimagenet': TinyImageNet}


def prepare(dataset, root='./data'):
    """Download ``dataset`` if missing (call once before starting parallel runs)."""
    if dataset in _DATASETS:
        for train in (True, False):
            _DATASETS[dataset](root, train=train, download=True)


def get_loaders(dataset='cifar10', root='./data', batch_size=128, num_workers=8,
                val_size=0, seed=0, fake_size=512):
    """Return (train_loader, eval_loader, eval_split_name).

    With ``val_size > 0`` the last ``val_size`` images of a seeded permutation of
    the training set are held out and returned as the evaluation loader.
    ``dataset='fake'`` gives random 32x32 tensors, used only for smoke tests.
    """
    if dataset == 'fake':
        tf = T.ToTensor()
        train = torchvision.datasets.FakeData(fake_size, (3, 32, 32), 10, transform=tf, random_offset=seed)
        test = torchvision.datasets.FakeData(fake_size // 4, (3, 32, 32), 10, transform=tf, random_offset=10_000)
        return (DataLoader(train, batch_size, shuffle=True, num_workers=0, drop_last=True),
                DataLoader(test, batch_size, shuffle=False, num_workers=0), 'test')

    if dataset not in _DATASETS:
        raise ValueError(f'unknown dataset {dataset}; choose from {sorted(_DATASETS) + ["fake"]}')
    mean, std = _STATS[dataset]
    hw = IMAGE_SIZE[dataset]
    tf_train = T.Compose([T.RandomCrop(hw, padding=hw // 8), T.RandomHorizontalFlip(), T.ToTensor(),
                          T.Normalize(mean, std)])
    tf_test = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    cls = _DATASETS[dataset]

    train = cls(root, train=True, download=True, transform=tf_train)
    if val_size > 0:
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(train), generator=g).tolist()
        train_idx, val_idx = perm[:-val_size], perm[-val_size:]
        val = Subset(cls(root, train=True, download=False, transform=tf_test), val_idx)
        train = Subset(train, train_idx)
        eval_set, split = val, 'val'
    else:
        eval_set, split = cls(root, train=False, download=True, transform=tf_test), 'test'

    kw = dict(num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)
    return (DataLoader(train, batch_size, shuffle=True, drop_last=True, **kw),
            DataLoader(eval_set, 512, shuffle=False, **kw), split)
