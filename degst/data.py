"""CIFAR-10/100 loaders with an optional held-out validation split.

The experiment plan does no tuning; if you tune anything, select on the
validation split (``val_size > 0``), never on the test set.
"""
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

_STATS = {
    'cifar10': ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    'cifar100': ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
}
NUM_CLASSES = {'cifar10': 10, 'cifar100': 100, 'fake': 10}


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

    mean, std = _STATS[dataset]
    tf_train = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(mean, std)])
    tf_test = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    cls = torchvision.datasets.CIFAR10 if dataset == 'cifar10' else torchvision.datasets.CIFAR100

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
