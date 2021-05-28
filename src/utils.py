from collections import namedtuple
import random
import warnings

import numpy as np
import tensorflow as tf
import torch
import torchvision


ATTACKS = (
    'fgsm',
    'bim',
    'cw',
)

PROJECTIONS = (
    'parametric_umap_oos',
    'parametric_umap',
    'pca_oos',
    'pca',
    'umap_oos',
    'umap',
    'tsne',
)


def get_devices():
    devices = ['cpu']
    # As of PyTorch 1.7.0, calling torch.cuda.is_available shows a warning ("...Found no NVIDIA
    # driver on your system..."). A related issue is reported in PyTorch Issue #47038.
    # Warnings are suppressed below to prevent a warning from showing when no GPU is available.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        cuda_available = torch.cuda.is_available()
    if cuda_available and torch.cuda.device_count() > 0:
        devices.append('cuda')
        for idx in range(torch.cuda.device_count()):
            devices.append('cuda:{}'.format(idx))
    return tuple(devices)


def cifar10_loader(batch_size=128, train=True, num_workers=0, root='data', shuffle=True):
    # Make sure test data is not shuffled, so that the order is consistent.
    assert train or not shuffle
    transforms_list = []
    if train:
        transforms_list.append(torchvision.transforms.RandomCrop(32, padding=4))
        transforms_list.append(torchvision.transforms.RandomHorizontalFlip())
    transforms_list.append(torchvision.transforms.ToTensor())
    transforms = torchvision.transforms.Compose(transforms_list)
    dataset = torchvision.datasets.CIFAR10(
        root=root, train=train, transform=transforms, download=True
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    return loader


def set_seed(seed, extra=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    tf.random.set_seed(seed)
    if extra:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
