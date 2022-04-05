from datasets.classification import (CompressedImageDataset,
                                     cifar10)
import config
from models.compress import HyperpriorWrapper, bmshj2018_hyperprior
import pickle
from torchvision import transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm

import numpy as np
import json


compressor = HyperpriorWrapper(
    bmshj2018_hyperprior(config.COMPRESS_QUALITY, pretrained=True)
    .eval()
    .to(config.DEVICE),
    type="s",
)

try:
    dataset_train = CompressedImageDataset(
        root=cifar10.train_root,
        transform=config.transform_train
    )
    dataset_val = CompressedImageDataset(
        root=cifar10.val_root,
        transform=config.transform_val
    )
    img, target = dataset_train[0]
    img = compressor.decompress(img.unsqueeze(dim=0)).squeeze()
    T.ToPILImage()(img).show()
    print(img, target, dataset_train.labels[target])

except AssertionError as e:
    print("no compressed images", e)
    print("compressing")
    dataset_train = CompressedImageDataset(
        root=cifar10.train_root,
        compressor=compressor,
        transform=config.transform_train,
        compressed=False
    )

    dataset_val = CompressedImageDataset(
        root=cifar10.val_root,
        compressor=compressor,
        transform=config.transform_val,
        compressed=False
    )
