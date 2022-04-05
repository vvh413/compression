from datasets.classification import (CompressedImageDataset,
                                     cifar10)
import config
from models.compress import HyperpriorWrapper, bmshj2018_hyperprior
import pickle

compressor = HyperpriorWrapper(
    bmshj2018_hyperprior(config.COMPRESS_QUALITY, pretrained=True)
    .eval()
    .to(config.DEVICE),
    type="s",
)

# dataset_train = CompressedImageDataset(
#     root=cifar10.train_root,
#     compressor=compressor,
#     transform=config.transform_train,
# )

# with open("datasets/classification/cifar10_train.bin", "wb") as f:
#     pickle.dump(dataset_train, f)

dataset_val = CompressedImageDataset(
    root=cifar10.val_root,
    compressor=compressor,
    transform=config.transform_val,
)

with open("datasets/classification/cifar10_val.bin", "wb") as f:
    pickle.dump(dataset_val, f)
