import os

__DIR = os.path.dirname(os.path.abspath(__file__))
train_root = os.path.join(__DIR, "train")
val_root = os.path.join(__DIR, "test")

train_compressed = os.path.join(__DIR, "cifar10_train.bin")
val_compressed = os.path.join(__DIR, "cifar10_train.bin")
