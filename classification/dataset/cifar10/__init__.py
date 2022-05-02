import os

__DIR = os.path.dirname(os.path.abspath(__file__))
train_root = os.path.join(__DIR, "train")
val_root = os.path.join(__DIR, "test")
train_compressed_images = os.path.join(__DIR, "train_images.bin")
val_compressed_images = os.path.join(__DIR, "val_images.bin")
