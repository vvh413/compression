import os

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from tqdm import tqdm

import pickle


class ImageDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform

        self.labels = os.listdir(root)
        self.label2int = {label: i for i, label in enumerate(self.labels)}

        self.images = [
            (label, image)
            for label in self.labels
            for image in os.listdir(os.path.join(root, label))
        ]

        self.len = len(self.images)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        label, image = self.images[index]
        image_path = os.path.join(self.root, label, image)

        img = Image.open(image_path).convert("RGB")

        if self.transform:
            img = self.transform(img)
        else:
            img = T.ToTensor()(img)

        return img, self.label2int[label]


class CompressedImageDataset(Dataset):
    def __init__(self, root=None, file=None, compressor=None, compressed=True, transform=None, device="cpu"):
        self.root = root
        self.transform = transform
        self.compressor = compressor
        self.device = device

        if file:
            self.images = pickle.load(file)
            self.labels = [pair[0] for pair in self.images]
        else:
            assert root is not None
            self.labels = os.listdir(root)
            if compressed:
                self.images = [
                    self.__load_image(label, image)
                    for label in tqdm(self.labels)
                    for image in os.listdir(os.path.join(root, label))
                    if image.endswith(".bin")
                ]
                assert self.images != []
            else:
                print("=> Compressing images")
                self.images = [
                    self.__compress_image(label, image)
                    for label in tqdm(self.labels)
                    for image in tqdm(os.listdir(os.path.join(root, label)))
                    if not image.endswith(".bin")
                ]

        self.label2int = {label: i for i, label in enumerate(self.labels)}
        self.len = len(self.images)

    def __load_image(self, label, image):
        image_path = os.path.join(self.root, label, image)
        return label, torch.load(image_path)

    def __compress_image(self, label, image):
        assert self.compressor is not None
        image_path = os.path.join(self.root, label, image)

        img = Image.open(image_path).convert("RGB")

        if self.transform:
            img = self.transform(img)
        else:
            img = T.ToTensor()(img)
        img = img.to(self.device)
        img = self.compressor.compress(img.unsqueeze(dim=0)).squeeze()
        torch.save(img, image_path+".bin")
        return label, img

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        label, image = self.images[index]
        return image, self.label2int[label]


if __name__ == "__main__":
    # from imagenet_mini import train_root, labels_dict
    from cifar10 import train_root

    dataset = ImageDataset(train_root)
    img, target = dataset[0]
    T.ToPILImage()(img).show()
    print(img, target, dataset.labels[target])
