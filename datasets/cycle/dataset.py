import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

import config


class CycleDataset(Dataset):
    def __init__(self, root_X, root_Y, transform=None):
        self.root_X = root_X
        self.root_Y = root_Y
        self.transform = transform

        self.X_images = os.listdir(root_X)
        self.Y_images = os.listdir(root_Y)
        self.X_len = len(self.X_images)
        self.Y_len = len(self.Y_images)
        self.dataset_len = max(self.X_len, self.Y_len)

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        X_img = self.X_images[index % self.X_len]
        Y_img = self.Y_images[index % self.Y_len]

        X_path = os.path.join(self.root_X, X_img)
        Y_path = os.path.join(self.root_Y, Y_img)

        X_img = Image.open(X_path).convert("RGB")
        Y_img = Image.open(Y_path).convert("RGB")

        if self.transform:
            X_img = self.transform(X_img)
            Y_img = self.transform(Y_img)
        else:
            X_img = T.ToTensor()(X_img)
            Y_img = T.ToTensor()(Y_img)

        return X_img, Y_img
