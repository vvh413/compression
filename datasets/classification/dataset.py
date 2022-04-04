import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


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


if __name__ == "__main__":
    # from imagenet_mini import train_root, labels_dict
    from cifar10 import train_root

    dataset = ImageDataset(train_root)
    img, target = dataset[0]
    T.ToPILImage()(img).show()
    print(
        img, target, dataset.labels[target]
    )
