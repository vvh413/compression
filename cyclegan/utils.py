import copy
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms as T

import config

import pickle


def pickle_dump(obj, file):
    with open(file, "wb") as f:
        pickle.dump(obj, f)


def pickle_load(file):
    with open(file, "rb") as f:
        return pickle.load(f)


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old
    # checkpoint and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


# def seed_everything(seed=42):
#     os.environ["PYTHONHASHSEED"] = str(seed)
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False


def norm(x):
    return (x - 0.5) / 0.5


def unnorm(x):
    return x * 0.5 + 0.5


def imshow_plt(img):
    img = unnorm(img)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def imshow_pil(img):
    img = unnorm(img)
    pil_img = T.ToPILImage()(img)
    pil_img.show()


def plot(train_loss_history, val_loss_history):
    plt.figure(figsize=(17, 8))

    plt.subplot(1, 2, 1)
    plt.plot(train_loss_history)
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(val_loss_history)
    plt.grid()

    plt.show()
