import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import utils
from torchvision.utils import save_image
from tqdm import tqdm

import config
from dataset import (CompressedImageDataset, ImageDataset,
                                     cifar10, imagenet_mini)
from model import Classifier
from compress import HyperpriorWrapper
from utils import load_checkpoint, norm, save_checkpoint, unnorm


def train(epoch, model, dataloader, opt, criterion, labels, co=None, **kwargs):
    progress = tqdm(
        dataloader, leave=True, desc=f"Epoch [{epoch+1}/{config.NUM_EPOCHS}]"
    )
    log_step = len(dataloader) // 10

    losses = []
    acc = []

    model.train()
    for i, (x, y) in enumerate(progress):
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)

        # if co:
        #     x = co.compress(x).detach()
        # else:
        #     x = norm(x)

        pred = model(x)

        loss = criterion(pred, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        losses.append(loss.item())
        acc.append((pred.argmax(dim=-1) == y).cpu().numpy().mean())

        progress.set_description(
            (
                f"train | epoch [{epoch+1}/{config.NUM_EPOCHS}] | "
                f"loss = {np.mean(losses):.3f} | "
                f"acc = {np.mean(acc):.3f} | "
            )
        )

        if i % log_step == 0:
            if co:
                x = co.decompress(x).detach()
            else:
                x = unnorm(x)
            x = x[0]
            y = pred.argmax(dim=-1)[0]
            label = labels[y]
            save_image(x, f"results/{epoch}_{i//log_step}_{label}.png")

    return losses


def val(epoch, model, dataloader, criterion, labels, co=None, **kwargs):
    progress = tqdm(
        dataloader, leave=True, desc=f"Epoch [{epoch+1}/{config.NUM_EPOCHS}]"
    )
    log_step = len(dataloader) // 10

    losses = []
    acc = []

    model.eval()
    for i, (x, y) in enumerate(progress):
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)

        # if co:
        #     x = co.compress(x).detach()
        # else:
        #     x = norm(x)

        pred = model(x)

        loss = criterion(pred, y)

        losses.append(loss.item())
        acc.append((pred.argmax(dim=-1) == y).cpu().numpy().mean())

        progress.set_description(
            (
                f"val | epoch [{epoch+1}/{config.NUM_EPOCHS}] | "
                f"loss = {np.mean(losses):.3f} | "
                f"acc = {np.mean(acc):.3f} | "
            )
        )

        if i == 0:
            if co:
                x = co.decompress(x).detach()
            else:
                x = unnorm(x)
            x = x[0]
            y = pred.argmax(dim=-1)[0]
            label = labels[y]
            save_image(x, f"results/{epoch}_val_{label}.png")
    return losses


def main():
    print(config.DEVICE)

    # compressor = None
    compressor = (
        HyperpriorWrapper(config.COMPRESS_QUALITY, pretrained=True)
        .eval()
        .to(config.DEVICE)
    )
    # CH = 3 if compressor is None else 192

    # model = Classifier(in_features=192*16*16, n_classes=10, hidden_layers=0, n_hidden=1024)
    model = Classifier(
        in_features=192 * 16 * 16, n_classes=10, hidden_layers=1, n_hidden=1024
    )
    opt = optim.Adam(
        list(model.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    criterion = nn.CrossEntropyLoss()

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_CLASS, model, opt, config.LEARNING_RATE)

    dataset_train = CompressedImageDataset(
        root=cifar10.train_root,
        compressor=compressor,
        transform=config.transform_train,
    )
    dataset_val = CompressedImageDataset(
        root=cifar10.val_root,
        compressor=compressor,
        transform=config.transform_val,
    )

    dataloader_train = DataLoader(
        dataset_train,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    for epoch in range(config.NUM_EPOCHS):
        train(
            epoch,
            model,
            dataloader_train,
            opt,
            criterion,
            dataset_train.labels,
            compressor,
        )
        val(epoch, model, dataloader_val, criterion, dataset_val.labels, compressor)

        if config.SAVE_MODEL:
            save_checkpoint(model, opt, filename=config.CHECKPOINT_CLASS)


if __name__ == "__main__":
    main()
