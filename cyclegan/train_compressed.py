import os
import torch

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

import config
from utils import load_checkpoint, save_checkpoint, norm, unnorm

from compress import HyperpriorWrapper
from model_compressed import CycleGAN

from dataset import CycleDataset
from dataset import horse2zebra as h2z


@torch.no_grad()
def compress(x, co):
    return co.encode(x)
    # compressed = co.compress(x)
    # return co.entropy_decode(compressed["strings"], compressed["shape"])


@torch.no_grad()
def decompress(y, co):
    return co.decode(y)
    # compressed = co.entropy_encode(y)
    # return co.decompress(compressed["strings"], compressed["shape"])


def train_cycle(
    epoch, model, dataloader, opt_dis, opt_gen, co=None
):

    progress = tqdm(
        dataloader, leave=True, desc=f"Epoch [{epoch+1}/{config.NUM_EPOCHS}]"
    )

    for i, (X, Y) in enumerate(progress):
        X = X.to(config.DEVICE)
        Y = Y.to(config.DEVICE)

        X = compress(X, co)
        Y = compress(Y, co)

        X_fake, Y_fake = model(X, Y)

        D_loss = model.loss_D(X, Y, X_fake.detach(), Y_fake.detach())

        opt_dis.zero_grad()
        D_loss.backward()
        opt_dis.step()

        G_loss = model.loss_G(X, Y, X_fake, Y_fake, co)

        opt_gen.zero_grad()
        G_loss.backward()
        opt_gen.step()

        progress.set_description(
            (
                f"Epoch [{epoch+1}/{config.NUM_EPOCHS}] | "
                f"D_loss = {D_loss:.3f} | "
                f"G_loss = {G_loss:.3f} "
                f"Y_fake = {Y_fake.mean().item():.3f} "
            )
        )

        assert str(G_loss.item()) != 'nan'

        if i % 200 == 0:
            X_fake = decompress(X_fake[:1], co).detach()
            Y_fake = decompress(Y_fake[:1], co).detach()
            X = decompress(X[:1], co).detach()
            Y = decompress(Y[:1], co).detach()
            save_image(X, os.path.join(config.RESULTS, f"{epoch}_{i//200}_X.png"))
            save_image(Y_fake, os.path.join(config.RESULTS, f"{epoch}_{i//200}_XY_f.png"))
            save_image(Y, os.path.join(config.RESULTS, f"{epoch}_{i//200}_Y.png"))
            save_image(X_fake, os.path.join(config.RESULTS, f"{epoch}_{i//200}_YX_f.png"))


def main():
    print(config.tag)
    print(config.DEVICE)

    if not os.path.exists(config.RESULTS):
        os.mkdir(config.RESULTS)

    compressor: HyperpriorWrapper = (
        HyperpriorWrapper(config.COMPRESS_QUALITY, pretrained=True)
        .eval()
        .to(config.DEVICE)
    )

    model = CycleGAN(
        in_channels=192,
        dis_features=[1024],
        gen_features=[1024, 1024, 1024],
        n_residuals=6
    ).to(config.DEVICE)

    opt_dis = optim.Adam(
        list(model.dis_X.parameters()) + list(model.dis_Y.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
        weight_decay=1e-4
    )
    opt_gen = optim.Adam(
        list(model.gen_X.parameters()) + list(model.gen_Y.parameters()),
        lr=config.LEARNING_RATE * 2,
        betas=(0.5, 0.999),
        weight_decay=1e-4
    )

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_CYCLE_GEN + ".4", model, opt_gen, config.LEARNING_RATE * 2
        )
        load_checkpoint(config.CHECKPOINT_CYCLE_DIS + ".4", model, opt_dis, config.LEARNING_RATE)
    else:
        config.START_EPOCH = 0

    dataset = CycleDataset(
        root_X=h2z.trainX,
        root_Y=h2z.trainY,
        transform=config.transform_train,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    for epoch in range(config.START_EPOCH, config.NUM_EPOCHS):
        train_cycle(
            epoch,
            model,
            dataloader,
            opt_dis,
            opt_gen,
            compressor,
        )

        if config.SAVE_MODEL:
            save_checkpoint(model, opt_gen, filename=config.CHECKPOINT_CYCLE_GEN + f".{epoch%5}")
            save_checkpoint(model, opt_dis, filename=config.CHECKPOINT_CYCLE_DIS + f".{epoch%5}")


if __name__ == "__main__":
    main()
