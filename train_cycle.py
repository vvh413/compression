import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

import config
from utils import load_checkpoint, save_checkpoint


def train_cycle(
    epoch, dis_X, dis_Y, gen_X, gen_Y, dataloader, opt_dis, opt_gen, l1, mse, co=None
):

    progress = tqdm(
        dataloader, leave=True, desc=f"Epoch [{epoch+1}/{config.NUM_EPOCHS}]"
    )

    for i, (X, Y) in enumerate(progress):
        X = X.to(config.DEVICE)
        Y = Y.to(config.DEVICE)

        if co:
            X = co.compress(X).detach()
            Y = co.compress(Y).detach()
        else:
            X = config.norm(X)
            Y = config.norm(Y)

        X_fake = gen_X(Y)
        Y_fake = gen_Y(X)

        D_loss = loss_D(dis_X, dis_Y, X, Y, X_fake.detach(), Y_fake.detach(), mse)

        opt_dis.zero_grad()
        D_loss.backward()
        opt_dis.step()

        G_loss = loss_G(gen_X, gen_Y, dis_X, dis_Y, X, Y, X_fake, Y_fake, mse, l1)

        opt_gen.zero_grad()
        G_loss.backward()
        opt_gen.step()

        progress.set_description(
            (
                f"Epoch [{epoch+1}/{config.NUM_EPOCHS}] | "
                f"D_loss = {D_loss:.3f} | "
                f"G_loss = {G_loss:.3f} "
            )
        )

        if i % 200 == 0:
            if co:
                X_fake = co.decompress(X_fake).detach()
                Y_fake = co.decompress(Y_fake).detach()
                X = co.decompress(X).detach()
                Y = co.decompress(Y).detach()
            else:
                X_fake = config.unnorm(X_fake)
                Y_fake = config.unnorm(Y_fake)
            save_image(X, f"saved_images/h2z/{epoch}_{i//200}_X.png")
            save_image(X_fake, f"saved_images/h2z/{epoch}_{i//200}_X_f.png")
            save_image(Y, f"saved_images/h2z/{epoch}_{i//200}_Y.png")
            save_image(Y_fake, f"saved_images/h2z/{epoch}_{i//200}_Y_f.png")


def main():
    print(config.DEVICE)

    compressor = None
    compressor = Wrapper(
        bmshj2018_hyperprior(config.COMPRESS_QUALITY, pretrained=True).eval(), type="s"
    )
    CH = 3 if compressor is None else 192

    dis_X = Discriminator(in_channels=CH, features=[192]).to(config.DEVICE)
    dis_Y = Discriminator(in_channels=CH, features=[192]).to(config.DEVICE)
    gen_X = Generator(in_channels=CH, n_features=[192, 192, 384], n_residuals=9).to(
        config.DEVICE
    )
    gen_Y = Generator(in_channels=CH, n_features=[192, 192, 384], n_residuals=9).to(
        config.DEVICE
    )
    opt_dis = optim.Adam(
        list(dis_X.parameters()) + list(dis_Y.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )
    opt_gen = optim.Adam(
        list(gen_X.parameters()) + list(gen_Y.parameters()),
        lr=config.LEARNING_RATE * 2,
        betas=(0.5, 0.999),
    )

    l1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN_X, gen_X, opt_gen, config.LEARNING_RATE * 2
        )
        load_checkpoint(
            config.CHECKPOINT_GEN_Y, gen_Y, opt_gen, config.LEARNING_RATE * 2
        )
        load_checkpoint(config.CHECKPOINT_DIS_X, dis_X, opt_dis, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_DIS_Y, dis_Y, opt_dis, config.LEARNING_RATE)

    dataset = CustomDataset(
        root_X=config.TRAIN_DIR + "A",
        root_Y=config.TRAIN_DIR + "B",
        transform=config.transforms,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    for epoch in range(config.NUM_EPOCHS):
        train(
            epoch,
            dis_X,
            dis_Y,
            gen_X,
            gen_Y,
            dataloader,
            opt_dis,
            opt_gen,
            l1,
            mse,
            compressor,
        )

        if config.SAVE_MODEL:
            save_checkpoint(gen_X, opt_gen, filename=config.CHECKPOINT_GEN_X)
            save_checkpoint(gen_Y, opt_gen, filename=config.CHECKPOINT_GEN_Y)
            save_checkpoint(dis_X, opt_dis, filename=config.CHECKPOINT_DIS_X)
            save_checkpoint(dis_Y, opt_dis, filename=config.CHECKPOINT_DIS_Y)


if __name__ == "__main__":
    main()
