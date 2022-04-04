import torch
import torch.nn as nn
from discriminator import Discriminator
from generator import Generator


class CycleGAN(nn.Module):
    def __init__(
        self,
        in_channels=3,
        dis_features=[64, 128, 256, 512],
        gen_features=[64, 128, 256],
        n_residuals=9,
        lambda_cycle=10,
        lambda_identity=0,
    ):
        super().__init__()
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity
        self.dis_X = Discriminator(in_channels=in_channels, features=dis_features)
        self.dis_Y = Discriminator(in_channels=in_channels, features=dis_features)
        self.gen_X = Generator(
            in_channels=in_channels, n_features=gen_features, n_residuals=n_residuals
        )
        self.gen_Y = Generator(
            in_channels=in_channels, n_features=gen_features, n_residuals=n_residuals
        )

    def loss_D(self, X, Y, X_fake, Y_fake, mse):
        D_X_loss = self.dis_X._loss(X, Y, X_fake, mse)
        D_Y_loss = self.dis_Y._loss(Y, X, Y_fake, mse)
        return (D_X_loss + D_Y_loss) / 2

    def loss_G(self, X, Y, X_fake, Y_fake, mse, l1):
        D_X_fake = self.dis_X(X_fake)
        D_Y_fake = self.dis_Y(Y_fake)
        G_X_loss = mse(D_X_fake, torch.ones_like(D_X_fake))
        G_Y_loss = mse(D_Y_fake, torch.ones_like(D_Y_fake))

        G_loss = G_X_loss + G_Y_loss

        if self.lambda_cycle != 0:
            X_hat = self.gen_X(Y_fake)
            Y_hat = self.gen_Y(X_fake)
            X_cycle_loss = l1(X, X_hat)
            Y_cycle_loss = l1(Y, Y_hat)
            G_loss += (
                X_cycle_loss * self.lambda_cycle + Y_cycle_loss * self.lambda_cycle
            )

        if self.lambda_identity != 0:
            X_identity = self.gen_X(X)
            Y_identity = self.gen_Y(Y)
            X_identity_loss = l1(X, X_identity)
            Y_identity_loss = l1(Y, Y_identity)
            G_loss += (
                X_identity_loss * self.lambda_identity
                + Y_identity_loss * self.lambda_identity
            )

        return G_loss

    def forward(self, X, Y):
        X_fake = self.gen_X(Y)
        Y_fake = self.gen_Y(X)
        return X_fake, Y_fake
