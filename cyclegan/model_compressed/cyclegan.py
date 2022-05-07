import torch
import torch.nn as nn
from .discriminator import Discriminator
from .generator import Generator
from torchvision.models import vgg19


class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = vgg19(pretrained=True).features[:35]
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.mse = nn.MSELoss()

    def forward(self, x, y):
        real_f = self.vgg(x)
        gen_f = self.vgg(y)
        return self.mse(real_f, gen_f)


def log(tag, x, y):
    print(tag, x.mean().item(), y.mean().item())


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

        for param in self.parameters():
            self._weights_init(param)

        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.l1 = nn.L1Loss()
        self.kl = nn.KLDivLoss()
        self.vgg = VGGLoss()

    def loss_D(self, X, Y, X_fake, Y_fake):
        D_X_loss = self.dis_X._loss(X, X_fake)
        D_Y_loss = self.dis_Y._loss(Y, Y_fake)
        return (D_X_loss + D_Y_loss) / 2

    def loss_G(self, X, Y, X_fake, Y_fake, co):
        D_X_fake = self.dis_X(X_fake)
        D_Y_fake = self.dis_Y(Y_fake)

        G_X_loss = self.bce(D_X_fake, torch.ones_like(D_X_fake))
        G_Y_loss = self.bce(D_Y_fake, torch.ones_like(D_Y_fake))

        G_loss = G_X_loss + G_Y_loss
        # log("g loss", G_X_loss, G_Y_loss)

        if self.lambda_cycle != 0:
            X_hat = self.gen_X(Y_fake)
            Y_hat = self.gen_Y(X_fake)
            # log("x y fake", Y_fake, X_fake)
            # log("hat", X_hat, Y_hat)
            X_cycle_kl = self.kl(X.detach(), X_hat.detach())
            Y_cycle_kl = self.kl(Y.detach(), Y_hat.detach())
            # print(X_cycle_kl.item(), Y_cycle_kl.item())
            X_cycle_loss = self.mse(X.detach(), X_hat.detach())
            Y_cycle_loss = self.mse(Y.detach(), Y_hat.detach())
            # log("loss", X_cycle_loss, Y_cycle_loss)
            X_de, Y_de = co.decode(X), co.decode(Y)
            X_hat_de, Y_hat_de = co.decode(X_hat), co.decode(Y_hat)
            # log("x y de", X_de, Y_de)
            X_cycle_img_loss = self.l1(X_de, X_hat_de)
            Y_cycle_img_loss = self.l1(Y_de, Y_hat_de)
            # log("hat de", X_hat_de, Y_hat_de)
            # log("img loss", X_cycle_img_loss, Y_cycle_img_loss)
            # X_content_loss = self.vgg(co.decode(X), co.decode(X_hat))
            # Y_content_loss = self.vgg(co.decode(Y), co.decode(Y_hat))
            G_loss += (
                + (X_cycle_loss + Y_cycle_loss)  # * self.lambda_cycle
                + (X_cycle_img_loss + Y_cycle_img_loss) * self.lambda_cycle
                + (X_cycle_kl + Y_cycle_kl)  # * self.lambda_cycle
                # + (X_content_loss + Y_content_loss) * self.lambda_cycle
            )

        if self.lambda_identity != 0:
            X_identity = self.gen_X(X)
            Y_identity = self.gen_Y(Y)
            # X_identity_loss = self.l1(co.decode(X), co.decode(X_identity))
            # Y_identity_loss = self.l1(co.decode(Y), co.decode(Y_identity))
            X_identity_loss = self.l1(X, X_identity)
            Y_identity_loss = self.l1(Y, Y_identity)
            G_loss += (
                (X_identity_loss + Y_identity_loss) * self.lambda_identity
            )

        return G_loss

    def _weights_init(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        if isinstance(m, nn.BatchNorm2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
            torch.nn.init.constant_(m.bias, 0)

    def forward(self, X, Y):
        X_fake = self.gen_X(Y)
        Y_fake = self.gen_Y(X)
        return X_fake, Y_fake
