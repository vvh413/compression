import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=stride,
                padding=1,
                bias=True,
                padding_mode="reflect",
            ),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


def log(x):
    shape = x.shape
    print(shape, shape[1] * shape[2] * shape[3])


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels,
                features[0],
                kernel_size=2,
                stride=2,
                padding=0,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(
                Block(in_channels, feature, stride=2 if feature != features[-1] else 1)
            )
            in_channels = feature
        layers.append(
            nn.Conv2d(
                in_channels,
                1,
                kernel_size=2,
                stride=1,
                padding=0,
                padding_mode="reflect",
            )
        )
        self.model = nn.Sequential(*layers)

        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

    def _loss(self, X, X_fake):
        D_X_real = self(X)
        D_X_fake = self(X_fake)
        D_X_real_loss = self.bce(D_X_real, torch.ones_like(D_X_real))
        D_X_fake_loss = self.bce(D_X_fake, torch.zeros_like(D_X_fake))
        return D_X_real_loss + D_X_fake_loss

    def forward_log(self, x):
        log(x)
        x = self.initial(x)
        log(x)
        for layer in self.model:
            x = layer(x)
            log(x)
        return x

    def forward(self, x):
        x = self.initial(x)
        x = self.model(x)
        return x


def test():
    x = torch.randn((10, 192, 16, 16))
    model = Discriminator(in_channels=192, features=[1024])
    preds = model.forward_log(x)
    # print(preds.shape)


if __name__ == "__main__":
    test()
