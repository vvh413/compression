import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, act=True, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs)
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if act else nn.Identity(),
        )

    def forward(self, x):
        return self.conv(x)


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, padding=1),
            ConvBlock(channels, channels, act=False, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, in_channels=3, n_features=[64, 128, 256], n_residuals=9):
        super().__init__()
        # kernel_size=7
        # padding=3
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels,
                n_features[0],
                kernel_size=7,
                stride=1,
                padding=3,
                padding_mode="reflect",
            ),
            nn.ReLU(inplace=True),
        )

        self.down = nn.ModuleList(
            [
                ConvBlock(
                    n_features[0], n_features[1], kernel_size=3, stride=1, padding=1
                ),
                ConvBlock(
                    n_features[1], n_features[2], kernel_size=3, stride=2, padding=1
                ),
            ]
        )

        self.res = nn.Sequential(*[ResBlock(n_features[2]) for _ in range(n_residuals)])

        self.up = nn.ModuleList(
            [
                ConvBlock(
                    n_features[2],
                    n_features[1],
                    down=False,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                ConvBlock(
                    n_features[1],
                    n_features[0],
                    down=False,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    output_padding=0,
                ),
            ]
        )
        # kernel_size=7
        # padding=3
        self.final = nn.Conv2d(
            n_features[0],
            in_channels,
            kernel_size=7,
            stride=1,
            padding=3,
            padding_mode="reflect",
        )

    def forward(self, x):
        x = self.initial(x)
        for layer in self.down:
            x = layer(x)
        x = self.res(x)
        for layer in self.up:
            x = layer(x)
        x = self.final(x)
        return torch.tanh(x)


def test():
    img_channels = 3
    img_size = 256
    n = 10
    x = torch.randn((n, img_channels, img_size, img_size))
    gen = Generator(img_channels)
    # print(gen)
    print(gen(x).shape)


if __name__ == "__main__":
    test()
