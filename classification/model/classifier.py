import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            # nn.MaxPool2d(2, 2),
        )
        # nn.InstanceNorm2d(out_channels),
        # nn.LeakyReLU(0.2),

    def forward(self, x):
        return self.conv(x)


class CNN(nn.Module):
    def __init__(
        self, in_channels=3, feature_layers=2, feature_scale=64, in_size=(256, 256)
    ):
        super().__init__()

        features = [in_channels] + [
            min(2**i * feature_scale, 512) for i in range(feature_layers)
        ]
        self.conv = nn.Sequential(
            *[ConvBlock(features[i], features[i + 1]) for i in range(feature_layers)]
        )

        self.feature_size = int(
            in_size[0] * in_size[1] / (4**feature_layers) * features[-1]
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels=3, n_features=64):
        super().__init__()

        self.enc = nn.Sequential(
            nn.Conv2d(in_channels, n_features, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_features),
            nn.ReLU(),
            nn.Conv2d(n_features, n_features*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_features*2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x):
        x = self.enc(x)
        return x.squeeze()


class Classifier(nn.Module):
    def __init__(
        self,
        in_features=49152,
        n_classes=1000,
        dropout=0.3,
        # hidden_layers=1,
        n_hidden=4096,
    ):
        super().__init__()

        # hidden = []
        # for _ in range(hidden_layers):
        #     hidden.extend(
        #         [
        #             nn.Dropout(p=dropout),
        #             nn.Linear(n_hidden, n_hidden),
        #             nn.ReLU(),
        #         ]
        #     )

        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, n_hidden),
            nn.ReLU(),
            # *hidden,
            nn.Dropout(p=dropout),
            nn.Linear(n_hidden, n_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.mlp(x)
        return x


if __name__ == "__main__":
    x = torch.randn((8, 192, 16, 16))
    encoder = Encoder(in_channels=192, n_features=1024)
    print(encoder)
    print(encoder(x).shape)
    clf = Classifier(in_features=2048)  # in_features=encoder.feature_size)
    print(clf)
    y = clf(encoder(x))
    print(y.shape)
