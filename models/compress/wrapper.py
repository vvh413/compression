import torch
from compressai.zoo import bmshj2018_hyperprior


class HyperpriorWrapper:
    def __init__(self, *args, **kwargs):
        self.model = bmshj2018_hyperprior(*args, **kwargs)

    def to(self, device):
        self.model.to(device)
        return self

    def eval(self):
        self.model.eval()
        return self

    def compress(self, x: torch.Tensor) -> dict:
        out = self.model.compress(x)
        return out

    def decompress(self, strings: list, shape: torch.Size) -> torch.Tensor:
        x_hat = self.model.decompress(strings, shape)["x_hat"]
        return x_hat

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        y = self.model.g_a(x)
        return y

    def entropy_encode(self, y: torch.Tensor) -> dict:
        z = self.model.h_a(torch.abs(y))
        shape = z.size()[-2:]

        z_strings = self.model.entropy_bottleneck.compress(z)
        z_hat = self.model.entropy_bottleneck.decompress(z_strings, shape)

        scales_hat = self.model.h_s(z_hat)
        indexes = self.model.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.model.gaussian_conditional.compress(y, indexes)
        return {"strings": [y_strings, z_strings], "shape": shape}

    def entropy_decode(self, strings: list, shape: torch.Size) -> torch.Tensor:
        z_hat = self.model.entropy_bottleneck.decompress(strings[1], shape)
        scales_hat = self.model.h_s(z_hat)
        indexes = self.model.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.model.gaussian_conditional.decompress(
            strings[0], indexes, z_hat.dtype
        )
        return y_hat

    def decode(self, y_hat: torch.Tensor) -> torch.Tensor:
        x_hat = self.model.g_s(y_hat).clamp_(0, 1)
        return x_hat
