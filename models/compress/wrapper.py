import torch
from compressai.zoo import bmshj2018_hyperprior


class HyperpriorWrapper:
    def __init__(self, type="a", *args, **kwargs):
        self.model = bmshj2018_hyperprior(*args, **kwargs)

    def to(self, device):
        return self.model.to(device)

    def eval(self):
        return self.model.eval()

    def compress(self, x):
        out = self.model.compress(x)
        return out

    def decompress(self, strings, shape):
        x_hat = self.model.decompress(strings, shape)["x_hat"]
        return x_hat

    def encode(self, x):
        y = self.model.g_a(x)
        return y

    def entropy_encode(self, y):
        z = self.model.h_a(torch.abs(y))
        shape = z.size()[-2:]

        z_strings = self.model.entropy_bottleneck.compress(z)
        z_hat = self.model.entropy_bottleneck.decompress(z_strings, shape)

        scales_hat = self.model.h_s(z_hat)
        indexes = self.model.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.model.gaussian_conditional.compress(y, indexes)
        return {"strings": [y_strings, z_strings], "shape": shape}

    def entropy_decode(self, strings, shape):
        z_hat = self.model.entropy_bottleneck.decompress(strings[1], shape)
        scales_hat = self.model.h_s(z_hat)
        indexes = self.model.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.model.gaussian_conditional.decompress(
            strings[0], indexes, z_hat.dtype
        )
        return y_hat

    def decode(self, y_hat):
        x_hat = self.model.g_s(y_hat).clamp_(0, 1)
        return x_hat
