import torch
from compressai.zoo import bmshj2018_hyperprior


class HyperpriorWrapper:
    def __init__(self, model, type="a"):
        self.model = model
        if type == "a":
            self.compress = self.compress_a
            self.decompress = self.decompress_a
        elif type == "s":
            self.compress = self.compress_s
            self.decompress = self.decompress_s

    def compress_a(self, x):
        y = self.model.g_a(x)
        return y

    def compress_s(self, x):
        out = self.model.compress(x)
        strings, shape = out["strings"], out["shape"]

        z_hat = self.model.entropy_bottleneck.decompress(strings[1], shape)
        scales_hat = self.model.h_s(z_hat)
        indexes = self.model.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.model.gaussian_conditional.decompress(
            strings[0], indexes, z_hat.dtype
        )
        return y_hat

    def decompress_a(self, y):
        z = self.model.h_a(torch.abs(y))
        shape = z.size()[-2:]

        z_strings = self.model.entropy_bottleneck.compress(z)
        z_hat = self.model.entropy_bottleneck.decompress(z_strings, shape)

        scales_hat = self.model.h_s(z_hat)
        indexes = self.model.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.model.gaussian_conditional.compress(y, indexes)
        strings = [y_strings, z_strings]

        x_hat = self.model.decompress(strings, shape)["x_hat"]
        return x_hat

    def decompress_s(self, y_hat):
        x_hat = self.model.g_s(y_hat).clamp_(0, 1)
        return x_hat
