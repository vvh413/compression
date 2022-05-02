import torch
from compressai.zoo import bmshj2018_hyperprior
from PIL import Image
import torchvision.transforms as T


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


if __name__ == "__main__":
    compressor: HyperpriorWrapper = (
        HyperpriorWrapper(1, pretrained=True)
        .eval()
        .to("cuda")
    )
    x = T.ToTensor()(Image.open("../dataset/horse2zebra/trainA/n02381460_1001.jpg")).to("cuda")
    print(x, x.shape)
    co = compressor.compress(x[None])
    y = compressor.entropy_decode(co["strings"], co["shape"])
    print(y.min(), y.max(), y.shape)
    y2 = compressor.encode(x[None])
    print(y2.min(), y2.max(), y2.shape)
    T.ToPILImage()(compressor.decode(y)[0].cpu()).show()
    T.ToPILImage()(compressor.decode(y2)[0].cpu()).show()
