import pickle

import argparse

import torch

from torchvision import transforms as T
from tqdm import tqdm

import os
import config
from datasets.captioning import coco
from models.compress import HyperpriorWrapper

from PIL import Image


to_tensor = T.ToTensor()
to_image = T.ToPILImage()
resize = T.Resize((256, 256))


def compress1(input, output, compressor):
    image = resize(
        to_tensor(
            Image.open(input).convert("RGB")
        ).to(config.DEVICE)[None]
    )
    compressed = compressor.compress(image)
    with open(output, "wb") as out:
        pickle.dump(compressed, out)


def compress1_decoded(input, output, compressor):
    image = resize(
        to_tensor(
            Image.open(input).convert("RGB")
        ).to(config.DEVICE)[None]
    )
    compressed = compressor.compress(image)
    decoded = compressor.entropy_decode(compressed["strings"], compressed["shape"]).detach().cpu()
    with open(output, "wb") as out:
        pickle.dump(decoded, out)


def compress_many_decoded(inputs, output, compressor):
    images = []
    for input in inputs:
        images.append(resize(
            to_tensor(
                Image.open(input).convert("RGB")
            ).to(config.DEVICE)[None]
        ))
    images = torch.vstack(images)
    compressed = compressor.compress(images)
    decodeds = compressor.entropy_decode(compressed["strings"], compressed["shape"]).cpu()
    for i, decoded in enumerate(decodeds):
        with open(output[i], "wb") as out:
            pickle.dump(decoded, out)


def compress(quality, path, tag, batch_size):
    compressor = (
        HyperpriorWrapper(quality, pretrained=True)
        .eval()
        .to(config.DEVICE)
    )

    result = os.path.dirname(path) + "_" + tag
    if not os.path.exists(result):
        os.mkdir(result)
    images = os.listdir(path)

    # for i in tqdm(range(0, len(images), batch_size)):
    for image in tqdm(images):
        # in_, out_ = [], []
        # for image in images[i:i+batch_size]:
        #     in_.append(os.path.join(path, image))
        #     out_.append(os.path.join(result, image + ".bin"))
        in_ = os.path.join(path, image)
        out_ = os.path.join(result, image + ".bin")
        # compress1(in_, out_, compressor)
        compress1_decoded(in_, out_, compressor)
        # compress_many_decoded(in_, out_, compressor)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='compress image with hyperprior')
    parser.add_argument('-q', dest='quality', type=int, default=1,
                        help='compression quality')
    parser.add_argument('-p', dest='path', type=str, default=".",
                        help='path of image dir')
    parser.add_argument('-t', dest='tag', type=str, default="co1",
                        help='tag for result')

    args = parser.parse_args()
    compress(args.quality, args.path, args.tag, 32)
