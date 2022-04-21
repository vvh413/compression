import pickle

import argparse

from torchvision import transforms as T
from tqdm import tqdm

import os
import config
from datasets.captioning import coco_raw
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


def compress(quality, path, tag):
    compressor = (
        HyperpriorWrapper(quality, pretrained=True)
        .eval()
        .to(config.DEVICE)
    )

    result = os.path.dirname(path) + "_" + tag
    if not os.path.exists(result):
        os.mkdir(result)
    images = os.listdir(path)

    for image in tqdm(images):
        in_ = os.path.join(path, image)
        out_ = os.path.join(result, image + ".bin")
        compress1(in_, out_, compressor)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='compress image with hyperprior')
    parser.add_argument('-q', dest='quality', type=int, default=1,
                        help='compression quality')
    parser.add_argument('-p', dest='path', type=str, default=".",
                        help='path of image dir')
    parser.add_argument('-t', dest='tag', type=str, default="co1",
                        help='tag for result')

    args = parser.parse_args()
    compress(args.quality, args.path, args.tag)
