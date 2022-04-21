import json
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.tokenize import TweetTokenizer
from torch.autograd import Variable
from torchvision.datasets import coco
from tqdm import tqdm

import config
from datasets.captioning import coco_preprocessed, coco_raw, annotations
from models.compress import HyperpriorWrapper, bmshj2018_hyperprior
from utils import pickle_dump

tag = ("_" + sys.argv[1]) if len(sys.argv) > 1 else ""
print("tag =", tag)

# with open("datasets/image_captioning/annotations/captions_train2017.json") as f:
#     j = json.load(f)
# capts = j["annotations"]
# new_capts = []
# k = 0
# i = -1
# for capt in tqdm(sorted(capts, key=lambda x: x["image_id"])):
#     if i != capt["image_id"]:
#         i = capt["image_id"]
#         k = 0
#     k += 1
#     if k <= 5:
#         new_capts.append(capt)
# j["annotations"] = new_capts
# with open("datasets/image_captioning/annotations/captions_train2017.json", "w") as f:
#     json.dump(j, f)

compressor = (
    HyperpriorWrapper(config.COMPRESS_QUALITY, pretrained=True)
    .eval()
    .to(config.DEVICE)
)

coco_train = coco.CocoCaptions(
    coco_raw,
    os.path.join(annotations, "captions_train2017.json"),
    transform=config.transform_train,
)

dataloader = torch.utils.data.DataLoader(
    dataset=coco_train, batch_size=16, shuffle=False, num_workers=4
)

images, captions = [], []
for img_batch, capt_batch in tqdm(dataloader):
    capt_batch = list(zip(*capt_batch))
    img_batch = img_batch.to(config.DEVICE)
    compressed = compressor.compress(img_batch)
    del img_batch

    strings, shape = compressed["strings"], compressed["shape"]
    del compressed
    captions.extend(capt_batch)
    images.extend([
        (strings[0][i], strings[1][i], shape)
        for i in range(len(strings[0]))
    ])
    del capt_batch, strings, shape


tokenizer = TweetTokenizer()
captions_tokenized = [
    [" ".join(filter(len, tokenizer.tokenize(cap.lower()))) for cap in img_captions]
    for img_captions in tqdm(captions)
]

del compressor, coco_train, dataloader, tokenizer

# np.save(os.path.join(coco_preprocessed, "co_image_codes_16.npy"), np.asarray(vectors))

pickle_dump(images, os.path.join(coco_preprocessed, f"image_codes{tag}.bin"))

with open(os.path.join(coco_preprocessed, f"captions{tag}.json"), "w") as f_cap:
    json.dump(captions, f_cap)
with open(os.path.join(coco_preprocessed, f"captions_tokenized{tag}.json"), "w") as f_cap:
    json.dump(captions_tokenized, f_cap)
