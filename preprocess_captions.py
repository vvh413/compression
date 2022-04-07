import json
import os

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
from datasets.image_captioning import coco_preprocessed, coco_raw
from models.compress import HyperpriorWrapper, bmshj2018_hyperprior

compressor = HyperpriorWrapper(
    bmshj2018_hyperprior(config.COMPRESS_QUALITY, pretrained=True)
    .eval()
    .to(config.DEVICE),
    type="s",
)

with open("datasets/image_captioning/annotations/captions_train2017.json") as f:
    j = json.load(f)
capts = j["annotations"]
new_capts = []
k = 0
i = -1
for capt in tqdm(sorted(capts, key=lambda x: x["image_id"])):
    if i != capt["image_id"]:
        i = capt["image_id"]
        k = 0
    k += 1
    if k <= 5:
        new_capts.append(capt)
j["annotations"] = new_capts
with open("datasets/image_captioning/annotations/captions_train2017.json", "w") as f:
    json.dump(j, f)

coco_train = coco.CocoCaptions(
    coco_raw,
    "datasets/image_captioning/annotations/captions_train2017.json",
    transform=config.transform_train,
)
dataloader = torch.utils.data.DataLoader(
    dataset=coco_train, batch_size=32, shuffle=False, num_workers=4
)

vectors, captions = [], []
for img_batch, capt_batch in tqdm(dataloader):
    capt_batch = list(zip(*capt_batch))
    img_batch_compressed = compressor.compress(img_batch)
    img_batch_compressed = F.max_pool2d(
        img_batch_compressed, kernel_size=2
    )  # .view(32, -1)
    img_batch_compressed = torch.flatten(img_batch_compressed, start_dim=1)
    vec_batch = img_batch_compressed.cpu().data.numpy()

    captions.extend(capt_batch)
    vectors.extend([vec for vec in vec_batch])

tokenizer = TweetTokenizer()
captions_tokenized = [
    [" ".join(filter(len, tokenizer.tokenize(cap.lower()))) for cap in img_captions]
    for img_captions in tqdm(captions)
]

np.save(os.path.join(coco_preprocessed, "co_image_codes_8m.npy"), np.asarray(vectors))

with open(os.path.join(coco_preprocessed, "co_captions_8m.json"), "w") as f_cap:
    json.dump(captions, f_cap)
with open(
    os.path.join(coco_preprocessed, "co_captions_tokenized_8m.json"), "w"
) as f_cap:
    json.dump(captions_tokenized, f_cap)
