import os
import pickle
import json
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from tqdm import tqdm
from collections import Counter

from torch.utils.data import DataLoader

from utils import pickle_load
from models.compress import HyperpriorWrapper
from torch.nn.utils.rnn import pad_sequence


def as_matrix(sequences, pad_ix, unk_ix, word_to_index, max_len=None):
    matrix = pad_sequence([
        torch.tensor([word_to_index.get(word, unk_ix)
                      for word in seq[:max_len]])
        for seq in sequences], batch_first=True, padding_value=pad_ix)
    return matrix


def make_collate_fn(pad_ix, unk_ix, word_to_index, max_caption_len=None):
    def collate_fn(batch):
        images = []
        captions = []
        for image, capts in batch:
            images.append(image)
            captions.append(random.choice(capts))
        images = torch.vstack(images)
        captions = as_matrix(captions, pad_ix, unk_ix, word_to_index, max_len=max_caption_len)
        return images, captions
    return collate_fn


class CompressedImageDataset(Dataset):
    def __init__(
        self,
        root: str,
        captions: str,
        compressor: HyperpriorWrapper,
        type: str = "coco",
        data_slice: slice = slice(None),
        transform=None,
        device="cpu",
    ):
        self.root = root
        self.transform = transform
        self.compressor = compressor
        self.device = device

        self.images = os.listdir(root)[data_slice]

        if type == "coco":
            self.__load_captions_coco(captions)
        self.__make_vocab()

        self.len = len(self.images)

    def __load_captions_coco(self, path):
        with open(path) as f:
            json_data = json.load(f)
        captions = {}
        self.word_counts = Counter()
        for capt in json_data["annotations"]:
            image_id = capt["image_id"]
            sentence = capt["tokenized"].split(" ")
            captions[image_id] = (
                captions.get(image_id, []) +
                [["#START#"] + sentence + ["#END#"]]
            )
            word_num = Counter(sentence)
            for word in word_num:
                self.word_counts[word] += word_num[word]
        for image in json_data["images"]:
            captions[image["file_name"]] = captions[image["id"]]
            del captions[image["id"]]
        self.captions = captions

    def __make_vocab(self):
        self.vocab = ["#UNK#", "#START#", "#END#", "#PAD#"]
        self.vocab += [
            k
            for k, v in self.word_counts.items()
            if v >= 5 if k not in self.vocab
        ]
        self.n_tokens = len(self.vocab)
        assert 10000 <= self.n_tokens <= 10500

        self.word_to_index = {w: i for i, w in enumerate(self.vocab)}

        self.eos_ix = self.word_to_index["#END#"]
        self.unk_ix = self.word_to_index["#UNK#"]
        self.pad_ix = self.word_to_index["#PAD#"]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        img = self.images[index]
        image_path = os.path.join(self.root, img)
        compressed = pickle_load(image_path)
        image = self.compressor.entropy_decode(compressed["strings"],
                                               compressed["shape"])
        captions = self.captions[img.rsplit(".", 1)[0]]
        return image, captions


if __name__ == "__main__":
    from datasets.captioning import coco_tag, annotations
    import time
    compressor = (
        HyperpriorWrapper(1, pretrained=True)
        .eval()
    )
    t1 = time.time()
    dataset = CompressedImageDataset(coco_tag("co1"),
                                     os.path.join(annotations, "captions_train2017_tok.json"),
                                     compressor)
    print(dataset.eos_ix)
    print(dataset.n_tokens)
    t2 = time.time()
    print("dataset loaded: ", t2 - t1)
    img, capts = dataset[0]
    print(img.shape, capts)
    t3 = time.time()
    print("get image: ", t3 - t2)

    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        collate_fn=make_collate_fn(dataset.pad_ix, dataset.unk_ix, dataset.word_to_index)
    )

    print(iter(dataloader).next())
    for images, captions in tqdm(dataloader):
        pass
