import json
import os
from collections import Counter
from random import choice

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms as T
from tqdm import tqdm

# class CaptionsDataset(Dataset):
__DIR = os.path.dirname(os.path.abspath(__file__))
coco_raw = os.path.join(__DIR, "coco2017/")
coco_preprocessed = os.path.join(__DIR, "coco2017_preprocessed/")

img_codes = np.load(os.path.join(__DIR, "coco2017_preprocessed/co_image_codes_8a.npy"))
captions = json.load(
    open(os.path.join(__DIR, "coco2017_preprocessed/co_captions_tokenized_8a.json"))
)

# split descriptions into tokens
for img_i in range(len(captions)):
    for caption_i in range(len(captions[img_i])):
        sentence = captions[img_i][caption_i]
        captions[img_i][caption_i] = ["#START#"] + sentence.split(" ") + ["#END#"]


word_counts = Counter()

for img_i in range(len(captions)):
    for caption_i in range(len(captions[img_i])):
        sentence = captions[img_i][caption_i][1:-1]
        word_num = Counter(sentence)
        for word in word_num:
            word_counts[word] += word_num[word]

vocab = ["#UNK#", "#START#", "#END#", "#PAD#"]
vocab += [k for k, v in word_counts.items() if v >= 5 if k not in vocab]
n_tokens = len(vocab)

assert 10000 <= n_tokens <= 10500

word_to_index = {w: i for i, w in enumerate(vocab)}

eos_ix = word_to_index["#END#"]
unk_ix = word_to_index["#UNK#"]
pad_ix = word_to_index["#PAD#"]


def as_matrix(sequences, max_len=None):
    max_len = max_len or max(map(len, sequences))

    matrix = np.zeros((len(sequences), max_len), dtype="int32") + pad_ix
    for i, seq in enumerate(sequences):
        row_ix = [word_to_index.get(word, unk_ix) for word in seq[:max_len]]
        matrix[i, : len(row_ix)] = row_ix

    return matrix


captions = np.array(captions)
l10 = len(captions) // 10
train_img_codes, val_img_codes = img_codes[:-l10], img_codes[-l10:]
train_captions, val_captions = captions[:-l10], captions[-l10:]

# train_test_split(
#     , test_size=0.1, random_state=42
# )


def generate_batch(img_codes, captions, batch_size, max_caption_len=None):
    random_image_ix = np.random.randint(0, len(img_codes), size=batch_size)
    batch_images = img_codes[random_image_ix]
    captions_for_batch_images = captions[random_image_ix]
    batch_captions = list(map(choice, captions_for_batch_images))
    batch_captions_ix = as_matrix(batch_captions, max_len=max_caption_len)
    return torch.tensor(batch_images, dtype=torch.float32), torch.tensor(
        batch_captions_ix, dtype=torch.int64
    )


if __name__ == "__main__":
    print("Each image code is a 2048-unit vector [ shape: %s ]" % str(img_codes.shape))
    print(img_codes[0, :10], end="\n\n")
    print("For each image there are 5 reference captions, e.g.:\n")
    print(captions[0])

    print(as_matrix(captions[1337]))
