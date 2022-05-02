from time import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython.display import clear_output
from tqdm import tqdm

import os
import json

import config
from datasets.captioning import coco_tag, annotations
# from datasets.captioning.dataset import CompressedImageDataset, make_collate_fn
from datasets.captioning.dataset_decoded import DecodedImageDataset, make_collate_fn
from models.captioning import CaptionNet, CaptionNetV3
from utils import load_checkpoint, save_checkpoint
from models.compress import HyperpriorWrapper

from torch.utils.data import DataLoader


train_size = int(118287 * 0.9)

dataset_train = DecodedImageDataset(
    coco_tag("co1d"),
    os.path.join(annotations, "captions_train2017_tok.json"),
    data_slice=slice(train_size)
)

dataset_val = DecodedImageDataset(
    coco_tag("co1d"),
    os.path.join(annotations, "captions_train2017_tok.json"),
    data_slice=slice(train_size, None),
)

dataloader_train = DataLoader(
    dataset_train,
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    num_workers=config.NUM_WORKERS,
    # pin_memory=True,
    collate_fn=make_collate_fn(dataset_train.pad_ix,
                               dataset_train.unk_ix, 
                               dataset_train.word_to_index)
)
dataloader_val = DataLoader(
    dataset_val,
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    num_workers=config.NUM_WORKERS,
    # pin_memory=True,
    collate_fn=make_collate_fn(dataset_val.pad_ix,
                               dataset_val.unk_ix,
                               dataset_val.word_to_index)
)


# network = CaptionNet(dataset_train.n_tokens, pad_ix=dataset_train.pad_ix, cnn_feature_size=512 * 49,
#                      cnn_in_channels=192, cnn_out_channels=512, pool=2).to(config.DEVICE)
tag = "capts_de_v3_co1d_e100_5e4_512emb"

# network_config = dict(
#     n_tokens=dataset_train.n_tokens, emb_size=256,
#     pad_ix=dataset_train.pad_ix, lstm_units=256,
#     feature_size=2048, cnn_in_channels=192
# )

network_config = dict(
    n_tokens=dataset_train.n_tokens, emb_size=512,
    pad_ix=dataset_train.pad_ix, lstm_units=512,
    feature_size=2048, cnn_in_channels=192
)

network = CaptionNetV3(**network_config).to(config.DEVICE)


def compute_loss(network, criterion, image_vectors, captions_ix):
    captions_ix_inp = captions_ix[:, :-1].contiguous()
    captions_ix_next = captions_ix[:, 1:].contiguous()

    logits_for_next = network(image_vectors, captions_ix_inp)

    logits_for_next = torch.transpose(logits_for_next, 1, 2)

    loss = criterion(logits_for_next, captions_ix_next).unsqueeze(0)

    return loss


optimizer = torch.optim.Adam(network.parameters(), lr=config.LEARNING_RATE) #, weight_decay=1e-4)

criterion = nn.CrossEntropyLoss(ignore_index=dataset_train.pad_ix)

# n_batches_per_epoch = 1000
# n_validation_batches = 100

train_losses = []
val_losses = []

# tag = "captions_compressed_full_ds_e100_m2_1e3"

if not os.path.exists("plots/" + tag):
    os.mkdir("plots/" + tag)
if not os.path.exists("checkpoints/" + tag):
    os.mkdir("checkpoints/" + tag)
    with open(f"checkpoints/{tag}/network_config.json", "w") as f:
        json.dump(network_config, f)
    config.LOAD_MODEL = False

epoch_start = 0
if config.LOAD_MODEL and len(os.listdir("checkpoints/" + tag)) > 0:
    checkpoint = sorted(os.listdir("checkpoints/" + tag))[-1]
    print("=> Found checkpoint:", checkpoint)
    epoch_start = int(checkpoint.split(".", 1)[0]) + 1
    load_checkpoint(f"checkpoints/{tag}/" + checkpoint,
                    network, optimizer,
                    config.LEARNING_RATE)

    losses_file = "plots/" + tag + "/" + checkpoint.split(".", 1)[0] + ".json"
    print("=> Loading losses from:", losses_file)
    with open(losses_file) as f:
        losses = json.load(f)
        train_losses = losses["train"]
        val_losses = losses["val"]

for epoch in range(epoch_start, config.NUM_EPOCHS):

    train_loss = 0
    network.train(True)
    progress = tqdm(dataloader_train, leave=True, desc=f"Epoch [{epoch+1}/{config.NUM_EPOCHS}]")
    for images_batch, captions_batch in progress:

        # images_batch = compressor.entropy_decode(*images_batch)
        images_batch, captions_batch = images_batch.to(
            config.DEVICE
        ), captions_batch.to(config.DEVICE)

        loss_t = compute_loss(network, criterion, images_batch, captions_batch)

        optimizer.zero_grad()
        loss_t.backward()
        optimizer.step()

        train_loss += loss_t.item()

        progress.set_description(
            (
                f"train | epoch [{epoch+1}/{config.NUM_EPOCHS}] | "
                f"loss = {loss_t.item():.3f} | "
            )
        )

    train_loss /= len(dataloader_train)
    train_losses.append(train_loss)

    val_loss = 0
    network.train(False)
    progress = tqdm(dataloader_val, leave=True, desc=f"Epoch [{epoch+1}/{config.NUM_EPOCHS}]")
    for images_batch, captions_batch in progress:

        # images_batch = compressor.entropy_decode(*images_batch)
        images_batch, captions_batch = images_batch.to(
            config.DEVICE
        ), captions_batch.to(config.DEVICE)
        loss_t = compute_loss(network, criterion, images_batch, captions_batch)
        val_loss += loss_t.item()

    val_loss /= len(dataloader_val)
    val_losses.append(val_loss)
    print("val_loss =", val_loss)

    epoch_tag = str(epoch).zfill(3)

    if config.SAVE_MODEL:
        save_checkpoint(network, optimizer,
                        filename=f"checkpoints/{tag}/{epoch_tag}.pth.tar")
    # clear_output()
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train loss", color="blue")
    plt.plot(val_losses, label="Val loss", color="orange")
    plt.axhline(y=3, color="gray", linestyle="--", label="Target loss")
    plt.legend()
    plt.ylabel("Loss")
    # plt.show()
    plt.savefig(f"plots/{tag}/{epoch_tag}.png")

    with open(f"plots/{tag}/{epoch_tag}.json", "w") as f:
        json.dump({"train": train_losses, "val": val_losses}, f)

print("Finished!")
