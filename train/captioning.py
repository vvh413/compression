import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython.display import clear_output
from tqdm import tqdm

import config
from datasets.captioning.dataset import *
from models.captioning import CaptionNet
from utils import load_checkpoint, save_checkpoint
from models.compress import HyperpriorWrapper


compressor: HyperpriorWrapper = (
    HyperpriorWrapper(config.COMPRESS_QUALITY, pretrained=True)
    .eval()
    .to(config.DEVICE)
)

network = CaptionNet(n_tokens, pad_ix=pad_ix, cnn_feature_size=512 * 49,
                     cnn_in_channels=192, cnn_out_channels=512, pool=2).to(config.DEVICE)

# dummy_img_vec = torch.randn(len(captions[0]), feature_size)
# dummy_capt_ix = torch.tensor(as_matrix(captions[0]), dtype=torch.int64)
# dummy_logits = network(dummy_img_vec.to(config.DEVICE), dummy_capt_ix.to(config.DEVICE))
# print("shape:", dummy_logits.shape)
# assert dummy_logits.shape == (dummy_capt_ix.shape[0], dummy_capt_ix.shape[1], n_tokens)


def compute_loss(network, image_vectors, captions_ix):
    captions_ix_inp = captions_ix[:, :-1].contiguous()
    captions_ix_next = captions_ix[:, 1:].contiguous()

    logits_for_next = network.forward(image_vectors, captions_ix_inp)

    logits_for_next = torch.transpose(logits_for_next, 1, 2)
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_ix)
    loss = loss_fn(logits_for_next, captions_ix_next).unsqueeze(0)

    return loss


optimizer = torch.optim.Adam(network.parameters(), lr=1e-3, weight_decay=1e-4)

batch_size = 32
n_epochs = 250
n_batches_per_epoch = 1000
n_validation_batches = 100

train_losses = []
val_losses = []

for epoch in range(n_epochs):

    train_loss = 0
    network.train(True)
    progress = tqdm(
        range(n_batches_per_epoch), leave=True, desc=f"Epoch [{epoch+1}/{n_epochs}]"
    )
    for _ in progress:

        images_batch, captions_batch = generate_batch(
            train_img_codes, train_captions, batch_size
        )

        images_batch = compressor.entropy_decode(*images_batch)
        images_batch, captions_batch = images_batch.to(
            config.DEVICE
        ), captions_batch.to(config.DEVICE)

        loss_t = compute_loss(network, images_batch, captions_batch)

        optimizer.zero_grad()
        loss_t.backward()
        optimizer.step()

        train_loss += loss_t.item()

        progress.set_description(
            (
                f"train | epoch [{epoch+1}/{n_epochs}] | "
                f"loss = {loss_t.item():.3f} | "
            )
        )

    train_loss /= n_batches_per_epoch
    train_losses.append(train_loss)

    val_loss = 0
    network.train(False)
    for _ in range(n_validation_batches):

        images_batch, captions_batch = generate_batch(
            val_img_codes, val_captions, batch_size
        )
        images_batch = compressor.entropy_decode(*images_batch)
        images_batch, captions_batch = images_batch.to(
            config.DEVICE
        ), captions_batch.to(config.DEVICE)
        loss_t = compute_loss(network, images_batch, captions_batch)
        val_loss += loss_t.item()
    val_loss /= n_validation_batches
    val_losses.append(val_loss)
    print("val_loss =", val_loss)

    save_checkpoint(network, optimizer,
                    filename="captions_compressed_enc_512_2_1e-3_1000.pth.tar")
    # clear_output()
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train loss", color="blue")
    plt.plot(val_losses, label="Val loss", color="orange")
    plt.axhline(y=3, color="gray", linestyle="--", label="Target loss")
    plt.legend()
    plt.ylabel("Loss")
    # plt.show()
    plt.savefig("plots/capts_enc_512_2_1e-3.png")

print("Finished!")
