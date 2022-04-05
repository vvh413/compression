import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from datasets.image_captioning import *
from models.captioning import CaptionNet

from utils import load_checkpoint, save_checkpoint

import matplotlib.pyplot as plt
from IPython.display import clear_output
from tqdm import tqdm


network = CaptionNet(n_tokens).to(config.DEVICE)

dummy_img_vec = torch.randn(len(captions[0]), 2048)
dummy_capt_ix = torch.tensor(as_matrix(captions[0]), dtype=torch.int64)
dummy_logits = network(dummy_img_vec.to(config.DEVICE), dummy_capt_ix.to(config.DEVICE))
print("shape:", dummy_logits.shape)
assert dummy_logits.shape == (dummy_capt_ix.shape[0], dummy_capt_ix.shape[1], n_tokens)


def compute_loss(network, image_vectors, captions_ix):
    """
    :param image_vectors: torch tensor containing inception vectors. shape: [batch, cnn_feature_size]
    :param captions_ix: torch tensor containing captions as matrix. shape: [batch, word_i].
        padded with pad_ix
    :returns: scalar crossentropy loss (neg llh) loss for next captions_ix given previous ones
    """

    # captions for input - all except last cuz we don't know next token for last one.
    captions_ix_inp = captions_ix[:, :-1].contiguous()
    captions_ix_next = captions_ix[:, 1:].contiguous()

    # apply the network, get predictions for captions_ix_next
    logits_for_next = network.forward(image_vectors, captions_ix_inp)

    # compute the loss function between logits_for_next and captions_ix_next
    # Use the mask, Luke: make sure that predicting next tokens after EOS do not contribute to loss
    # you can do that either by multiplying elementwise loss by (captions_ix_next != pad_ix)
    # or by using ignore_index in some losses.

    # <YOUR CODE>
    logits_for_next = torch.transpose(logits_for_next, 1, 2)
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_ix)
    loss = loss_fn(logits_for_next, captions_ix_next).unsqueeze(0)

    return loss


dummy_loss = compute_loss(network, dummy_img_vec.cuda(), dummy_capt_ix.cuda())

assert dummy_loss.shape == torch.Size([1]), "loss must be scalar"
assert (
    dummy_loss.data.detach().cpu().numpy() > 0
), "did you forget the 'negative' part of negative log-likelihood"

dummy_loss.backward()

assert all(
    param.grad is not None for param in network.parameters()
), "loss should depend differentiably on all neural network weights"

optimizer = torch.optim.Adam(network.parameters(), lr=1e-3, weight_decay=1e-4)

batch_size = 32  # adjust me (done)
n_epochs = 220  # adjust me (done)
n_batches_per_epoch = 40  # adjust me (done)
n_validation_batches = 5  # how many batches are used for validation after each epoch

train_losses = []
val_losses = []

for epoch in range(n_epochs):

    train_loss = 0
    network.train(True)
    progress = tqdm(
        range(n_batches_per_epoch), leave=True, desc=f"Epoch [{epoch+1}/{n_epochs}]"
    )
    for _ in progress:

        images_batch, captions_batch = generate_batch(train_img_codes, train_captions, batch_size)
        images_batch, captions_batch = images_batch.to(config.DEVICE), captions_batch.to(config.DEVICE)

        loss_t = compute_loss(
            network, images_batch, captions_batch
        )

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
        loss_t = compute_loss(
            network, *generate_batch(val_img_codes, val_captions, batch_size)
        )
        val_loss += loss_t.item()
    val_loss /= n_validation_batches
    val_losses.append(val_loss)
    print("val_loss =", val_loss)

    save_checkpoint(network, optimizer, filename="captions.pth.tar")

    clear_output()
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train loss", color="blue")
    plt.plot(val_losses, label="Val loss", color="orange")
    plt.axhline(y=3, color="gray", linestyle="--", label="Target loss")
    plt.legend()
    plt.ylabel("Loss")
    plt.show()

print("Finished!")
