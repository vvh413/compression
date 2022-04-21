import torch
import torch.nn as nn
import torch.nn.functional as F


class CaptionNet(nn.Module):
    def __init__(
        self,
        n_tokens=10000,
        emb_size=128,
        pad_ix=0,
        lstm_units=256,
        cnn_feature_size=2048,
        cnn_in_channels=0,
        cnn_out_channels=0,
        pool=-1,
    ):
        super(self.__class__, self).__init__()

        self.initial = (
            nn.Sequential(
                nn.Conv2d(cnn_in_channels, cnn_out_channels, kernel_size=3),
                nn.BatchNorm2d(cnn_out_channels),
                nn.ReLU()
            )
            if cnn_in_channels > 0
            else nn.Identity()
        )
        self.pool = pool

        self.feature_to_h0 = nn.Linear(cnn_feature_size, lstm_units)
        self.feature_to_c0 = nn.Linear(cnn_feature_size, lstm_units)

        self.emb = nn.Embedding(n_tokens, emb_size, padding_idx=pad_ix)
        self.lstm = nn.LSTM(emb_size, lstm_units, batch_first=True)
        self.logits = nn.Linear(lstm_units, n_tokens)

    def forward(self, image_vectors, captions_ix):
        image_vectors = self.initial(image_vectors)
        if self.pool == 0:
            image_vectors = F.avg_pool2d(image_vectors, image_vectors.shape[-1])
        elif self.pool > 0:
            image_vectors = F.max_pool2d(image_vectors, self.pool)
        image_vectors = image_vectors.view(image_vectors.shape[0], -1)

        initial_cell = self.feature_to_h0(image_vectors)
        initial_hid = self.feature_to_c0(image_vectors)

        captions_emb = self.emb(captions_ix)
        state = (initial_cell[None], initial_hid[None])
        lstm_out, state = self.lstm(captions_emb, state)

        logits = self.logits(lstm_out)
        return logits
