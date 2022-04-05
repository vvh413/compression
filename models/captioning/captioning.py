import torch
import torch.nn as nn


class CaptionNet(nn.Module):
    def __init__(
        self,
        n_tokens=10000,
        emb_size=128,
        lstm_units=256,
        cnn_feature_size=2048,
        pad_ix=0,
    ):
        """A recurrent 'head' network for image captioning. See scheme above."""
        super(self.__class__, self).__init__()

        self.feature_to_h0 = nn.Linear(cnn_feature_size, lstm_units)
        self.feature_to_c0 = nn.Linear(cnn_feature_size, lstm_units)

        self.emb = nn.Embedding(n_tokens, emb_size, padding_idx=pad_ix)
        self.lstm = nn.LSTM(emb_size, lstm_units, batch_first=True)
        self.logits = nn.Linear(lstm_units, n_tokens)

    def forward(self, image_vectors, captions_ix):
        """
        Apply the network in training mode.
        :param image_vectors: torch tensor containing inception vectors. shape: [batch, cnn_feature_size]
        :param captions_ix: torch tensor containing captions as matrix. shape: [batch, word_i].
            padded with pad_ix
        :returns: logits for next token at each tick, shape: [batch, word_i, n_tokens]
        """
        initial_cell = self.feature_to_h0(image_vectors)
        initial_hid = self.feature_to_c0(image_vectors)

        captions_emb = self.emb(captions_ix)
        # apply recurrent layer to captions_emb.
        # 1. initialize lstm state with initial_* from above
        # 2. feed it with captions. Mind the dimension order in docstring
        # 3. compute logits for next token probabilities
        # Note: if you used nn.LSTM, you can just give it (initial_cell[None], initial_hid[None]) as second arg

        # lstm_out should be lstm hidden state sequence of shape [batch, caption_length, lstm_units]
        state = (initial_cell[None], initial_hid[None])
        lstm_out, state = self.lstm(captions_emb, state)

        logits = self.logits(lstm_out)
        return logits
