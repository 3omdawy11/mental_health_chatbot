# src/emotion_classifier/model.py

import torch
import torch.nn as nn


class EmotionClassifier(nn.Module):

    def __init__(self, config):
        super(EmotionClassifier, self).__init__()

        embed_dim      = config["embed_dim"]
        lstm_units     = config["lstm_units"]
        num_layers     = config["num_layers"]
        dropout        = config["dropout"]
        fc_hidden_dim  = config["fc_hidden_dim"]
        num_classes    = config["num_classes"]
        vocab_size     = config["vocab_size"]

        # 1. Embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embed_dropout = nn.Dropout(0.2)

        # 2. Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size    = embed_dim,
            hidden_size   = lstm_units,
            num_layers    = num_layers,
            batch_first   = True,
            bidirectional = True,
            dropout       = dropout if num_layers > 1 else 0.0
            # dropout between layers only makes sense if num_layers > 1
        )

        # 3. Fully connected head
        # lstm_units * 2 because bidirectional
        self.fc = nn.Sequential(
            nn.Linear(lstm_units * 2, fc_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_dim, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        # 1. Embed
        x = self.embedding(input_ids)       # (batch, max_len, embed_dim)
        x = self.embed_dropout(x)

        # 2. Pack → BiLSTM → Unpack
        lengths = attention_mask.sum(dim=1).cpu()
        x = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        x, _ = self.lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        # (batch, max_len, lstm_units * 2)

        # 3. GlobalMaxPool
        x = x.max(dim=1).values             # (batch, lstm_units * 2)

        # 4. Classify
        out = self.fc(x)                    # (batch, num_classes)
        return out