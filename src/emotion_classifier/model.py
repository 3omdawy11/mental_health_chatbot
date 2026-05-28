# src/emotion_classifier/model.py

import torch
import torch.nn as nn
from config import VOCAB_SIZE, EMBED_DIM, LSTM_UNITS, DROPOUT, NUM_CLASSES


class EmotionClassifier(nn.Module):

    def __init__(self):
        super(EmotionClassifier, self).__init__()

        # 1. Embedding — converts token IDs into dense vectors
        self.embedding = nn.Embedding(
            num_embeddings = VOCAB_SIZE,
            embedding_dim  = EMBED_DIM,
            padding_idx    = 0          # tells model to ignore padding tokens
        )

        # 2. Dropout on embedding output
        self.embed_dropout = nn.Dropout(0.2)

        # 3. Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size    = EMBED_DIM,
            hidden_size   = LSTM_UNITS,
            num_layers    = 2,           # stack two LSTM layers for more depth
            batch_first   = True,        # input shape: (batch, seq_len, features)
            bidirectional = True,
            dropout       = 0.3          # dropout between the two LSTM layers
        )

        # 4. Fully connected classification head
        # LSTM_UNITS * 2 because bidirectional concatenates forward + backward
        self.fc = nn.Sequential(
            nn.Linear(LSTM_UNITS * 2, 64),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, NUM_CLASSES)
        )

    def forward(self, input_ids, attention_mask):
        # input_ids:      (batch_size, max_len)
        # attention_mask: (batch_size, max_len) — 1 for real tokens, 0 for padding

        # Step 1 — embed
        x = self.embedding(input_ids)       # (batch, max_len, embed_dim)
        x = self.embed_dropout(x)

        # Step 2 — mask padding before passing to LSTM
        lengths = attention_mask.sum(dim=1).cpu()
        x = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )

        # Step 3 — BiLSTM
        packed_out, _ = self.lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        # x shape: (batch, max_len, LSTM_UNITS * 2)

        # Step 4 — GlobalMaxPool (take strongest signal across time steps)
        x = x.max(dim=1).values             # (batch, LSTM_UNITS * 2)

        # Step 5 — classify
        out = self.fc(x)                    # (batch, NUM_CLASSES)
        return out