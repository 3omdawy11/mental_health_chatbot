# src/tokenizer.py
import torch
import pickle
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from config import MAX_LEN, BATCH_SIZE, TOKENIZER_PATH


# ── 1. Load the pretrained tokenizer ─────────────────────────────────────────

def load_tokenizer():
    """
    Loads a pretrained BERT tokenizer — we use it purely for
    its vocabulary and encoding, not the BERT model itself.
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return tokenizer


def save_tokenizer(tokenizer, path=TOKENIZER_PATH):
    with open(path, 'wb') as f:
        pickle.dump(tokenizer, f)
    print(f'Tokenizer saved to {path}')


def load_saved_tokenizer(path=TOKENIZER_PATH):
    with open(path, 'rb') as f:
        tokenizer = pickle.load(f)
    print(f'Tokenizer loaded from {path}')
    return tokenizer


# ── 2. PyTorch Dataset class ──────────────────────────────────────────────────

class EmotionDataset(Dataset):
    """
    Wraps a DataFrame split into a PyTorch Dataset.
    Each item returns input_ids, attention_mask, and label as tensors.
    """
    def __init__(self, df, tokenizer, max_len=MAX_LEN):
        self.texts     = df['text'].tolist()
        self.labels    = df['label'].tolist()
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length      = self.max_len,
            padding         = 'max_length',
            truncation      = True,
            return_tensors  = 'pt'
        )
        return {
            'input_ids'      : encoding['input_ids'].squeeze(0),       # (max_len,)
            'attention_mask' : encoding['attention_mask'].squeeze(0),  # (max_len,)
            'label'          : torch.tensor(self.labels[idx], dtype=torch.long)
        }


# ── 3. DataLoaders ────────────────────────────────────────────────────────────

def get_dataloaders(train_df, val_df, test_df, tokenizer):
    """
    Main function — call this from main.py.
    Returns train, val, and test DataLoaders.
    """
    train_dataset = EmotionDataset(train_df, tokenizer)
    val_dataset   = EmotionDataset(val_df,   tokenizer)
    test_dataset  = EmotionDataset(test_df,  tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

    print(f'Train batches: {len(train_loader)}')
    print(f'Val batches:   {len(val_loader)}')
    print(f'Test batches:  {len(test_loader)}')

    return train_loader, val_loader, test_loader