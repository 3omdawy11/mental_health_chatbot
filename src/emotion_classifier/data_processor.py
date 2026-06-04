# src/tokenizer.py
import torch
import pickle
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from config import MAX_LEN, BATCH_SIZE, TOKENIZER_PATH, EMOTION_SPLIT_PATH
import pandas as pd

def load_data():
    """
    Loads the dataset from CSV files and returns train, val, test DataFrames.
    Expects files: data/train.csv, data/val.csv, data/test.csv
    Each file should have columns: 'text' and 'label'
    """
    train_df = pd.read_csv(EMOTION_SPLIT_PATH.format('train'))
    val_df   = pd.read_csv(EMOTION_SPLIT_PATH.format('val'))
    test_df  = pd.read_csv(EMOTION_SPLIT_PATH.format('test'))
    return train_df, val_df, test_df


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


def get_dataloaders(train_df, val_df, test_df, tokenizer, run_config):
    train_dataset = EmotionDataset(train_df, tokenizer, max_len=run_config["max_len"])
    val_dataset   = EmotionDataset(val_df,   tokenizer, max_len=run_config["max_len"])
    test_dataset  = EmotionDataset(test_df,  tokenizer, max_len=run_config["max_len"])

    train_loader = DataLoader(train_dataset, batch_size=run_config["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=run_config["batch_size"], shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=run_config["batch_size"], shuffle=False)

    return train_loader, val_loader, test_loader