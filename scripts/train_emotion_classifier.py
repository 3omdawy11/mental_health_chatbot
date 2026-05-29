# scripts/train_emotion_classifier.py

import argparse
import torch
import wandb
import os
from config import IS_KAGGLE
from dotenv import load_dotenv
load_dotenv()

if IS_KAGGLE:
    from kaggle_secrets import UserSecretsClient


from src.emotion_classifier import (
    load_data,
    load_tokenizer,
    save_tokenizer,
    get_dataloaders,
    EmotionClassifier,
    train
)
from config import BASELINE_RUN_CONFIG, RUN2_CONFIG, RUN3_CONFIG, RUN4_CONFIG

# ── 0. Argument Parsing ───────────────────────────────────────────────────────

RUN_MAP = {
    "baseline" : BASELINE_RUN_CONFIG,
    "run2"     : RUN2_CONFIG,
    "run3"     : RUN3_CONFIG,
    "run4"     : RUN4_CONFIG,
}

parser = argparse.ArgumentParser(description="Train Emotion Classifier")
parser.add_argument(
    "--run",
    type    = str,
    choices = list(RUN_MAP.keys()),   # only accept valid run names
    default = "baseline",
    help    = "Which run config to use"
)
args = parser.parse_args()

run_config = RUN_MAP[args.run]
print(f"[CONFIG] Using run: '{args.run}'")
print(f"[CONFIG] Settings: {run_config}")


# ── 1. Device ─────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n[DEVICE] Using: {device}")


# ── 2. Wandb Login ────────────────────────────────────────────────────────────
print("\n[WANDB] Logging in...")
wandb.login(key=os.getenv("WANDB_API_KEY"))
print("[WANDB] Login successful")


# ── 3. Load Data ──────────────────────────────────────────────────────────────
print("\n[DATA] Loading dataset...")
train_df, val_df, test_df = load_data()
print(f"[DATA] Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
print(f"[DATA] Sample:\n{train_df.head(2)}")


# ── 4. Tokenizer ──────────────────────────────────────────────────────────────
print("\n[TOKENIZER] Loading pretrained BERT tokenizer...")
tokenizer = load_tokenizer()
print("[TOKENIZER] Loaded successfully")

print("[TOKENIZER] Saving tokenizer...")
save_tokenizer(tokenizer)
print("[TOKENIZER] Saved successfully")


# ── 5. DataLoaders ────────────────────────────────────────────────────────────
print("\n[DATALOADER] Creating dataloaders...")
train_loader, val_loader, test_loader = get_dataloaders(train_df, val_df, test_df, tokenizer, run_config)
print(f"[DATALOADER] Train batches: {len(train_loader)}")
print(f"[DATALOADER] Val batches:   {len(val_loader)}")
print(f"[DATALOADER] Test batches:  {len(test_loader)}")

print("\n[DATALOADER] Inspecting one batch...")
sample_batch = next(iter(train_loader))
print(f"[DATALOADER] input_ids shape:      {sample_batch['input_ids'].shape}")
print(f"[DATALOADER] attention_mask shape: {sample_batch['attention_mask'].shape}")
print(f"[DATALOADER] labels shape:         {sample_batch['label'].shape}")
print(f"[DATALOADER] Sample labels:        {sample_batch['label'][:8]}")


# ── 6. Model ──────────────────────────────────────────────────────────────────
print("\n[MODEL] Building EmotionClassifier...")
model = EmotionClassifier(run_config).to(device)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"[MODEL] Trainable parameters: {total_params:,}")


# ── 7. Sanity Check ───────────────────────────────────────────────────────────
print("\n[SANITY CHECK] Running one forward pass...")
model.eval()
with torch.no_grad():
    test_ids  = sample_batch['input_ids'][:2].to(device)
    test_mask = sample_batch['attention_mask'][:2].to(device)
    test_out  = model(test_ids, test_mask)
    print(f"[SANITY CHECK] Output shape: {test_out.shape}")  # should be (2, 6)
print("[SANITY CHECK] Forward pass OK ✅")


# ── 8. Train ──────────────────────────────────────────────────────────────────
print("\n[TRAIN] Starting training loop...")
train(model, train_loader, val_loader, device, run_config)
print("[TRAIN] Training complete ✅")


# ── 9. Done ───────────────────────────────────────────────────────────────────
print(f"\n[DONE] Best model saved to: {run_config['name']}")
print("[DONE] Check your wandb dashboard for training curves")