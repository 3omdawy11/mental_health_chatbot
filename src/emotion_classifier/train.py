# src/emotion_classifier/train.py

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb
from config import (
    EPOCHS, LEARNING_RATE, MODEL_SAVE_PATH,
    WANDB_PROJECT, WANDB_RUN_NAME,
    EMBED_DIM, LSTM_UNITS, DROPOUT, BATCH_SIZE
)


# ── 1. One Training Epoch ─────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0

    for batch in loader:
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels         = batch['label'].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss   = criterion(logits, labels)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        preds          = torch.argmax(logits, dim=1)
        total_correct += (preds == labels).sum().item()
        total_loss    += loss.item() * len(labels)
        total_samples += len(labels)

    return total_loss / total_samples, total_correct / total_samples


# ── 2. Validation Epoch ───────────────────────────────────────────────────────

def evaluate_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['label'].to(device)

            logits = model(input_ids, attention_mask)
            loss   = criterion(logits, labels)

            preds          = torch.argmax(logits, dim=1)
            total_correct += (preds == labels).sum().item()
            total_loss    += loss.item() * len(labels)
            total_samples += len(labels)

    return total_loss / total_samples, total_correct / total_samples


# ── 3. Full Training Loop ─────────────────────────────────────────────────────

def train(model, train_loader, val_loader, device):

    # --- Init wandb run ---
    wandb.init(
        project = WANDB_PROJECT,
        name    = WANDB_RUN_NAME,
        config  = {
            "epochs"        : EPOCHS,
            "learning_rate" : LEARNING_RATE,
            "batch_size"    : BATCH_SIZE,
            "embed_dim"     : EMBED_DIM,
            "lstm_units"    : LSTM_UNITS,
            "dropout"       : DROPOUT,
            "optimizer"     : "Adam",
            "scheduler"     : "ReduceLROnPlateau",
        }
    )

    # Watch model — logs gradients and weights every epoch
    wandb.watch(model, log='all', log_freq=10)

    criterion  = nn.CrossEntropyLoss()
    optimizer  = Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler  = ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5, verbose=True)

    best_val_acc   = 0.0
    patience       = 3
    patience_count = 0

    print(f"Training on: {device}")
    print(f"{'Epoch':<8} {'Train Loss':<14} {'Train Acc':<14} {'Val Loss':<14} {'Val Acc':<14}")
    print("-" * 64)

    for epoch in range(1, EPOCHS + 1):

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss,   val_acc   = evaluate_one_epoch(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        # --- Log metrics to wandb ---
        wandb.log({
            "epoch"      : epoch,
            "train_loss" : train_loss,
            "train_acc"  : train_acc,
            "val_loss"   : val_loss,
            "val_acc"    : val_acc,
            "lr"         : optimizer.param_groups[0]['lr']  # track if LR changes
        })

        print(f"{epoch:<8} {train_loss:<14.4f} {train_acc:<14.4f} {val_loss:<14.4f} {val_acc:<14.4f}")

        # --- Save best model checkpoint to wandb ---
        if val_acc > best_val_acc:
            best_val_acc = val_acc

            # Save locally
            torch.save({
                'epoch'      : epoch,
                'model_state': model.state_dict(),
                'val_acc'    : best_val_acc,
                'optimizer'  : optimizer.state_dict(),
            }, MODEL_SAVE_PATH)

            # Upload checkpoint to wandb
            wandb.save(MODEL_SAVE_PATH)
            wandb.run.summary['best_val_acc'] = best_val_acc
            wandb.run.summary['best_epoch']   = epoch

            print(f"  ✅ Best model saved (val_acc: {best_val_acc:.4f})")
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"\n⚠️  Early stopping at epoch {epoch}")
                break

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")
    wandb.finish()