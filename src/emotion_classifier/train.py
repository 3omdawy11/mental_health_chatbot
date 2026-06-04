# src/emotion_classifier/train.py

import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import wandb
import numpy as np
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from config import BASELINE_RUN_CONFIG, WANDB_PROJECT, MODEL_SAVE_PATH


# ── 1. Helper: Compute Class Weights Once ─────────────────────────────────────

def calculate_loss_weights(train_loader, device):
    """
    Computes balanced class weights efficiently from the train loader 
    to handle severe class imbalances without looping endlessly.
    """
    print("⚖️  Calculating class balance vectors from dataset...")
    all_labels = []
    # Loop once to gather labels from the underlying dataset structural array
    for batch in train_loader:
        all_labels.extend(batch['label'].numpy())
        
    all_labels = np.array(all_labels)
    unique_classes = np.unique(all_labels)
    
    weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=all_labels
    )
    return torch.tensor(weights, dtype=torch.float).to(device)


# ── 2. One Training Epoch ─────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_samples = 0, 0
    all_preds, all_labels = [], []

    progress_bar = tqdm(loader, desc="   Training", leave=False)
    
    for batch in progress_bar:
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
        
        # Collect arrays for macro metric computations
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        total_loss   += loss.item() * len(labels)
        total_samples += len(labels)
        
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    epoch_loss = total_loss / total_samples
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    micro_f1 = f1_score(all_labels, all_preds, average='micro') # Equals Accuracy

    return epoch_loss, macro_f1, micro_f1


# ── 3. Validation Epoch ───────────────────────────────────────────────────────

def evaluate_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, total_samples = 0, 0
    all_preds, all_labels = [], []

    progress_bar = tqdm(loader, desc="   Validating", leave=False)
    
    with torch.no_grad():
        for batch in progress_bar:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['label'].to(device)

            logits = model(input_ids, attention_mask)
            loss   = criterion(logits, labels)

            preds          = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            total_loss   += loss.item() * len(labels)
            total_samples += len(labels)

    epoch_loss = total_loss / total_samples
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    micro_f1 = f1_score(all_labels, all_preds, average='micro') # Equals Accuracy

    return epoch_loss, macro_f1, micro_f1


# ── 4. Full Training Loop ─────────────────────────────────────────────────────

def train(model, train_loader, val_loader, device, run_config=BASELINE_RUN_CONFIG):

    # --- Init wandb run ---
    wandb.init(
        project = WANDB_PROJECT,
        name    = run_config["name"],
        config  = {
            "epochs"        : run_config["epochs"],
            "learning_rate" : run_config["learning_rate"],
            "batch_size"    : run_config["batch_size"],
            "embed_dim"     : run_config["embed_dim"],
            "lstm_units"    : run_config["lstm_units"],
            "dropout"       : run_config["dropout"],
            "optimizer"     : "AdamW",
            "scheduler"     : "ReduceLROnPlateau",
        }
    )

    wandb.watch(model, log='all', log_freq=10)

    # ── INJECT CLASS WEIGHTS INTO LOSS CRITERION ─────────────────────────────
    class_weights = calculate_loss_weights(train_loader, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    # ─────────────────────────────────────────────────────────────────────────
    
    optimizer = AdamW(model.parameters(), lr=run_config["learning_rate"], weight_decay=0.0)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

    best_val_f1    = 0.0  # Monitoring Validation Macro F1 instead of Accuracy
    patience       = 6    # Slightly increased to give overfitting presets room to breathe
    patience_count = 0

    print(f"\nTraining on: {device}")
    print(f"{'Epoch':<6} {'Train Loss':<11} {'Train Macro':<12} {'Train Micro':<12} {'Val Loss':<10} {'Val Macro':<11} {'Val Micro':<11}")
    print("-" * 80)

    for epoch in range(1, run_config["epochs"] + 1):

        train_loss, train_macro, train_micro = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss,   val_macro,   val_micro   = evaluate_one_epoch(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        # --- Log metrics to wandb ---
        wandb.log({
            "epoch":       epoch,
            "train_loss":  train_loss,
            "train_macro": train_macro,
            "train_micro": train_micro,
            "val_loss":    val_loss,
            "val_macro":   val_macro,
            "val_micro":   val_micro,
            "lr":          optimizer.param_groups[0]['lr']
        })

        print(f"{epoch:<6} {train_loss:<11.4f} {train_macro:<12.4f} {train_micro:<12.4f} {val_loss:<10.4f} {val_macro:<11.4f} {val_micro:<11.4f}")

        # --- Save best model checkpoint based on Macro F1 ---
        if val_macro > best_val_f1:
            best_val_f1 = val_macro

            os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

            torch.save({
                'epoch':       epoch,
                'model_state': model.state_dict(),
                'val_macro_f1': best_val_f1,
                'optimizer':   optimizer.state_dict(),
            }, MODEL_SAVE_PATH)

            wandb.save(MODEL_SAVE_PATH)
            wandb.run.summary['best_val_macro_f1'] = best_val_f1
            wandb.run.summary['best_epoch']        = epoch

            print(f"   ✅ Best model saved (val_macro_f1: {best_val_f1:.4f})")
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"\n⚠️  Early stopping triggered at epoch {epoch}")
                break

    print(f"\nTraining complete. Best Validation Macro F1: {best_val_f1:.4f}")
    wandb.finish()