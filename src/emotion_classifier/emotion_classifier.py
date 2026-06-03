# """
# src/modules/emotion_classifier.py
# ===================================
# Two responsibilities:

# 1.  Training helpers  (EmotionDataset, Trainer)
#     – used by scripts/02_train_emotion_classifier.py
#     – and by the Kaggle notebook

# 2.  Inference wrapper  (EmotionClassifier)
#     – used at runtime by the chatbot pipeline

# Usage – inference
# -----------------
#     from src.modules.emotion_classifier import EmotionClassifier
#     clf = EmotionClassifier()                     # lazy-loads on first call
#     result = clf.classify("I feel absolutely hopeless today")
#     # {"emotion": "sadness", "confidence": 0.91, "all_scores": {...}}

# Usage – training
# ----------------
#     from src.modules.emotion_classifier import EmotionDataset, Trainer
#     # (see scripts/02_train_emotion_classifier.py for full example)
# """

# from __future__ import annotations

# import json
# import logging
# import time
# from pathlib import Path
# from typing import Optional

# import numpy as np
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from transformers import (
#     DistilBertTokenizerFast,
#     DistilBertForSequenceClassification,
#     get_linear_schedule_with_warmup,
# )
# from sklearn.utils.class_weight import compute_class_weight

# logger = logging.getLogger(__name__)

# # ── Constants ─────────────────────────────────────────────────────────────────

# EMOTION_LABELS = ["anger", "fear", "joy", "love", "sadness", "surprise"]
# LABEL2ID = {e: i for i, e in enumerate(EMOTION_LABELS)}
# ID2LABEL = {i: e for i, e in enumerate(EMOTION_LABELS)}

# _ROOT = Path(__file__).resolve().parent.parent.parent
# _DEFAULT_MODEL_DIR = _ROOT / "models" / "emotion_classifier"


# # ─────────────────────────────────────────────────────────────────────────────
# # 1.  Dataset
# # ─────────────────────────────────────────────────────────────────────────────

# class EmotionDataset(Dataset):
#     """
#     PyTorch Dataset for emotion classification.

#     Parameters
#     ----------
#     texts  : list of raw text strings
#     labels : list of emotion label strings  (or None for inference)
#     tokenizer : HuggingFace tokenizer
#     max_length : truncation length (default 128 — tweets are short)
#     """

#     def __init__(
#         self,
#         texts: list[str],
#         labels: list[str] | None,
#         tokenizer: DistilBertTokenizerFast,
#         max_length: int = 128,
#     ) -> None:
#         self.encodings = tokenizer(
#             texts,
#             truncation=True,
#             padding="max_length",
#             max_length=max_length,
#             return_tensors="pt",
#         )
#         if labels is not None:
#             self.labels = torch.tensor(
#                 [LABEL2ID[lbl] for lbl in labels], dtype=torch.long
#             )
#         else:
#             self.labels = None

#     def __len__(self) -> int:
#         return self.encodings["input_ids"].shape[0]

#     def __getitem__(self, idx: int) -> dict:
#         item = {
#             "input_ids":      self.encodings["input_ids"][idx],
#             "attention_mask": self.encodings["attention_mask"][idx],
#         }
#         if self.labels is not None:
#             item["labels"] = self.labels[idx]
#         return item


# # ─────────────────────────────────────────────────────────────────────────────
# # 2.  Class-weight helper
# # ─────────────────────────────────────────────────────────────────────────────

# def compute_weights(label_list: list[str], device: torch.device) -> torch.Tensor:
#     """
#     Compute inverse-frequency class weights to address the 10:1 imbalance
#     (sadness=4637 vs surprise=458 in this dataset).
#     Passed directly to nn.CrossEntropyLoss(weight=...).
#     """
#     classes = np.array(sorted(set(LABEL2ID.values())))
#     label_ids = np.array([LABEL2ID[l] for l in label_list])
#     weights = compute_class_weight("balanced", classes=classes, y=label_ids)
#     return torch.tensor(weights, dtype=torch.float).to(device)


# # ─────────────────────────────────────────────────────────────────────────────
# # 3.  Trainer
# # ─────────────────────────────────────────────────────────────────────────────

# class Trainer:
#     """
#     Lightweight training loop for DistilBERT fine-tuning.

#     Design choices
#     ---------------
#     * Weighted cross-entropy loss  →  handles 10:1 class imbalance
#       (preferred over oversampling: simpler, no data leakage risk)
#     * Linear LR schedule with warmup (10 % of steps)
#     * Early stopping with configurable patience
#     * Mixed-precision (fp16) when CUDA is available
#     * Logs every `log_every` batches; saves best checkpoint by val F1

#     Parameters
#     ----------
#     model_name   : HuggingFace model id (default "distilbert-base-uncased")
#     model_dir    : where to save checkpoints + final weights
#     lr           : learning rate  (2e-5 works well for DistilBERT fine-tuning)
#     batch_size   : 32 on GPU, auto-reduced to 16 on CPU
#     epochs       : max epochs (early stopping may terminate earlier)
#     max_length   : tokeniser truncation length
#     patience     : early stopping patience (epochs without val improvement)
#     """

#     def __init__(
#         self,
#         model_name: str = "distilbert-base-uncased",
#         model_dir: str | Path | None = None,
#         lr: float = 2e-5,
#         batch_size: int = 32,
#         epochs: int = 5,
#         max_length: int = 128,
#         patience: int = 2,
#     ) -> None:
#         self.model_name = model_name
#         self.model_dir  = Path(model_dir) if model_dir else _DEFAULT_MODEL_DIR
#         self.model_dir.mkdir(parents=True, exist_ok=True)

#         self.lr         = lr
#         self.epochs     = epochs
#         self.max_length = max_length
#         self.patience   = patience

#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         # Auto-reduce batch on CPU to avoid OOM
#         self.batch_size = batch_size if self.device.type == "cuda" else min(batch_size, 16)
#         print(f"  Device : {self.device}  |  batch_size={self.batch_size}")

#         self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
#         self.model: Optional[DistilBertForSequenceClassification] = None
#         self.history: list[dict] = []

#     # ── Data helpers ──────────────────────────────────────────────────────────

#     def _make_loader(
#         self,
#         texts: list[str],
#         labels: list[str] | None,
#         shuffle: bool = False,
#     ) -> DataLoader:
#         ds = EmotionDataset(texts, labels, self.tokenizer, self.max_length)
#         return DataLoader(
#             ds,
#             batch_size=self.batch_size,
#             shuffle=shuffle,
#             num_workers=0,          # 0 = safe on all platforms / Kaggle
#             pin_memory=(self.device.type == "cuda"),
#         )

#     # ── Core train / eval steps ───────────────────────────────────────────────

#     def _train_epoch(
#         self,
#         loader: DataLoader,
#         optimizer: torch.optim.Optimizer,
#         scheduler,
#         loss_fn: nn.CrossEntropyLoss,
#         scaler,
#     ) -> float:
#         self.model.train()
#         total_loss = 0.0
#         for batch in loader:
#             optimizer.zero_grad()
#             input_ids      = batch["input_ids"].to(self.device)
#             attention_mask = batch["attention_mask"].to(self.device)
#             labels         = batch["labels"].to(self.device)

#             if scaler is not None:
#                 with torch.amp.autocast("cuda"):
#                     logits = self.model(input_ids, attention_mask=attention_mask).logits
#                     loss   = loss_fn(logits, labels)
#                 scaler.scale(loss).backward()
#                 scaler.unscale_(optimizer)
#                 nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
#                 scaler.step(optimizer)
#                 scaler.update()
#             else:
#                 logits = self.model(input_ids, attention_mask=attention_mask).logits
#                 loss   = loss_fn(logits, labels)
#                 loss.backward()
#                 nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
#                 optimizer.step()

#             scheduler.step()
#             total_loss += loss.item()

#         return total_loss / len(loader)

#     @torch.no_grad()
#     def _eval_epoch(
#         self,
#         loader: DataLoader,
#         loss_fn: nn.CrossEntropyLoss,
#     ) -> tuple[float, float, np.ndarray, np.ndarray]:
#         """Returns (loss, accuracy, all_preds, all_labels)."""
#         self.model.eval()
#         total_loss, correct, total = 0.0, 0, 0
#         all_preds, all_labels = [], []

#         for batch in loader:
#             input_ids      = batch["input_ids"].to(self.device)
#             attention_mask = batch["attention_mask"].to(self.device)
#             labels         = batch["labels"].to(self.device)

#             logits = self.model(input_ids, attention_mask=attention_mask).logits
#             loss   = loss_fn(logits, labels)

#             preds   = logits.argmax(dim=-1)
#             correct += (preds == labels).sum().item()
#             total   += labels.size(0)
#             total_loss += loss.item()
#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())

#         acc = correct / total
#         return total_loss / len(loader), acc, np.array(all_preds), np.array(all_labels)

#     # ── Public: fit ───────────────────────────────────────────────────────────

#     def fit(
#         self,
#         train_texts: list[str],
#         train_labels: list[str],
#         val_texts: list[str],
#         val_labels: list[str],
#     ) -> list[dict]:
#         """
#         Fine-tune DistilBERT.
#         Returns history: list of per-epoch dicts with train/val loss + accuracy.
#         """
#         from sklearn.metrics import f1_score

#         print(f"\n  Loading {self.model_name} …")
#         self.model = DistilBertForSequenceClassification.from_pretrained(
#             self.model_name,
#             num_labels=len(EMOTION_LABELS),
#             id2label=ID2LABEL,
#             label2id=LABEL2ID,
#         ).to(self.device)

#         train_loader = self._make_loader(train_texts, train_labels, shuffle=True)
#         val_loader   = self._make_loader(val_texts,   val_labels,   shuffle=False)

#         # Weighted loss — key for 10:1 imbalance
#         weights  = compute_weights(train_labels, self.device)
#         loss_fn  = nn.CrossEntropyLoss(weight=weights)

#         optimizer = torch.optim.AdamW(
#             self.model.parameters(), lr=self.lr, weight_decay=0.01
#         )
#         total_steps  = len(train_loader) * self.epochs
#         warmup_steps = int(0.1 * total_steps)
#         scheduler = get_linear_schedule_with_warmup(
#             optimizer, num_warmup_steps=warmup_steps,
#             num_training_steps=total_steps,
#         )
#         scaler = torch.amp.GradScaler() if self.device.type == "cuda" else None

#         best_val_f1   = -1.0
#         patience_left = self.patience
#         self.history  = []

#         print(f"  Training {self.epochs} epochs  |  "
#               f"steps/epoch={len(train_loader)}  |  warmup={warmup_steps}")
#         print(f"  Class weights: { {e: round(float(w),3) for e,w in zip(EMOTION_LABELS, weights.cpu())} }")
#         print()

#         for epoch in range(1, self.epochs + 1):
#             t0 = time.time()

#             train_loss = self._train_epoch(
#                 train_loader, optimizer, scheduler, loss_fn, scaler
#             )
#             val_loss, val_acc, val_preds, val_true = self._eval_epoch(
#                 val_loader, loss_fn
#             )
#             val_f1 = f1_score(val_true, val_preds, average="macro")
#             elapsed = time.time() - t0

#             row = dict(
#                 epoch=epoch,
#                 train_loss=round(train_loss, 4),
#                 val_loss=round(val_loss, 4),
#                 val_acc=round(val_acc, 4),
#                 val_f1=round(val_f1, 4),
#                 elapsed=round(elapsed, 1),
#             )
#             self.history.append(row)

#             marker = ""
#             if val_f1 > best_val_f1:
#                 best_val_f1   = val_f1
#                 patience_left = self.patience
#                 self._save_checkpoint("best")
#                 marker = "  ← best ✓"
#             else:
#                 patience_left -= 1

#             print(
#                 f"  Epoch {epoch}/{self.epochs}  "
#                 f"train_loss={train_loss:.4f}  "
#                 f"val_loss={val_loss:.4f}  "
#                 f"val_acc={val_acc:.4f}  "
#                 f"val_f1={val_f1:.4f}  "
#                 f"({elapsed:.0f}s){marker}"
#             )

#             if patience_left <= 0:
#                 print(f"\n  Early stopping after epoch {epoch} (patience={self.patience})")
#                 break

#         print(f"\n  Best val macro-F1 : {best_val_f1:.4f}")
#         return self.history

#     # ── Save / load checkpoints ───────────────────────────────────────────────

#     def _save_checkpoint(self, tag: str) -> None:
#         path = self.model_dir / tag
#         self.model.save_pretrained(path)
#         self.tokenizer.save_pretrained(path)

#     def save_final(self) -> None:
#         """
#         Copy best checkpoint to model_dir root and save training artefacts.
#         """
#         import shutil
#         best_path  = self.model_dir / "best"
#         final_path = self.model_dir

#         if best_path.exists():
#             for f in best_path.iterdir():
#                 shutil.copy2(f, final_path / f.name)

#         # Training log
#         log_path = self.model_dir / "training_log.txt"
#         with open(log_path, "w") as f:
#             f.write("epoch,train_loss,val_loss,val_acc,val_f1,elapsed_s\n")
#             for row in self.history:
#                 f.write(
#                     f"{row['epoch']},{row['train_loss']},{row['val_loss']},"
#                     f"{row['val_acc']},{row['val_f1']},{row['elapsed']}\n"
#                 )

#         # Hyperparameter config
#         cfg = dict(
#             model_name=self.model_name,
#             num_labels=len(EMOTION_LABELS),
#             label2id=LABEL2ID,
#             id2label=ID2LABEL,
#             lr=self.lr,
#             batch_size=self.batch_size,
#             max_length=self.max_length,
#             patience=self.patience,
#             imbalance_strategy="weighted_cross_entropy",
#             best_val_f1=max(r["val_f1"] for r in self.history) if self.history else None,
#         )
#         with open(self.model_dir / "train_config.json", "w") as f:
#             json.dump(cfg, f, indent=2)

#         print(f"\n  Saved final model → {final_path}")
#         print(f"  Training log      → {log_path}")


# # ─────────────────────────────────────────────────────────────────────────────
# # 4.  Inference wrapper
# # ─────────────────────────────────────────────────────────────────────────────

# class EmotionClassifier:
#     """
#     Runtime inference wrapper.  Lazy-loads the fine-tuned DistilBERT model.

#     Parameters
#     ----------
#     model_dir : directory containing config.json + pytorch weights + tokenizer.
#                 Defaults to models/emotion_classifier/.
#     device    : 'cuda', 'cpu', or None (auto-detect).
#     """

#     def __init__(
#         self,
#         model_dir: str | Path | None = None,
#         device: str | None = None,
#     ) -> None:
#         self._dir    = Path(model_dir) if model_dir else _DEFAULT_MODEL_DIR
#         self._device = torch.device(
#             device if device else ("cuda" if torch.cuda.is_available() else "cpu")
#         )
#         self._model     = None
#         self._tokenizer = None
#         self._loaded    = False

#     # ── Lazy load ─────────────────────────────────────────────────────────────

#     def _load(self) -> None:
#         if self._loaded:
#             return
#         if not (self._dir / "config.json").exists():
#             raise FileNotFoundError(
#                 f"No model found at {self._dir}.\n"
#                 "Run  python scripts/02_train_emotion_classifier.py  "
#                 "(or download from Kaggle) first."
#             )
#         self._tokenizer = DistilBertTokenizerFast.from_pretrained(str(self._dir))
#         self._model     = DistilBertForSequenceClassification.from_pretrained(
#             str(self._dir)
#         ).to(self._device)
#         self._model.eval()
#         self._loaded = True

#     # ── Public API ────────────────────────────────────────────────────────────

#     def classify(self, text: str) -> dict:
#         """
#         Classify a single text.

#         Returns
#         -------
#         {
#             "emotion"     : "sadness",
#             "confidence"  : 0.87,
#             "all_scores"  : {"anger": 0.02, "fear": 0.03, ...}
#         }
#         """
#         self._load()
#         if not isinstance(text, str) or not text.strip():
#             return {"emotion": "unknown", "confidence": 0.0, "all_scores": {}}

#         enc = self._tokenizer(
#             text,
#             return_tensors="pt",
#             truncation=True,
#             padding=True,
#             max_length=128,
#         ).to(self._device)

#         with torch.no_grad():
#             logits = self._model(**enc).logits
#         probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

#         all_scores = {
#             ID2LABEL[i]: round(float(p), 4) for i, p in enumerate(probs)
#         }
#         top_emotion = max(all_scores, key=all_scores.__getitem__)

#         return {
#             "emotion":    top_emotion,
#             "confidence": all_scores[top_emotion],
#             "all_scores": dict(
#                 sorted(all_scores.items(), key=lambda x: -x[1])
#             ),
#         }

#     def classify_batch(self, texts: list[str], batch_size: int = 32) -> list[dict]:
#         """Batch inference. Empty/invalid strings return emotion='unknown'."""
#         self._load()
#         _empty = {"emotion": "unknown", "confidence": 0.0, "all_scores": {}}
#         valid_indices = [i for i, t in enumerate(texts) if isinstance(t, str) and t.strip()]
#         valid_texts   = [texts[i] for i in valid_indices]
#         valid_results: list[dict] = []
#         for start in range(0, len(valid_texts), batch_size):
#             chunk = valid_texts[start : start + batch_size]
#             enc = self._tokenizer(
#                 chunk, return_tensors="pt", truncation=True,
#                 padding=True, max_length=128,
#             ).to(self._device)
#             with torch.no_grad():
#                 logits = self._model(**enc).logits
#             probs = torch.softmax(logits, dim=-1).cpu().numpy()
#             for row in probs:
#                 all_scores  = {ID2LABEL[j]: round(float(p), 4) for j, p in enumerate(row)}
#                 top_emotion = max(all_scores, key=all_scores.__getitem__)
#                 valid_results.append({
#                     "emotion": top_emotion,
#                     "confidence": all_scores[top_emotion],
#                     "all_scores": dict(sorted(all_scores.items(), key=lambda x: -x[1])),
#                 })
#         result_map = dict(zip(valid_indices, valid_results))
#         return [result_map.get(i, dict(_empty)) for i in range(len(texts))]

#     @property
#     def labels(self) -> list[str]:
#         return EMOTION_LABELS

#     def __repr__(self) -> str:
#         status = "loaded" if self._loaded else "not loaded"
#         return f"EmotionClassifier(model_dir='{self._dir}', device={self._device}, {status})"


# """
# src/modules/emotion_classifier.py
# ===================================

import os
import pickle
from pathlib import Path
import yaml
import torch
import torch.nn as nn
from transformers import BertTokenizer
# Import the PyTorch module architecture you provided
from src.emotion_classifier.model import EmotionClassifierModel as BiLSTMNet

# ── Mappings & Constants ──────────────────────────────────────────────────────
EMOTION_LABELS = ['sadness','joy','love','anger','fear','surprise']

ID2LABEL = {i: label for i, label in enumerate(EMOTION_LABELS)}
LABEL2ID = {label: i for i, label in enumerate(EMOTION_LABELS)}

_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_MODEL_DIR = _ROOT / "models" / "emotion_classifier"

class EmotionClassifier:
    """
    Runtime inference wrapper for the custom Bidirectional LSTM Emotion Classifier.
    Lazy-loads weights, configs, and your custom tokenizer on the first API call.
    """
    def __init__(self, model_dir: str | Path | None = None, device: str | None = None) -> None:
        self._dir = Path(model_dir) if model_dir else _DEFAULT_MODEL_DIR
        self._device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        
        # Placeholders for lazy loading
        self._model = None
        self._tokenizer = None
        self._max_len = 128
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return

        from transformers import BertTokenizerFast

        cfg_path = self._dir / "emotion_config.yaml"
        weights_path = self._dir / "emotion_classifier_best_model.pt"
        tok_path = self._dir / "tokenizer.pkl"

        if not cfg_path.exists() or not weights_path.exists():
            raise FileNotFoundError(
                f"Required model files missing in {self._dir}.\n"
                f"Ensure 'emotion_config.yaml' and 'emotion_classifier_best_model.pt' are present."
            )

        print(f"📖 Reading configuration from {cfg_path}...")
        with open(cfg_path, "r") as f:
            yaml_data = yaml.safe_load(f) or {}

        # Helper function to smoothly extract from the nested sweep "value" format
        def get_val(key: str, default):
            item = yaml_data.get(key)
            if isinstance(item, dict) and "value" in item:
                return item["value"]
            return default

        # ── 1. EXTRACT NESTED VALUES AND ENFORCE SCALAR TYPES ────────────────
        config_data = {
            "embed_dim":     int(get_val("embed_dim", 128)),
            "lstm_units":    int(get_val("lstm_units", 128)),
            "num_layers":    int(get_val("num_layers", 2)),       # Defaulting to 2 layers
            "dropout":       float(get_val("dropout", 0.4)),
            "fc_hidden_dim": int(get_val("fc_hidden_dim", 64)),   # Defaulting to 64 hidden units
            "num_classes":   int(get_val("num_classes", 6)),      # 6 core target emotions
            "vocab_size":    int(get_val("vocab_size", 30522)),   # bert-base-uncased matrix limit
        }
        
        # Capture maximum token length constraint for tensor sizing (fallback to 50)
        self._max_len = int(get_val("max_len", 50))
        # ──────────────────────────────────────────────────────────────────────

        print(f"📥 Initializing Tokenizer (bert-base-uncased)...")
        self._tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        print(f"🧠 Building BiLSTM and injecting state dict...")
        self._model = BiLSTMNet(config=config_data)
        
        # ── 2. UNBOX CHECKPOINT WRAPPER FOR PYTORCH LAYERS ────────────────────
        checkpoint = torch.load(weights_path, map_location=self._device)
        
        # Pull out raw state dict vectors if wrapped up inside training metadata structures
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            print("📦 Training checkpoint wrapper detected. Extracting 'model_state' weights...")
            real_state_dict = checkpoint["model_state"]
        else:
            real_state_dict = checkpoint

        # Bind the weights onto your instantiated network
        self._model.load_state_dict(real_state_dict)
        self._model.to(self._device)
        self._model.eval()

        self._loaded = True
        print("✅ BiLSTM Emotion Classifier state weights successfully mapped and loaded!")

    def predict(self, text: str) -> dict:
        """Alias to keep consistent with the FastAPI endpoint expectations."""
        return self.classify(text)

    def classify(self, text: str) -> dict:
        """
        Tokenizes text using the BertTokenizerFast, handles padding and truncation 
        to max_len, passes it through the BiLSTM, and computes emotion probabilities.
        """
        self._load()

        if not isinstance(text, str) or not text.strip():
            return {"emotion": "unknown", "confidence": 0.0, "all_scores": {}}

        # ── 1. HUGGING FACE AUTOMATIC TOKENIZATION & PADDING ──────────────────
        # This handles truncation, adds special tokens ([CLS], [SEP]), balances 
        # padding up to your exact max_len (50), and returns pure PyTorch tensors.
        encoded_inputs = self._tokenizer(
            text,
            max_length=self._max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Extract the input matrices and push them to your target device (CPU/CUDA)
        input_ids = encoded_inputs["input_ids"].to(self._device)
        attention_mask = encoded_inputs["attention_mask"].to(self._device)

        # ── 2. MODEL EVALUATION PASS ──────────────────────────────────────────
        with torch.no_grad():
            logits = self._model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

        # ── 3. FORMAT INFERENCE METRICS RESPONSE ──────────────────────────────
        all_scores = {ID2LABEL[i]: round(float(p), 4) for i, p in enumerate(probs)}
        top_emotion = max(all_scores, key=all_scores.__getitem__)

        return {
            "emotion": top_emotion,
            "confidence": all_scores[top_emotion],
            "all_scores": dict(sorted(all_scores.items(), key=lambda x: -x[1]))
        }