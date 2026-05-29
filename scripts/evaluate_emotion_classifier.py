# scripts/evaluate_emotion_classifier.py

import sys
sys.path.insert(0, '.')

import torch
from src.emotion_classifier import load_data, load_tokenizer, get_dataloaders, EmotionClassifier, evaluate
from config import BASELINE_RUN_CONFIG

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_df, val_df, test_df = load_data()
tokenizer = load_tokenizer()
_, _, test_loader = get_dataloaders(train_df, val_df, test_df, tokenizer, BASELINE_RUN_CONFIG)

model = EmotionClassifier(BASELINE_RUN_CONFIG).to(device)

test_acc, preds, labels = evaluate(model, test_loader, device)