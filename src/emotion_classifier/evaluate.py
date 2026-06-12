import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from config import MODEL_SAVE_PATH, BASELINE_RUN_CONFIG

LABEL_MAP = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}


def evaluate(model, test_loader, device, run_config=BASELINE_RUN_CONFIG):

    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    print(f"[EVAL] Loaded best checkpoint from epoch {checkpoint['epoch']}")
    print(f"[EVAL] Checkpoint val_acc: {checkpoint['val_acc']:.4f}")

    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['label'].to(device)

            logits = model(input_ids, attention_mask)
            preds  = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    test_acc = (all_preds == all_labels).mean()
    print(f"\n[EVAL] Test Accuracy: {test_acc:.4f}")

    print("\n[EVAL] Classification Report:")
    print(classification_report(
        all_labels,
        all_preds,
        target_names=list(LABEL_MAP.values())
    ))

    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot      = True,
        fmt        = 'd',
        cmap       = 'Blues',
        xticklabels = list(LABEL_MAP.values()),
        yticklabels = list(LABEL_MAP.values())
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f"Confusion Matrix — {run_config['name']}")
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()
    print("[EVAL] Confusion matrix saved to confusion_matrix.png")

    return test_acc, all_preds, all_labels