import sys, argparse, json, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, accuracy_score
)
from src.modules.emotion_classifier import (
    Trainer, EmotionClassifier, EMOTION_LABELS
)

SPLITS    = ROOT / "data" / "splits"
MODEL_DIR = ROOT / "models" / "emotion_classifier"
LOGS      = ROOT / "logs"
LOGS.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
def load_splits():
    print("\n[1/5] Loading emotion splits …")
    train = pd.read_csv(SPLITS / "emotion_train.csv")
    val   = pd.read_csv(SPLITS / "emotion_val.csv")
    test  = pd.read_csv(SPLITS / "emotion_test.csv")
    print(f"  train={len(train):,}  val={len(val):,}  test={len(test):,}")
    print("  Train class distribution:")
    for emo, cnt in train["emotion"].value_counts().items():
        pct = cnt / len(train) * 100
        bar = "█" * int(pct / 2)
        print(f"    {emo:<10} {cnt:>5}  {pct:>5.1f}%  {bar}")
    vc    = train["emotion"].value_counts()
    ratio = vc.max() / vc.min()
    print(f"\n  Imbalance ratio  : {ratio:.1f}x  ({vc.idxmax()} vs {vc.idxmin()})")
    print( "  Strategy chosen  : weighted cross-entropy loss")
    return train, val, test


def plot_training_history(history: list[dict], save_path: Path) -> None:
    if not history:
        return
    epochs     = [r["epoch"]      for r in history]
    train_loss = [r["train_loss"] for r in history]
    val_loss   = [r["val_loss"]   for r in history]
    val_acc    = [r["val_acc"]    for r in history]
    val_f1     = [r["val_f1"]     for r in history]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(epochs, train_loss, "o-", label="train", color="#3498db")
    axes[0].plot(epochs, val_loss,   "o-", label="val",   color="#e74c3c")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, val_acc, "o-", color="#2ecc71")
    axes[1].set_title("Val Accuracy"); axes[1].set_xlabel("Epoch"); axes[1].set_ylim(0, 1)

    axes[2].plot(epochs, val_f1, "o-", color="#9b59b6")
    axes[2].set_title("Val Macro-F1"); axes[2].set_xlabel("Epoch"); axes[2].set_ylim(0, 1)

    plt.suptitle("DistilBERT Emotion Classifier — Training History", fontsize=13)
    plt.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"  Training history → {save_path}")


def evaluate_test(test: pd.DataFrame) -> dict:
    print("\n[4/5] Evaluating on test set …")
    clf = EmotionClassifier()
    t0  = time.time()
    results = clf.classify_batch(test["text"].tolist())
    elapsed = time.time() - t0

    y_pred = [r["emotion"] for r in results]
    y_true = test["emotion"].tolist()
    acc    = accuracy_score(y_true, y_pred)
    f1     = f1_score(y_true, y_pred, average="macro")
    report = classification_report(y_true, y_pred, target_names=EMOTION_LABELS)

    print(f"\n  Test accuracy   : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Test macro-F1   : {f1:.4f}")
    print(f"  Inference speed : {elapsed/len(test)*1000:.1f} ms/sample  "
          f"(< 1000 ms target {'ok' if elapsed/len(test)*1000 < 1000 else '✗'})")
    print(f"\n{report}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=EMOTION_LABELS)
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
                xticklabels=EMOTION_LABELS, yticklabels=EMOTION_LABELS,
                linewidths=0.4, ax=ax)
    ax.set_title(f"Emotion Classifier — Confusion Matrix  (acc={acc:.4f})", fontsize=12)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout()
    cm_path = MODEL_DIR / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=120); plt.close(fig)
    print(f"  Confusion matrix → {cm_path}")

    # Per-emotion grouped bar
    cr_dict = classification_report(y_true, y_pred, output_dict=True)
    emotions = EMOTION_LABELS
    prec = [cr_dict.get(e, {}).get("precision", 0) for e in emotions]
    rec  = [cr_dict.get(e, {}).get("recall",    0) for e in emotions]
    f1s  = [cr_dict.get(e, {}).get("f1-score",  0) for e in emotions]

    x = range(len(emotions)); w = 0.26
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar([i - w for i in x], prec, w, label="Precision", color="#3498db")
    ax.bar(list(x),             f1s,  w, label="F1",        color="#2ecc71")
    ax.bar([i + w for i in x], rec,  w, label="Recall",    color="#e74c3c")
    ax.set_xticks(list(x)); ax.set_xticklabels(emotions)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.75, color="grey", linestyle="--", linewidth=0.8, label="0.75 target")
    ax.set_title("Per-emotion Precision / F1 / Recall", fontsize=12)
    ax.legend(); ax.set_ylabel("Score"); plt.tight_layout()
    pe_path = MODEL_DIR / "per_emotion_metrics.png"
    fig.savefig(pe_path, dpi=120); plt.close(fig)
    print(f"  Per-emotion chart → {pe_path}")

    # Hardest emotions
    sorted_emo = sorted(
        [(e, cr_dict.get(e, {}).get("f1-score", 0)) for e in emotions],
        key=lambda x: x[1]
    )
    print("\n  Hardest emotions (lowest F1):")
    for emo, f in sorted_emo[:3]:
        print(f"    {emo:<10} F1={f:.3f}")

    # Log
    with open(LOGS / "model_performance.log", "a") as fh:
        fh.write(f"\n{'='*60}\nEmotion Classifier — Test Evaluation\n")
        fh.write(f"Accuracy: {acc:.4f}  |  Macro-F1: {f1:.4f}\n\n{report}\n")

    return {"accuracy": acc, "macro_f1": f1}


# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",     type=int,   default=5)
    p.add_argument("--lr",         type=float, default=2e-5)
    p.add_argument("--batch_size", type=int,   default=32)
    p.add_argument("--max_length", type=int,   default=128)
    p.add_argument("--patience",   type=int,   default=2)
    p.add_argument("--model_name", type=str,   default="distilbert-base-uncased")
    p.add_argument("--skip_train", action="store_true",
                   help="Skip training and evaluate existing model")
    return p.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print("  Emotion Classifier (DistilBERT)")
    print("=" * 60)

    train, val, test = load_splits()

    if not args.skip_train:
        print("\n[2/5] Setting up trainer …")
        trainer = Trainer(
            model_name=args.model_name,
            model_dir=MODEL_DIR,
            lr=args.lr,
            batch_size=args.batch_size,
            epochs=args.epochs,
            max_length=args.max_length,
            patience=args.patience,
        )
        print("\n[3/5] Fine-tuning DistilBERT …")
        history = trainer.fit(
            train["text"].tolist(), train["emotion"].tolist(),
            val["text"].tolist(),   val["emotion"].tolist(),
        )
        trainer.save_final()
        plot_training_history(history, MODEL_DIR / "training_history.png")
    else:
        print("\n  [skip_train] Using existing model.")
        history = []

    eval_results = evaluate_test(test)

    # Smoke test
    print("\n[5/5] Wrapper smoke test …")
    clf = EmotionClassifier()
    smoke_cases = [
        ("I feel absolutely hopeless and want to give up everything", "sadness"),
        ("This is the best day of my life, I am so happy!",           "joy"),
        ("I am furious about what just happened",                     "anger"),
        ("I am so scared and nervous, I cannot stop shaking",         "fear"),
        ("I love you so much, you mean everything to me",             "love"),
        ("Wow I never expected that, completely shocked!",            "surprise"),
    ]
    all_ok = True
    for text, expected in smoke_cases:
        t0  = time.time()
        res = clf.classify(text)
        ms  = (time.time() - t0) * 1000
        ok  = res["emotion"] == expected
        all_ok = all_ok and ok
        mark = "ok" if ok else "✗"
        print(f"  {mark} [{expected:<8}→{res['emotion']:<8}]  "
              f"conf={res['confidence']:.3f}  {ms:.0f}ms  {text[:42]}")

    print(f"\n  test: {'PASSED' if all_ok else 'PARTIAL — see above'}")
    print("\n" + "=" * 60)
    print(f"  Test accuracy : {eval_results['accuracy']*100:.2f}%")
    print(f"  Test macro-F1 : {eval_results['macro_f1']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()