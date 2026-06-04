"""
Train TF-IDF + Logistic Regression language detector.

Steps
-----
1. Load train / val / test splits
2. Fit TF-IDF vectoriser on train
3. Grid-search over C values using validation set
4. Retrain on train+val with best C
5. Evaluate on test set (accuracy, per-class F1, confusion matrix)
6. Save vectorizer.pkl, model.pkl
7. Print full summary

Run from project root:
    python scripts/01_train_language_detector.py
"""

import sys, time, pickle, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")          # headless — saves PNG, no display needed
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
)

# ── Paths ────────────────────────────────────────────────────────────────────
# ── Detect environment ────────────────────────────────────────────────────────
#
# Kaggle has two zones:
#   /kaggle/input  — read-only  → where datasets (splits) live
#   /kaggle/working — writable  → where we write models, logs, artefacts
#
# We detect Kaggle by checking for the canonical dataset path; if it exists
# we use it, otherwise we fall back to the local project layout.
# ─────────────────────────────────────────────────────────────────────────────

_KAGGLE_INPUT_SPLITS = Path(
    "/kaggle/input/datasets/ziadmahmoudamr/mental-health-chatbot"
    "/mental_health_chatbot/data/splits"
)

IN_KAGGLE = _KAGGLE_INPUT_SPLITS.exists()

if IN_KAGGLE:
    # READ  → /kaggle/input  (read-only dataset)
    DATA_SPLITS = _KAGGLE_INPUT_SPLITS
    # WRITE → /kaggle/working  (writable scratch space)
    MODEL_DIR   = Path("/kaggle/working/models/language_detection")
    LOGS        = Path("/kaggle/working/logs")
else:
    # Local project layout
    DATA_SPLITS = ROOT / "data" / "splits"
    MODEL_DIR   = ROOT / "models" / "language_detection"
    LOGS        = ROOT / "logs"

# Only create writable directories (never touch /kaggle/input)
for d in [MODEL_DIR, LOGS]:
    d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Language priority list for low-confidence fallback
#
# When the model is uncertain (confidence < threshold), we scan the top-N
# predicted languages and return the highest-priority one that appears there.
# Order reflects approximate global speaker population / web prevalence.
# Edit freely — only languages your model actually knows matter here.
# ─────────────────────────────────────────────────────────────────────────────
LANGUAGE_PRIORITY = [
    "en",   # English        ~1.5 B speakers
    "zh",   # Chinese        ~1.1 B
    "hi",   # Hindi          ~600 M
    "es",   # Spanish        ~560 M
    "fr",   # French         ~280 M
    "ar",   # Arabic         ~270 M
    "bn",   # Bengali        ~270 M
    "pt",   # Portuguese     ~260 M
    "ru",   # Russian        ~255 M
    "ur",   # Urdu           ~230 M
    "id",   # Indonesian     ~200 M
    "de",   # German         ~135 M
    "ja",   # Japanese       ~125 M
    "tr",   # Turkish        ~88 M
    "ko",   # Korean         ~82 M
    "vi",   # Vietnamese     ~77 M
    "it",   # Italian        ~68 M
    "fa",   # Persian        ~65 M
    "pl",   # Polish         ~45 M
    "nl",   # Dutch          ~30 M
    "sw",   # Swahili        ~20 M
]


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Load data
# ─────────────────────────────────────────────────────────────────────────────

def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("\n[1/6] Loading splits …")
    print(f"  Reading from: {DATA_SPLITS}")
    train = pd.read_csv(DATA_SPLITS / "train" / "language_train.csv")
    val   = pd.read_csv(DATA_SPLITS / "val"   / "language_val.csv")
    test  = pd.read_csv(DATA_SPLITS / "test"  / "language_test.csv")
    print(f"  train={len(train):,}  val={len(val):,}  test={len(test):,}")
    print(f"  Languages: {sorted(train['labels'].unique())}")
    return train, val, test


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Vectoriser
# ─────────────────────────────────────────────────────────────────────────────

def build_vectorizer(train_texts: pd.Series) -> TfidfVectorizer:
    print("\n[2/6] Fitting TF-IDF vectoriser …")
    vec = TfidfVectorizer(
        max_features=50_000,
        ngram_range=(2, 6),      # char n-grams 2-6
        analyzer="char_wb",      # character-level, word boundaries added
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )
    t0 = time.time()
    vec.fit(train_texts)
    print(f"  Vocabulary size : {len(vec.vocabulary_):,}")
    print(f"  Fit time        : {time.time()-t0:.1f}s")
    return vec

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Grid search over C
# ─────────────────────────────────────────────────────────────────────────────

def grid_search_C(
    X_train, y_train, X_val, y_val,
    C_values: list[float] | None = None,
) -> float:
    print("\n[3/6] Grid-searching regularisation C on validation set …")
    if C_values is None:
        C_values = [0.1, 0.5, 1.0, 5.0, 10.0]

    results = []
    for C in C_values:
        t0 = time.time()
        clf = LogisticRegression(
            C=C, max_iter=1000, solver="lbfgs",
            random_state=42, n_jobs=-1,
        )
        clf.fit(X_train, y_train)
        acc = accuracy_score(y_val, clf.predict(X_val))
        elapsed = time.time() - t0
        results.append((C, acc, elapsed))
        print(f"  C={C:<5}  val_acc={acc:.4f}  ({elapsed:.1f}s)")

    best_C, best_acc, _ = max(results, key=lambda x: x[1])
    print(f"\n  ✓ Best C={best_C}  val_accuracy={best_acc:.4f}")
    return best_C


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Train final model on train + val
# ─────────────────────────────────────────────────────────────────────────────

def train_final(X_trainval, y_trainval, best_C: float) -> LogisticRegression:
    print(f"\n[4/6] Training final model (C={best_C}) on train+val …")
    clf = LogisticRegression(
        C=best_C, max_iter=1000, solver="lbfgs",
        random_state=42, n_jobs=-1,
    )
    t0 = time.time()
    clf.fit(X_trainval, y_trainval)
    print(f"  Training time: {time.time()-t0:.1f}s")
    return clf


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(clf, vec, X_test, y_test, labels: list[str]) -> dict:
    print("\n[5/6] Evaluating on test set …")
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=labels, output_dict=True)
    report_str = classification_report(y_test, y_pred, target_names=labels)

    print(f"\n  Overall accuracy : {acc:.4f}  ({acc*100:.2f}%)")
    print("\n" + report_str)

    # Log to file
    with open(LOGS / "model_performance.log", "a") as f:
        f.write(f"\n{'='*60}\nLanguage Detector — Test Evaluation\n")
        f.write(f"Accuracy: {acc:.4f}\n\n{report_str}\n")

    # ── Confusion matrix PNG ──────────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        linewidths=0.4, ax=ax, cbar=True,
    )
    ax.set_title(f"Language Detection — Confusion Matrix  (acc={acc:.4f})", fontsize=14)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    plt.tight_layout()
    cm_path = MODEL_DIR / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=120)
    plt.close(fig)
    print(f"\n  Confusion matrix saved → {cm_path}")

    # ── Per-language bar chart ────────────────────────────────────────────────
    per_lang = {
        lang: {
            "precision": report[lang]["precision"],
            "recall":    report[lang]["recall"],
            "f1-score":  report[lang]["f1-score"],
        }
        for lang in labels
        if lang in report
    }
    _plot_per_language(per_lang, MODEL_DIR / "per_language_metrics.png")

    # ── Hardest languages ─────────────────────────────────────────────────────
    sorted_langs = sorted(per_lang.items(), key=lambda x: x[1]["f1-score"])
    print("\n  5 hardest languages (lowest F1):")
    for lang, m in sorted_langs[:5]:
        print(f"    {lang}  F1={m['f1-score']:.3f}  P={m['precision']:.3f}  R={m['recall']:.3f}")

    return {"accuracy": acc, "per_language": per_lang, "report": report}


def _plot_per_language(per_lang: dict, save_path: Path) -> None:
    langs = list(per_lang.keys())
    f1s   = [per_lang[l]["f1-score"]  for l in langs]
    precs = [per_lang[l]["precision"] for l in langs]
    recs  = [per_lang[l]["recall"]    for l in langs]

    x = np.arange(len(langs))
    w = 0.26
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.bar(x - w, precs, w, label="Precision", color="#3498db")
    ax.bar(x,     f1s,   w, label="F1",        color="#2ecc71")
    ax.bar(x + w, recs,  w, label="Recall",    color="#e74c3c")
    ax.set_xticks(x)
    ax.set_xticklabels(langs, rotation=45, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Per-language Precision / F1 / Recall", fontsize=13)
    ax.axhline(0.90, color="grey", linestyle="--", linewidth=0.8, label="0.90 target")
    ax.legend()
    plt.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"  Per-language chart saved → {save_path}")


def top_features_per_language(vec: TfidfVectorizer, clf: LogisticRegression,
                               n: int = 15) -> None:
    """Print top TF-IDF features for each language class."""
    print("\n  Top features per language (char n-grams):")
    feature_names = np.array(vec.get_feature_names_out())
    for i, lang in enumerate(clf.classes_):
        top_idx = np.argsort(clf.coef_[i])[-n:][::-1]
        top_feats = feature_names[top_idx]
        print(f"  {lang}: {', '.join(repr(f) for f in top_feats[:8])}")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Save model
# ─────────────────────────────────────────────────────────────────────────────

def save_model(vec: TfidfVectorizer, clf: LogisticRegression,
               eval_results: dict, best_C: float) -> None:
    print("\n[6/6] Saving model artefacts …")

    with open(MODEL_DIR / "vectorizer.pkl", "wb") as f:
        pickle.dump(vec, f)
    with open(MODEL_DIR / "model.pkl", "wb") as f:
        pickle.dump(clf, f)

    # Update config.yaml with training outcomes
    cfg_path = MODEL_DIR / "config.yaml"
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {}

    # Ensure nested dictionaries exist
    if "model" not in cfg:
        cfg["model"] = {}
    if "training" not in cfg:
        cfg["training"] = {}

    cfg["model"]["C"] = best_C
    cfg["training"]["test_accuracy"] = round(eval_results["accuracy"], 4)
    cfg["training"]["languages_trained"] = list(clf.classes_)

    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    print(f"  vectorizer.pkl → {MODEL_DIR / 'vectorizer.pkl'}")
    print(f"  model.pkl      → {MODEL_DIR / 'model.pkl'}")
    print(f"  config.yaml    → {cfg_path}")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print(" language Detection Model Training")
    print("=" * 60)

    # 1. Load
    train, val, test = load_splits()

    # 2. Vectoriser (fit on train only)
    vec = build_vectorizer(train["text"])
    X_train = vec.transform(train["text"])
    X_val   = vec.transform(val["text"])
    X_test  = vec.transform(test["text"])

    y_train, y_val, y_test = train["labels"], val["labels"], test["labels"]

    # 3. Grid search C
    best_C = grid_search_C(X_train, y_train, X_val, y_val)

    # 4. Retrain on train+val
    import scipy.sparse as sp
    X_tv = sp.vstack([X_train, X_val])
    y_tv = pd.concat([y_train, y_val], ignore_index=True)
    clf  = train_final(X_tv, y_tv, best_C)

    # 5. Evaluate
    labels = sorted(train["labels"].unique().tolist())
    eval_results = evaluate(clf, vec, X_test, y_test, labels)

    # Feature inspection
    top_features_per_language(vec, clf, n=15)

    # 6. Save
    save_model(vec, clf, eval_results, best_C)

    # ── Smoke-test the wrapper class ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Wrapper class smoke test")
    print("=" * 60)
    from src.language_detector import LanguageDetector
    detector = LanguageDetector()
    test_cases = [
        ("Hello, how are you feeling today?", "en"),
        ("Je me sens très triste aujourd'hui.", "fr"),
        ("Ich fühle mich heute nicht gut.", "de"),
        ("今日は気分が優れません。", "ja"),
        ("أنا لست بخير اليوم.", "ar"),
        ("hey", "en"),
        ("hi", "en"),
        ("hello", "en"),
        ("3amel eh yasta", "en"),
        ("الو", "ar"),



    ]
    all_ok = True
    for text, expected in test_cases:
        result = detector.detect(text)
        ok = result["language"] == expected
        all_ok = all_ok and ok
        status = "✓" if ok else "✗"
        print(f"  {status}  [{expected}→{result['language']}]  conf={result['confidence']:.3f}  {text[:45]}")

    print(f"\n  Smoke test: {'PASSED ✅' if all_ok else 'FAILED ❌'}")

    print("\n" + "=" * 60)
    print("  ✅  Phase 2 Complete")
    print(f"  Test accuracy : {eval_results['accuracy']*100:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()