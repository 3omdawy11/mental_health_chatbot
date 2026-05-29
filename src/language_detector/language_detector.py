"""
Inference wrapper for the TF-IDF + Logistic Regression language detector.

Usage
-----
    from src.modules.language_detector import LanguageDetector

    detector = LanguageDetector()                  # loads saved model
    result   = detector.detect("Hello, how are you?")
    # {"language": "en", "confidence": 0.99, "all_scores": {"en": 0.99, ...}}
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

# ── Paths (relative to project root) ─────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent.parent

# ── Detect environment ───────────────────────────────────────────────────────
_IN_KAGGLE = "/kaggle" in str(Path.cwd())

try:
    # Kaggle environment
    if _IN_KAGGLE:
        _DEFAULT_MODEL_DIR = Path("/kaggle/working/models/language_detection")
    else:
        raise ValueError("Not in Kaggle")
except:
    # Local environment fallback
    _DEFAULT_MODEL_DIR = _ROOT / "models" / "language_detection"


class LanguageDetector:
    """
    Thin wrapper around the saved TF-IDF vectorizer + Logistic Regression
    classifier for language identification.

    Parameters
    ----------
    model_dir : path to the directory containing model.pkl, vectorizer.pkl,
                config.yaml.  Defaults to models/language_detection/.
    """

    def __init__(self, model_dir: str | Path | None = None) -> None:
        self._dir = Path(model_dir) if model_dir else _DEFAULT_MODEL_DIR
        self._vectorizer = None
        self._model = None
        self._classes: list[str] = []
        self._language_names: dict[str, str] = {}
        self._loaded = False

    # ── Lazy loading ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        if self._loaded:
            return

        vec_path = self._dir / "vectorizer.pkl"
        mdl_path = self._dir / "model.pkl"
        cfg_path = self._dir / "config.yaml"

        if not vec_path.exists() or not mdl_path.exists():
            raise FileNotFoundError(
                f"Model files not found in {self._dir}.\n"
                "Run  python scripts/01_train_language_detector.py  first."
            )

        with open(vec_path, "rb") as f:
            self._vectorizer = pickle.load(f)
        with open(mdl_path, "rb") as f:
            self._model = pickle.load(f)

        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            self._language_names = cfg.get("language_names", {})

        self._classes = list(self._model.classes_)
        self._loaded = True

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(self, text: str) -> dict:
        """
        Detect the language of *text*.

        Returns
        -------
        {
            "language"   : "en",          # ISO 639-1 code
            "language_name": "English",   # human-readable name
            "confidence" : 0.98,          # probability of top prediction
            "all_scores" : {"en": 0.98, "fr": 0.01, ...}
        }
        """
        self._load()

        if not isinstance(text, str) or not text.strip():
            return {
                "language": "unknown",
                "language_name": "Unknown",
                "confidence": 0.0,
                "all_scores": {},
            }

        X = self._vectorizer.transform([text])
        proba = self._model.predict_proba(X)[0]

        all_scores = {
            lang: round(float(p), 4)
            for lang, p in sorted(
                zip(self._classes, proba), key=lambda x: -x[1]
            )
        }

        top_lang = max(all_scores, key=all_scores.__getitem__)

        return {
            "language": top_lang,
            "language_name": self._language_names.get(top_lang, top_lang),
            "confidence": all_scores[top_lang],
            "all_scores": all_scores,
        }

    def detect_batch(self, texts: list[str]) -> list[dict]:
        """Detect languages for a list of texts (vectorised — fast)."""
        self._load()

        results = []
        valid_mask = [isinstance(t, str) and bool(t.strip()) for t in texts]
        valid_texts = [t for t, ok in zip(texts, valid_mask) if ok]

        if valid_texts:
            X = self._vectorizer.transform(valid_texts)
            probas = self._model.predict_proba(X)
        else:
            probas = np.empty((0, len(self._classes)))

        proba_iter = iter(probas)
        for ok in valid_mask:
            if ok:
                proba = next(proba_iter)
                all_scores = {
                    lang: round(float(p), 4)
                    for lang, p in sorted(
                        zip(self._classes, proba), key=lambda x: -x[1]
                    )
                }
                top_lang = max(all_scores, key=all_scores.__getitem__)
                results.append(
                    {
                        "language": top_lang,
                        "language_name": self._language_names.get(top_lang, top_lang),
                        "confidence": all_scores[top_lang],
                        "all_scores": all_scores,
                    }
                )
            else:
                results.append(
                    {
                        "language": "unknown",
                        "language_name": "Unknown",
                        "confidence": 0.0,
                        "all_scores": {},
                    }
                )
        return results

    @property
    def supported_languages(self) -> list[str]:
        self._load()
        return self._classes

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "not loaded"
        return f"LanguageDetector(model_dir='{self._dir}', status={status})"