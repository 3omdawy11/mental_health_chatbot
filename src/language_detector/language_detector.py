"""
I have come up with a new version of the language detector that adds probability thresholding and an optional short-text ensemble with langid.  The goal is to improve accuracy on short, ambiguous texts (e.g. "hi") where the model may be uncertain and prone to picking visually-similar but less common languages.
──────────────────────────────────────────
1. Probability thresholding
   After predict_proba(), if max confidence < CONFIDENCE_THRESHOLD the model
   is considered uncertain.  In that case we walk the top-K predicted languages
   and return the first one that appears in LANGUAGE_PRIORITY (most-spoken
   languages globally).  This prevents rare / visually-similar languages
   (Swahili, Italian, …) from "winning" on short, ambiguous text like "hi".

2. Optional short-text ensemble  (USE_SHORT_TEXT_ENSEMBLE = True/False)
   When enabled, texts shorter than SHORT_TEXT_THRESHOLD characters are first
   passed to langid (lightweight, pure-Python, ships as a single file).
   - If langid's result exists in our model's known classes it is used directly.
   - Otherwise we fall back to our thresholded model prediction.
   Toggle USE_SHORT_TEXT_ENSEMBLE = False (or set env-var
   LANG_DETECTOR_ENSEMBLE=0) to disable completely — zero import cost.

"""

from __future__ import annotations

import os
import pickle
import warnings
from pathlib import Path
from typing import Any

import numpy as np

# Minimum confidence for the model's top prediction to be accepted as-is.
CONFIDENCE_THRESHOLD: float = 0.55

# How many top candidates to consider when doing the priority fallback.
FALLBACK_TOP_K: int = 10

# Set to True  → use langid for texts shorter than SHORT_TEXT_THRESHOLD chars.
# Set to False → always use our TF-IDF/LR model (+ thresholding above).
USE_SHORT_TEXT_ENSEMBLE: bool = True 

SHORT_TEXT_THRESHOLD: int = 30

# ─────────────────────────────────────────────────────────────────────────────
# ❷  LANGUAGE PRIORITY LIST
#    When the model is uncertain, we prefer whichever of these languages
#    appears highest in the model's top-K candidates.
#    Order = approximate global speaker population / web prevalence.
#    Only languages our model actually knows matter, others are ignored.
# ─────────────────────────────────────────────────────────────────────────────
LANGUAGE_PRIORITY: list[str] = [
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

_ROOT = Path(__file__).resolve().parent.parent.parent

if "/kaggle" in str(Path.cwd()):
    _MODEL_DIR = Path("/kaggle/working/models/language_detection")
else:
    _MODEL_DIR = _ROOT / "models" / "language_detection"



def _ensemble_enabled() -> bool:
    env = os.environ.get("LANG_DETECTOR_ENSEMBLE")
    if env is not None:
        return env.strip() not in ("0", "false", "False", "no")
    return USE_SHORT_TEXT_ENSEMBLE


def _try_load_langid():
    try:
        import langid
        return langid
    except ImportError:
        warnings.warn(
            "langid is not installed — short-text ensemble is disabled. "
            "Run `pip install langid` to enable it.",
            RuntimeWarning,
            stacklevel=3,
        )
        return None



class LanguageDetector:

    def __init__(
        self,
        model_dir: "Path | str | None" = None,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        fallback_top_k: int = FALLBACK_TOP_K,
        use_ensemble: "bool | None" = None,
        short_text_threshold: int = SHORT_TEXT_THRESHOLD,
    ) -> None:
        self.model_dir           = Path(model_dir) if model_dir else _MODEL_DIR
        self.confidence_threshold = confidence_threshold
        self.fallback_top_k      = fallback_top_k
        self.short_text_threshold = short_text_threshold
        self._loaded = False


        if use_ensemble is None:
            self._use_ensemble = _ensemble_enabled()
        else:
            self._use_ensemble = use_ensemble

        with open(self.model_dir / "vectorizer.pkl", "rb") as f:
            self.vectorizer = pickle.load(f)
        with open(self.model_dir / "model.pkl", "rb") as f:
            self.model = pickle.load(f)

        self._known_classes: set[str] = set(self.model.classes_)

        self._priority_index: dict[str, int] = {
            lang: rank
            for rank, lang in enumerate(LANGUAGE_PRIORITY)
            if lang in self._known_classes
        }

        self._langid = None
        if self._use_ensemble:
            self._langid = _try_load_langid()
            if self._langid is None:
                self._use_ensemble = False   # graceful degradation
        self._loaded = True


    def detect(self, text: str) -> dict[str, Any]:
        text = text.strip()

        if self._use_ensemble and len(text) < self.short_text_threshold:
            return self._detect_with_ensemble(text)

        return self._detect_with_model(text)

    def detect_batch(self, texts: "list[str]") -> "list[dict[str, Any]]":
        return [self.detect(t) for t in texts]


    def _detect_with_model(self, text: str) -> dict[str, Any]:
        X        = self.vectorizer.transform([text])
        proba    = self.model.predict_proba(X)[0]          # shape: (n_classes,)
        top_conf = float(proba.max())
        top_idx  = int(proba.argmax())
        top_lang = self.model.classes_[top_idx]
        all_scores = {lang: float(prob) for lang, prob in zip(self.model.classes_, proba)}
        if top_conf >= self.confidence_threshold:
            return {
                "language":   top_lang,
                "confidence": top_conf,
                "method":     "tfidf_lr",
                "all_scores": all_scores, 
            }

        # Low confidence: priority fallback
        # Get indices of top-K candidates (unsorted; we'll sort below)
        k          = min(self.fallback_top_k, len(proba))
        topk_idx   = np.argpartition(proba, -k)[-k:]          # top-K indices (unordered)
        topk_langs = [self.model.classes_[i] for i in topk_idx]

        # Find the highest-priority language among the top-K candidates
        best_lang = None
        best_rank = float("inf")
        for lang in topk_langs:
            rank = self._priority_index.get(lang, float("inf"))
            if rank < best_rank:
                best_rank = rank
                best_lang = lang

        if best_lang is not None:
            # Confidence = the probability assigned to that language
            best_lang_idx  = list(self.model.classes_).index(best_lang)
            best_confidence = float(proba[best_lang_idx])
            return {
                "language":   best_lang,
                "confidence": best_confidence,
                "method":     "tfidf_lr_priority_fallback",
                "all_scores": all_scores, 
            }

        return {
            "language":   top_lang,
            "confidence": top_conf,
            "method":     "tfidf_lr_raw_fallback",
            "all_scores": all_scores, 
        }

    def _detect_with_ensemble(self, text: str) -> dict[str, Any]:
        lang_code, log_prob = self._langid.classify(text)

        if lang_code in self._known_classes:
            # Convert log-prob to a 0-1 confidence proxy via sigmoid-style clip
            # langid log-probs are typically large negative numbers;
            # we normalise to [0, 1] just for consistency.
            confidence = float(np.clip(1.0 / (1.0 + np.exp(-log_prob * 0.05)), 0.0, 1.0))
            return {
                "language":   lang_code,
                "confidence": confidence,
                "method":     "langid_ensemble",
                "all_scores": None,
            }

        result = self._detect_with_model(text)
        result["method"] = "model_fallback_from_langid"
        return result
    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "not loaded"
        return f"LanguageDetector(model_dir='{self.model_dir}', status={status})"