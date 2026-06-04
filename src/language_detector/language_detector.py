"""
src/language_detector.py
────────────────────────
Drop-in replacement for the original LanguageDetector wrapper.

New behaviour (fully backward-compatible)
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

Public API (unchanged)
───────────────────────
    detector = LanguageDetector()
    result   = detector.detect("Hello")
    # → {"language": "en", "confidence": 0.87, "method": "tfidf_lr"}

    result   = detector.detect("hi")
    # → {"language": "en", "confidence": 0.43, "method": "tfidf_lr_priority_fallback"}
    #                                                  or  "langid_ensemble" if enabled

The "method" key is NEW and tells you which decision path was taken.
Existing callers that only read "language" and "confidence" are unaffected.
"""

from __future__ import annotations

import os
import pickle
import warnings
from pathlib import Path
from typing import Any

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Minimum confidence for the model's top prediction to be accepted as-is.
# Below this value the priority-fallback logic kicks in.
# Range 0–1.  Good starting point: 0.50–0.65.
CONFIDENCE_THRESHOLD: float = 0.55

# How many top candidates to consider when doing the priority fallback.
# E.g. 5 means "look at the 5 most-probable languages and pick the most
# globally-common one among them".
FALLBACK_TOP_K: int = 10

# ── Short-text ensemble toggle ───────────────────────────────────────────────
# Set to True  → use langid for texts shorter than SHORT_TEXT_THRESHOLD chars.
# Set to False → always use our TF-IDF/LR model (+ thresholding above).
# Can also be overridden at runtime via env-var: LANG_DETECTOR_ENSEMBLE=0 / 1
USE_SHORT_TEXT_ENSEMBLE: bool = True   # ← flip to True to enable

# Texts strictly shorter than this character count are routed to the ensemble
# detector when USE_SHORT_TEXT_ENSEMBLE is True.
SHORT_TEXT_THRESHOLD: int = 30

# ─────────────────────────────────────────────────────────────────────────────
# ❷  LANGUAGE PRIORITY LIST
#    When the model is uncertain, we prefer whichever of these languages
#    appears highest in the model's top-K candidates.
#    Order = approximate global speaker population / web prevalence.
#    Only languages your model actually knows matter — others are ignored.
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

# ─────────────────────────────────────────────────────────────────────────────
# ❸  Paths (mirrors the training script layout)
# ─────────────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent.parent

if "/kaggle" in str(Path.cwd()):
    _MODEL_DIR = Path("/kaggle/working/models/language_detection")
else:
    _MODEL_DIR = _ROOT / "models" / "language_detection"


# ─────────────────────────────────────────────────────────────────────────────
# ❹  Optional ensemble: lazy-import langid
# ─────────────────────────────────────────────────────────────────────────────

def _ensemble_enabled() -> bool:
    """Respect the module-level flag AND the env-var override."""
    env = os.environ.get("LANG_DETECTOR_ENSEMBLE")
    if env is not None:
        return env.strip() not in ("0", "false", "False", "no")
    return USE_SHORT_TEXT_ENSEMBLE


def _try_load_langid():
    """Return the langid module if available, else None (with a single warning)."""
    try:
        import langid  # pip install langid
        return langid
    except ImportError:
        warnings.warn(
            "langid is not installed — short-text ensemble is disabled. "
            "Run `pip install langid` to enable it.",
            RuntimeWarning,
            stacklevel=3,
        )
        return None


# ─────────────────────────────────────────────────────────────────────────────
# ❺  Core detector class
# ─────────────────────────────────────────────────────────────────────────────

class LanguageDetector:
    """
    Wrapper around the trained TF-IDF + Logistic Regression language detector.

    Parameters
    ----------
    model_dir : Path | str | None
        Directory containing vectorizer.pkl and model.pkl.
        Defaults to the project's models/language_detection folder.
    confidence_threshold : float
        Override CONFIDENCE_THRESHOLD for this instance.
    fallback_top_k : int
        Override FALLBACK_TOP_K for this instance.
    use_ensemble : bool | None
        Override USE_SHORT_TEXT_ENSEMBLE for this instance.
        None → use the module-level setting (+ env-var).
    short_text_threshold : int
        Override SHORT_TEXT_THRESHOLD for this instance.
    """

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


        # Resolve ensemble flag
        if use_ensemble is None:
            self._use_ensemble = _ensemble_enabled()
        else:
            self._use_ensemble = use_ensemble

        # Load sklearn artefacts
        with open(self.model_dir / "vectorizer.pkl", "rb") as f:
            self.vectorizer = pickle.load(f)
        with open(self.model_dir / "model.pkl", "rb") as f:
            self.model = pickle.load(f)

        # Build a fast set of classes for O(1) look-up
        self._known_classes: set[str] = set(self.model.classes_)

        # Build priority index: {lang: rank} for languages the model knows
        self._priority_index: dict[str, int] = {
            lang: rank
            for rank, lang in enumerate(LANGUAGE_PRIORITY)
            if lang in self._known_classes
        }

        # Lazy-load langid only when ensemble is requested
        self._langid = None
        if self._use_ensemble:
            self._langid = _try_load_langid()
            if self._langid is None:
                self._use_ensemble = False   # graceful degradation
        self._loaded = True

    # ── public API ────────────────────────────────────────────────────────────

    def detect(self, text: str) -> dict[str, Any]:
        """
        Detect the language of *text*.

        Returns
        -------
        dict with keys:
            language   : str   — ISO 639-1 code
            confidence : float — probability in [0, 1]
            method     : str   — decision path taken (informational)
        """
        text = text.strip()

        # ── Route very short texts to ensemble detector first ─────────────
        if self._use_ensemble and len(text) < self.short_text_threshold:
            return self._detect_with_ensemble(text)

        # ── Full TF-IDF/LR pipeline ───────────────────────────────────────
        return self._detect_with_model(text)

    def detect_batch(self, texts: "list[str]") -> "list[dict[str, Any]]":
        """Detect languages for a list of texts (vectorised for speed)."""
        return [self.detect(t) for t in texts]

    # ── internal helpers ──────────────────────────────────────────────────────

    def _detect_with_model(self, text: str) -> dict[str, Any]:
        """
        Run the TF-IDF + LR model with probability thresholding.

        Decision flow
        ─────────────
        1. Vectorise and call predict_proba().
        2. If top confidence ≥ threshold  →  return that prediction.
        3. Otherwise look at top-K candidates and return the one with the
           highest LANGUAGE_PRIORITY rank (lowest rank index = most popular).
        4. If none of the top-K appears in the priority list, fall back to
           the raw top prediction (best we can do).
        """
        X        = self.vectorizer.transform([text])
        proba    = self.model.predict_proba(X)[0]          # shape: (n_classes,)
        top_conf = float(proba.max())
        top_idx  = int(proba.argmax())
        top_lang = self.model.classes_[top_idx]
        all_scores = {lang: float(prob) for lang, prob in zip(self.model.classes_, proba)}
        # ── High confidence: accept as-is ──────────────────────────────────
        if top_conf >= self.confidence_threshold:
            return {
                "language":   top_lang,
                "confidence": top_conf,
                "method":     "tfidf_lr",
                "all_scores": all_scores,  # ← NEW: include full score dict for debugging/analysis
            }

        # ── Low confidence: priority fallback ──────────────────────────────
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
                "all_scores": all_scores,  # ← NEW: include full score dict for debugging/analysis
            }

        # ── Last resort: return the raw top prediction ─────────────────────
        return {
            "language":   top_lang,
            "confidence": top_conf,
            "method":     "tfidf_lr_raw_fallback",
            "all_scores": all_scores,  # ← NEW: include full score dict for debugging/analysis
        }

    def _detect_with_ensemble(self, text: str) -> dict[str, Any]:
        """
        For short texts: use langid as the primary signal.

        langid returns (lang_code, log_probability).
        - If the predicted language is one our model knows, trust it.
        - Otherwise fall through to our model's thresholded prediction.
        """
        lang_code, log_prob = self._langid.classify(text)

        if lang_code in self._known_classes:
            # Convert log-prob to a 0-1 confidence proxy via sigmoid-style clip
            # langid log-probs are typically large negative numbers;
            # we normalise to [0, 1] just for API consistency.
            confidence = float(np.clip(1.0 / (1.0 + np.exp(-log_prob * 0.05)), 0.0, 1.0))
            return {
                "language":   lang_code,
                "confidence": confidence,
                "method":     "langid_ensemble",
                "all_scores": None,  # ← NEW: include full score dict for debugging/analysis
            }

        # langid predicted a language our model doesn't know → fall back
        result = self._detect_with_model(text)
        result["method"] = "model_fallback_from_langid"
        return result
    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "not loaded"
        return f"LanguageDetector(model_dir='{self.model_dir}', status={status})"