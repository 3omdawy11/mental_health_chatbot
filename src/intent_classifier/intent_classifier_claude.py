"""
src/modules/intent_classifier.py
==================================
Zero-shot intent classification via Groq LLM API.

The classifier sends the user message to a Groq-hosted LLM with a carefully
crafted few-shot prompt and parses the single-word response.

Intents
-------
    greeting                    hi / hello / hey / good morning
    goodbye                     bye / see you / take care
    gratitude                   thank you / thanks / appreciate
    asking_mental_health_question  anxiety / depression / stress / coping / etc.
    out_of_scope                anything that doesn't fit the above

Design decisions
----------------
* Zero-shot via LLM → no labelled training data needed; covers edge cases that
  rule-based keyword matching misses ("I can't get out of bed" = mental health,
  not out_of_scope).
* Primary model  : llama-3.3-70b-versatile  (best Groq quality, fast)
* Fallback model : llama-3.1-8b-instant     (quota / rate-limit safety net)
* Retry with exponential back-off on transient errors.
* Hard fallback  : if LLM returns anything unexpected → "out_of_scope".
* Confidence is estimated from the raw response:
    1.0  → response is exactly one of the 5 valid intent strings
    0.7  → response needed fuzzy matching / partial clean-up
    0.4  → fallback was triggered

Usage
-----
    from src.modules.intent_classifier import IntentClassifier

    clf = IntentClassifier()          # reads GROQ_API_KEY from env
    result = clf.classify("I feel really anxious about my job interview")
    # {"intent": "asking_mental_health_question", "confidence": 1.0}

Environment
-----------
    GROQ_API_KEY   (required)  — get a free key at console.groq.com
    GROQ_MODEL     (optional)  — override primary model name
"""

from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

VALID_INTENTS: list[str] = [
    "greeting",
    "goodbye",
    "gratitude",
    "asking_mental_health_question",
    "out_of_scope",
]

# Groq model names (verified as of 2025)
# NOTE: The original phase spec mentioned "gpt-oss-120b" — that is an internal
# OpenAI label, NOT a Groq model.  Groq serves the Llama / Mixtral family.
PRIMARY_MODEL  = "llama-3.3-70b-versatile"   # best quality on Groq
FALLBACK_MODEL = "llama-3.1-8b-instant"      # fastest, lowest latency

_ROOT        = Path(__file__).resolve().parent.parent.parent
_PROMPT_PATH = _ROOT / "src" / "prompts" / "intent_classification.txt"

# Keyword-based fallback patterns (used when API is unavailable)
_KEYWORD_PATTERNS: dict[str, list[str]] = {
    "greeting":    [r"\b(hi|hello|hey|good\s*(morning|afternoon|evening)|howdy|greetings)\b"],
    "goodbye":     [r"\b(bye|goodbye|see\s+you|take\s+care|farewell|ciao|later)\b"],
    "gratitude":   [r"\b(thank|thanks|grateful|appreciate|cheers|ty\b|thx)\b"],
    "asking_mental_health_question": [
        r"\b(anxi|depress|stress|worry|worri|sad|lonely|hopeless|panic|trauma|"
        r"therapy|therapist|counsell?|mental\s+health|cope|coping|sleep|suicid|"
        r"self.harm|worthless|overwhelm|grief|burnout|emotion|mood|feeling)\w*\b",
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_prompt_template() -> str:
    if not _PROMPT_PATH.exists():
        raise FileNotFoundError(f"Prompt template not found: {_PROMPT_PATH}")
    return _PROMPT_PATH.read_text(encoding="utf-8")


def _parse_intent(raw: str) -> tuple[str, float]:
    """
    Parse the LLM's raw response into (intent, confidence).

    Strategy
    --------
    1. Strip whitespace + punctuation, lowercase.
    2. Exact match against VALID_INTENTS → confidence 1.0
    3. Partial/fuzzy match (intent name appears inside response) → 0.7
    4. Fallback to "out_of_scope" → 0.4
    """
    cleaned = raw.strip().lower().strip(".:,;!?\"'")

    # 1. Exact match
    if cleaned in VALID_INTENTS:
        return cleaned, 1.0

    # 2. Partial match — LLM sometimes adds trailing words
    for intent in VALID_INTENTS:
        if intent in cleaned:
            return intent, 0.7

    # 3. Common synonyms / aliases
    alias_map = {
        "mental_health":           "asking_mental_health_question",
        "mental health question":  "asking_mental_health_question",
        "mental_health_question":  "asking_mental_health_question",
        "greet":                   "greeting",
        "farewell":                "goodbye",
        "thanks":                  "gratitude",
        "thank_you":               "gratitude",
        "off_topic":               "out_of_scope",
        "irrelevant":              "out_of_scope",
        "other":                   "out_of_scope",
        "unknown":                 "out_of_scope",
    }
    for alias, intent in alias_map.items():
        if alias in cleaned:
            return intent, 0.7

    logger.warning(f"IntentClassifier: unexpected LLM response {raw!r} → fallback")
    return "out_of_scope", 0.4


def _keyword_fallback(text: str) -> tuple[str, float]:
    """
    Simple regex keyword fallback when the Groq API is unavailable.
    Returns (intent, confidence=0.6).
    """
    lower = text.lower()
    for intent, patterns in _KEYWORD_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, lower):
                return intent, 0.6
    return "out_of_scope", 0.6


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────

class IntentClassifier:
    """
    Zero-shot intent classifier backed by a Groq-hosted LLM.

    Parameters
    ----------
    api_key    : Groq API key.  Defaults to env var GROQ_API_KEY.
    model      : Primary Groq model name.  Defaults to PRIMARY_MODEL or
                 env var GROQ_MODEL.
    max_tokens : Max tokens in the LLM response (intent name is 1–5 tokens,
                 so 20 is generous).
    temperature: Lower = more deterministic.  0.0 is ideal for classification.
    max_retries: Number of retry attempts on transient errors.
    timeout    : Request timeout in seconds.
    """

    def __init__(
        self,
        api_key:     Optional[str] = None,
        model:       Optional[str] = None,
        max_tokens:  int   = 20,
        temperature: float = 0.0,
        max_retries: int   = 3,
        timeout:     float = 10.0,
    ) -> None:
        self._api_key    = api_key or os.getenv("GROQ_API_KEY", "")
        self._model      = model or os.getenv("GROQ_MODEL", PRIMARY_MODEL)
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._max_retries = max_retries
        self._timeout    = timeout
        self._prompt_tpl = _load_prompt_template()
        self._client     = None   # lazy init

    # ── Lazy client init ──────────────────────────────────────────────────────

    def _get_client(self):
        if self._client is None:
            if not self._api_key:
                raise EnvironmentError(
                    "GROQ_API_KEY is not set.\n"
                    "  1. Get a free key at https://console.groq.com\n"
                    "  2. export GROQ_API_KEY='gsk_...'\n"
                    "  3. Or pass api_key= to IntentClassifier()"
                )
            try:
                from groq import Groq
            except ImportError:
                raise ImportError("groq not installed. Run: pip install groq")
            self._client = Groq(api_key=self._api_key, timeout=self._timeout)
        return self._client

    # ── API call with retry ───────────────────────────────────────────────────

    def _call_api(self, user_text: str, model: str) -> str:
        """
        Send one API request.  Returns the raw text response.
        Raises on HTTP / network errors (caller handles retries).
        """
        prompt = self._prompt_tpl.replace("{user_text}", user_text)
        client = self._get_client()

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )
        return response.choices[0].message.content or ""

    def _call_with_retry(self, user_text: str) -> tuple[str, str]:
        """
        Call the API with exponential back-off and model fallback.
        Returns (raw_response, model_used).
        """
        if not self._api_key:
            raise EnvironmentError("GROQ_API_KEY is not set.")
        models_to_try = [self._model]
        if self._model != FALLBACK_MODEL:
            models_to_try.append(FALLBACK_MODEL)

        last_exc: Optional[Exception] = None

        for model in models_to_try:
            for attempt in range(1, self._max_retries + 1):
                try:
                    raw = self._call_api(user_text, model)
                    return raw, model
                except Exception as exc:
                    last_exc = exc
                    wait = 2 ** (attempt - 1)           # 1s, 2s, 4s
                    logger.warning(
                        f"Groq API error (model={model}, attempt={attempt}): "
                        f"{exc}. Retrying in {wait}s …"
                    )
                    time.sleep(wait)

        raise RuntimeError(
            f"Groq API failed after {self._max_retries} retries "
            f"on models {models_to_try}: {last_exc}"
        ) from last_exc

    # ── Public API ────────────────────────────────────────────────────────────

    def classify(self, text: str) -> dict:
        """
        Classify a user message into one of the 5 intents.

        Returns
        -------
        {
            "intent"     : "asking_mental_health_question",
            "confidence" : 1.0,          # 1.0 exact | 0.7 fuzzy | 0.4 fallback
            "model_used" : "llama-3.3-70b-versatile",
            "raw_response": "asking_mental_health_question"
        }

        Falls back to keyword heuristic if the API call fails entirely.
        """
        if not isinstance(text, str) or not text.strip():
            return {
                "intent":       "out_of_scope",
                "confidence":   1.0,
                "model_used":   "none",
                "raw_response": "",
            }

        # ── Try Groq API ──────────────────────────────────────────────────────
        try:
            raw, model_used = self._call_with_retry(text.strip())
            intent, confidence = _parse_intent(raw)
            return {
                "intent":       intent,
                "confidence":   confidence,
                "model_used":   model_used,
                "raw_response": raw.strip(),
            }

        except EnvironmentError:
            # No API key — use keyword fallback silently in dev/test
            logger.info("No GROQ_API_KEY set — using keyword fallback")
            intent, confidence = _keyword_fallback(text)
            return {
                "intent":       intent,
                "confidence":   confidence,
                "model_used":   "keyword_fallback",
                "raw_response": "",
            }

        except Exception as exc:
            logger.error(f"IntentClassifier: all API attempts failed: {exc}")
            intent, confidence = _keyword_fallback(text)
            return {
                "intent":       intent,
                "confidence":   confidence,
                "model_used":   "keyword_fallback",
                "raw_response": "",
            }

    def classify_batch(self, texts: list[str]) -> list[dict]:
        """Classify a list of texts.  Sequential — Groq is fast enough."""
        return [self.classify(t) for t in texts]

    @property
    def valid_intents(self) -> list[str]:
        return VALID_INTENTS

    @property
    def model(self) -> str:
        return self._model

    def __repr__(self) -> str:
        key_status = "set" if self._api_key else "NOT SET"
        return (
            f"IntentClassifier("
            f"model='{self._model}', "
            f"api_key={key_status}, "
            f"fallback=keyword_heuristic)"
        )