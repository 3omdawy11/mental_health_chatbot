"""
Multi-signal crisis detection and safety resource generation.

Signals used:
  1. Keyword matching  — explicit crisis language
  2. Emotion intensity — fear/sadness confidence > threshold
  3. Message pattern   — rapid-fire short messages
  4. Intent signals    — implicit self-harm phrasing
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Optional


_HIGH_KEYWORDS = re.compile(
    r"\b(kill\s*(my)?self|suicide|suicidal|want\s+to\s+die|end\s+my\s+life|"
    r"end\s+it\s+all|take\s+my\s+(own\s+)?life|overdose|slit\s+my|"
    r"hang\s+myself|jump\s+off|shoot\s+myself|no\s+reason\s+to\s+live|"
    r"better\s+off\s+dead|don['']t\s+want\s+to\s+(be\s+here|exist|live))\b",
    re.IGNORECASE,
)

_MEDIUM_KEYWORDS = re.compile(
    r"\b(hurt(ing)?\s+myself|self[-\s]?harm|self[-\s]?hurt|cutting|burning\s+myself|"
    r"can['\u2019]t\s+go\s+on|can['\u2019]t\s+take\s+it|nothing\s+to\s+live\s+for|"
    r"give\s+up\s+on\s+(life|everything)|disappear\s+forever|"
    r"everyone\s+would\s+be\s+better\s+without\s+me|"
    r"feel\s+like\s+dying|wish\s+I\s+was\s+(dead|gone))\b",
    re.IGNORECASE,
)

_LOW_KEYWORDS = re.compile(
    r"\b(hopeless|worthless|no\s+point|can['']t\s+cope|"
    r"falling\s+apart|breaking\s+down|rock\s+bottom|"
    r"don['']t\s+see\s+a\s+way\s+out|losing\s+the\s+will)\b",
    re.IGNORECASE,
)

_CRISIS_EMOTION_THRESHOLD   = 0.80  
_CRISIS_EMOTIONS             = {"fear", "sadness"}

# Rapid-fire detection: N messages in T seconds
_RAPID_FIRE_COUNT   = 4
_RAPID_FIRE_WINDOW  = 60



_RESOURCES_BY_LANGUAGE: dict[str, list[dict]] = {
    "ar": [
        {"name": "IASP Crisis Centres (Arabic)",         "contact": "https://www.iasp.info/resources/Crisis_Centres/", "url": "https://www.iasp.info", "available": "varies"},
    ],
    "en": [
        {"name": "988 Suicide & Crisis Lifeline",        "contact": "Call or text 988",      "url": "https://988lifeline.org",       "available": "24/7"},
        {"name": "Crisis Text Line",                     "contact": "Text HOME to 741741",   "url": "https://crisistextline.org",    "available": "24/7"},
        {"name": "International Association for Suicide Prevention", "contact": "https://www.iasp.info/resources/Crisis_Centres/", "url": "https://www.iasp.info", "available": "varies"},
    ],
    "es": [
        {"name": "Teléfono de la Esperanza",             "contact": "717 003 717",            "url": "https://telefonodelaesperanza.org", "available": "24/7"},
        {"name": "Crisis Text Line (en español)",        "contact": "Text HOLA to 741741",   "url": "https://crisistextline.org",    "available": "24/7"},
    ],
    "fr": [
        {"name": "Numéro National Prévention Suicide",   "contact": "3114",                   "url": "https://3114.fr",               "available": "24/7"},
    ],
    "de": [
        {"name": "Telefonseelsorge",                     "contact": "0800 111 0 111",         "url": "https://www.telefonseelsorge.de","available": "24/7"},
    ],
    
}

_DEFAULT_RESOURCES = _RESOURCES_BY_LANGUAGE["en"]



_DEESCALATION = {
    "high": (
        "You're not alone, and what you're feeling right now is real. "
        "Please reach out to a crisis line immediately — trained counsellors are available 24/7 "
        "and are ready to listen without judgement."
    ),
    "medium": (
        "It sounds like you're carrying a very heavy weight right now. "
        "You deserve support, and there are people who want to help. "
        "Please consider reaching out to a counsellor or crisis line today."
    ),
    "low": (
        "I hear that things feel very difficult right now. "
        "Talking to a mental health professional can help you find a way through this."
    ),
}


#  (for rapid-fire detection) ────────────────────────
class _MessageTimer:
    def __init__(self) -> None:
        self._timestamps: list[float] = []

    def record(self) -> int:
        """Record a new message timestamp. Returns recent message count."""
        now = time.time()
        self._timestamps.append(now)
        # Keep only last 60 seconds
        self._timestamps = [t for t in self._timestamps if now - t <= _RAPID_FIRE_WINDOW]
        return len(self._timestamps)


class SafetyChecker:
    def __init__(self, emotion_threshold: float = _CRISIS_EMOTION_THRESHOLD) -> None:
        self._emo_threshold = emotion_threshold
        self._timer = _MessageTimer()

    def check(
        self,
        text:               str,
        emotion:            Optional[str]  = None,
        emotion_confidence: Optional[float] = None,
        language:           str            = "en",
    ) -> dict:
        signals:  list[str] = []
        severity: str       = "none"

        _SOFTENERS = re.compile(
            r"\b(a\s+little|sometimes|occasionally|bit|slightly|"
            r"how\s+(do|can|should)|manage|help\s+with)\b", re.IGNORECASE
        )

        # Keyword signals
        if _HIGH_KEYWORDS.search(text):
            signals.append("high-severity crisis keyword detected")
            severity = "high"
        elif _MEDIUM_KEYWORDS.search(text):
            signals.append("medium-severity self-harm keyword detected")
            if severity == "none":
                severity = "medium"
        elif _LOW_KEYWORDS.search(text) and not _SOFTENERS.search(text):
            signals.append("low-severity hopelessness keyword detected")
            if severity == "none":
                severity = "low"

        # Emotion intensity signal
        if (emotion in _CRISIS_EMOTIONS
                and emotion_confidence is not None
                and emotion_confidence >= self._emo_threshold):
            signals.append(
                f"high {emotion} confidence ({emotion_confidence:.2f} ≥ {self._emo_threshold})"
            )
            if severity == "none":
                severity = "low"
            elif severity == "low":
                severity = "medium"

        # Rapid-fire message pattern
        recent_count = self._timer.record()
        if recent_count >= _RAPID_FIRE_COUNT:
            signals.append(f"rapid-fire messaging ({recent_count} msgs in {_RAPID_FIRE_WINDOW}s)")
            if severity == "none":
                severity = "low"

        # Implicit self-harm intent phrases (no explicit keywords)
        implicit = re.search(
            r"\b(how\s+(many|much).*(pills|tablets|medication)|"
            r"what\s+does\s+it\s+feel\s+like\s+to\s+die|"
            r"how\s+to\s+(disappear|not\s+exist))\b",
            text, re.IGNORECASE
        )
        if implicit:
            signals.append("implicit self-harm intent phrase detected")
            severity = "high"

        is_crisis = severity in ("low", "medium", "high")

        resources = _RESOURCES_BY_LANGUAGE.get(language, _DEFAULT_RESOURCES)

        action_map = {
            "high":   "Provide crisis resources immediately. Do not continue standard RAG. Express empathy.",
            "medium": "Provide crisis resources. Continue with extra care and empathy. Recommend professional support.",
            "low":    "Continue with care. Mention professional support. Monitor subsequent messages.",
            "none":   "Continue standard pipeline.",
        }

        return {
            "is_crisis":          is_crisis,
            "severity":           severity,
            "signals":            signals,
            "resources":          resources if is_crisis else [],
            "deescalation_text":  _DEESCALATION.get(severity, "") if is_crisis else "",
            "recommended_action": action_map[severity],
        }

    def format_resources(self, resources: list[dict]) -> str:
        if not resources:
            return ""
        lines = ["\n\n**If you're in crisis, please reach out:**"]
        for r in resources:
            lines.append(f"• **{r['name']}**: {r['contact']} ({r['available']})")
        return "\n".join(lines)