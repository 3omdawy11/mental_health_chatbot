"""
src/modules/ner_extractor.py
==============================
Zero-shot Named Entity Recognition for mental health queries.

Extracts: symptoms, triggers, duration, severity
Uses Groq LLM with structured JSON output; falls back to regex when
no API key is set or the call fails.

Usage
-----
    from src.modules.ner_extractor import NERExtractor
    ner = NERExtractor()
    result = ner.extract("I've been feeling anxious about work for 3 weeks")
    # {
    #   "symptoms": ["anxiety"],
    #   "triggers": ["work"],
    #   "duration": "3 weeks",
    #   "severity": "medium"
    # }
"""

from __future__ import annotations
import json, logging, os, re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Mental health vocabulary for regex fallback ───────────────────────────────
_SYMPTOMS = [
    "anxiety","anxious","depression","depressed","stress","stressed","panic",
    "insomnia","fatigue","exhaustion","hopeless","worthless","lonely","loneliness",
    "grief","trauma","burnout","overwhelm","overwhelmed","worry","fear","anger",
    "sadness","sad","guilt","shame","irritable","mood swings","concentration",
    "self-harm","suicidal","eating disorder","ocd","phobia","ptsd",
]
_TRIGGERS = [
    "work","job","career","workplace","office","boss","colleague",
    "relationship","partner","spouse","boyfriend","girlfriend","marriage","divorce",
    "family","parents","mother","father","siblings","children","kids",
    "school","university","college","exams","studies","grades",
    "money","financial","debt","bills","health","illness","chronic",
    "social","friends","isolation","loneliness","loss","death","grief",
]
_DURATION_RE = re.compile(
    r"(for\s+)?(a\s+few\s+)?"
    r"(\d+\s+)?(day|week|month|year|hour|minute)s?"
    r"(\s+ago|\s+now)?|"
    r"since\s+(yesterday|last\s+\w+|this\s+\w+|\d+\s+\w+)|"
    r"(all\s+day|all\s+week|all\s+month|past\s+\w+|recently|lately|for\s+a\s+while)",
    re.IGNORECASE,
)
_HIGH_SEVERITY = re.compile(
    r"\b(unbearable|severe|extreme|constant|always|can't\s+(stop|sleep|function)|"
    r"suicid|self.harm|hopeless|worthless|can't\s+go\s+on)\b", re.IGNORECASE
)
_LOW_SEVERITY = re.compile(
    r"\b(little|slight|bit|mild|sometimes|occasionally|minor|manageable)\b",
    re.IGNORECASE
)

_NER_PROMPT = """You are a clinical NLP assistant. Extract mental health entities from the user message.

Return ONLY valid JSON, no explanation, no markdown, exactly this schema:
{
  "symptoms": ["list of mental health symptoms mentioned"],
  "triggers": ["list of life stressors or triggers mentioned"],
  "duration": "time period mentioned or null",
  "severity": "high | medium | low based on language intensity"
}

Rules:
- symptoms: psychological/emotional states (anxiety, depression, panic, insomnia, etc.)
- triggers: external causes (work, relationship, school, family, money, health, etc.)
- duration: exact phrase from text or null if not mentioned
- severity: high=intense/constant/overwhelming, medium=moderate/frequent, low=mild/occasional
- Use null for missing fields, empty list [] for no items found

User message: "{text}"
JSON:"""


class NERExtractor:
    """
    Zero-shot NER for mental health entities.

    Parameters
    ----------
    api_key : Groq API key (defaults to GROQ_API_KEY env var).
              If absent, uses fast regex fallback — no LLM call.
    model   : Groq model name.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "llama-3.3-70b-versatile",
        timeout: float = 8.0,
    ) -> None:
        self._api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self._model   = model
        self._timeout = timeout
        self._client  = None

    def _get_client(self):
        if self._client is None:
            from groq import Groq
            self._client = Groq(api_key=self._api_key, timeout=self._timeout)
        return self._client

    # ── LLM extraction ────────────────────────────────────────────────────────

    def _extract_via_llm(self, text: str) -> dict:
        prompt = _NER_PROMPT.replace("{text}", text)
        resp = self._get_client().chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.0,
        )
        raw = resp.choices[0].message.content or "{}"
        # Strip markdown fences if present
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
        data = json.loads(raw)
        return self._normalise(data)

    # ── Regex fallback ────────────────────────────────────────────────────────

    def _extract_via_regex(self, text: str) -> dict:
        lower = text.lower()
        symptoms = [s for s in _SYMPTOMS if s in lower]
        triggers = [t for t in _TRIGGERS if t in lower]
        dur_match = _DURATION_RE.search(text)
        duration  = dur_match.group(0).strip() if dur_match else None
        if _HIGH_SEVERITY.search(text):
            severity = "high"
        elif _LOW_SEVERITY.search(text):
            severity = "low"
        elif symptoms:
            severity = "medium"
        else:
            severity = "low"
        return {"symptoms": symptoms, "triggers": triggers,
                "duration": duration, "severity": severity}

    # ── Normalise / validate schema ───────────────────────────────────────────

    @staticmethod
    def _normalise(data: dict) -> dict:
        return {
            "symptoms": [s.lower() for s in (data.get("symptoms") or []) if s],
            "triggers": [t.lower() for t in (data.get("triggers") or []) if t],
            "duration": data.get("duration") or None,
            "severity": data.get("severity", "medium").lower()
                        if data.get("severity") in ("high","medium","low",
                                                     "High","Medium","Low")
                        else "medium",
        }

    # ── Public API ────────────────────────────────────────────────────────────

    def extract(self, text: str) -> dict:
        """
        Extract mental health entities from text.

        Returns
        -------
        {"symptoms": [...], "triggers": [...], "duration": str|None, "severity": str}
        """
        if not isinstance(text, str) or not text.strip():
            return {"symptoms": [], "triggers": [], "duration": None, "severity": "low"}

        if self._api_key:
            try:
                return self._extract_via_llm(text)
            except Exception as exc:
                logger.warning(f"NERExtractor LLM failed: {exc} — using regex fallback")

        return self._extract_via_regex(text)

    def extract_batch(self, texts: list[str]) -> list[dict]:
        return [self.extract(t) for t in texts]

    def __repr__(self) -> str:
        mode = "LLM" if self._api_key else "regex-fallback"
        return f"NERExtractor(model='{self._model}', mode={mode})"