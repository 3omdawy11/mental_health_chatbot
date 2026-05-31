"""
src/utils/query_optimizer.py
==============================
Query rewriting/optimization for better RAG retrieval.

Transforms informal user queries into terminology-rich search strings by:
  1. Asking Groq to rewrite (primary path)
  2. Expanding via a mental-health synonym dictionary (fallback)

Usage
-----
    from src.utils.query_optimizer import QueryOptimizer
    opt = QueryOptimizer()
    result = opt.optimize("i feel anxious at work")
    # {
    #   "original":  "i feel anxious at work",
    #   "optimized": "anxiety stress nervousness worry work workplace occupational stress",
    #   "method":    "groq" | "synonym_expansion"
    # }
"""

from __future__ import annotations

import logging
import os
import re
from typing import Optional

logger = logging.getLogger(__name__)

# ── Synonym / expansion dictionary ───────────────────────────────────────────
# Maps informal words → clinical + related terms for BM25/semantic enrichment
_EXPANSIONS: dict[str, list[str]] = {
    "anxious":      ["anxiety", "nervousness", "worry", "apprehension", "fear"],
    "anxiety":      ["anxious", "nervousness", "panic", "worry", "apprehension"],
    "depressed":    ["depression", "sadness", "low mood", "hopelessness", "melancholy"],
    "depression":   ["depressed", "low mood", "hopelessness", "anhedonia", "grief"],
    "stressed":     ["stress", "overwhelmed", "pressure", "burnout", "tension"],
    "stress":       ["stressed", "overwhelmed", "pressure", "burnout", "strain"],
    "panic":        ["panic attack", "anxiety", "fear", "terror", "dread"],
    "sad":          ["sadness", "grief", "low mood", "depression", "unhappy"],
    "lonely":       ["loneliness", "isolation", "social withdrawal", "disconnection"],
    "angry":        ["anger", "frustration", "irritability", "rage", "hostility"],
    "scared":       ["fear", "anxiety", "phobia", "terror", "apprehension"],
    "tired":        ["fatigue", "exhaustion", "burnout", "low energy", "lethargy"],
    "sleep":        ["insomnia", "sleep disorder", "fatigue", "rest", "sleep quality"],
    "insomnia":     ["sleep problems", "fatigue", "exhaustion", "sleep disorder"],
    "work":         ["workplace", "occupational stress", "job stress", "career", "professional"],
    "relationship": ["interpersonal", "partner", "communication", "conflict", "attachment"],
    "family":       ["family dynamics", "parental", "siblings", "home environment"],
    "trauma":       ["ptsd", "post-traumatic", "traumatic event", "emotional wound"],
    "grief":        ["loss", "bereavement", "mourning", "sadness", "depression"],
    "ocd":          ["obsessive compulsive", "intrusive thoughts", "compulsions", "rituals"],
    "ptsd":         ["trauma", "post-traumatic stress", "flashbacks", "hypervigilance"],
    "eating":       ["eating disorder", "nutrition", "body image", "anorexia", "bulimia"],
    "self-harm":    ["self-injury", "coping", "crisis", "emotional pain"],
    "suicidal":     ["crisis", "suicidal ideation", "self-harm", "hopelessness"],
    "therapy":      ["counselling", "psychotherapy", "CBT", "treatment", "mental health support"],
    "cope":         ["coping strategies", "resilience", "management", "techniques"],
    "concentration":["focus", "attention", "cognitive", "adhd", "distraction"],
    "motivation":   ["apathy", "anhedonia", "energy", "drive", "depression"],
}

# Filler words to strip before expansion
_FILLERS = re.compile(
    r"\b(i|me|my|we|the|a|an|is|am|are|was|were|been|be|"
    r"feel|feeling|felt|have|has|had|do|does|did|can|"
    r"really|very|so|just|like|kind of|sort of|bit|"
    r"about|with|for|at|in|on|of|to|it|that|this|there)\b",
    re.IGNORECASE,
)

_REWRITE_PROMPT = """You are a mental health search query optimizer.
Rewrite the user message into an enriched search query for a mental health knowledge base.

Rules:
- Replace informal language with clinical/psychological terms
- Add relevant synonyms and related concepts
- Include treatment/coping approaches if implied
- Remove filler words (I, feel, really, just, etc.)
- Output ONLY the rewritten query — no explanation, no punctuation at the end
- Keep it under 20 words

Examples:
Input: "i feel anxious at work"
Output: anxiety stress nervousness worry workplace occupational pressure coping strategies

Input: "cant sleep and feel hopeless"
Output: insomnia sleep disorder hopelessness depression fatigue low mood treatment

Input: "my relationship is falling apart"
Output: relationship conflict interpersonal difficulties communication breakdown attachment issues

Input: "{text}"
Output:"""


class QueryOptimizer:
    """
    Rewrites user queries for better mental health RAG retrieval.

    Parameters
    ----------
    api_key : Groq API key (defaults to GROQ_API_KEY env var).
    model   : Groq model (default: llama-3.1-8b-instant — fast, good enough).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "llama-3.1-8b-instant",
        timeout: float = 6.0,
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

    # ── LLM rewriting ─────────────────────────────────────────────────────────

    def _rewrite_via_llm(self, text: str) -> str:
        prompt = _REWRITE_PROMPT.replace("{text}", text)
        resp = self._get_client().chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=60,
            temperature=0.1,
        )
        return (resp.choices[0].message.content or "").strip()

    # ── Synonym expansion fallback ────────────────────────────────────────────

    def _expand_via_synonyms(self, text: str) -> str:
        # Remove filler words
        cleaned = _FILLERS.sub(" ", text.lower())
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        words   = cleaned.split()

        extra: list[str] = []
        for word in words:
            word_clean = re.sub(r"[^a-z\-]", "", word)
            if word_clean in _EXPANSIONS:
                extra.extend(_EXPANSIONS[word_clean])

        combined = words + extra
        # Deduplicate while preserving order
        seen, result = set(), []
        for w in combined:
            if w not in seen and w:
                seen.add(w); result.append(w)
        return " ".join(result)

    # ── Public API ────────────────────────────────────────────────────────────

    def optimize(self, text: str) -> dict:
        """
        Optimize a query for RAG retrieval.

        Returns
        -------
        {
            "original":  str,   # unchanged input
            "optimized": str,   # enriched query string
            "method":    str    # "groq" | "synonym_expansion"
        }
        """
        if not isinstance(text, str) or not text.strip():
            return {"original": text, "optimized": text or "", "method": "passthrough"}

        if self._api_key:
            try:
                optimized = self._rewrite_via_llm(text)
                if optimized:
                    return {"original": text, "optimized": optimized, "method": "groq"}
            except Exception as exc:
                logger.warning(f"QueryOptimizer LLM failed: {exc} — using synonym expansion")

        optimized = self._expand_via_synonyms(text)
        return {"original": text, "optimized": optimized, "method": "synonym_expansion"}

    def optimize_batch(self, texts: list[str]) -> list[dict]:
        return [self.optimize(t) for t in texts]

    def __repr__(self) -> str:
        mode = "LLM" if self._api_key else "synonym_expansion"
        return f"QueryOptimizer(model='{self._model}', mode={mode})"