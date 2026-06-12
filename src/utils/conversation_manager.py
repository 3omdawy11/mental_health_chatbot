"""
this file I made it to track multi-turn conversation history and accumulates clinical context
(symptoms, triggers, topics, suggestions) across a session.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Turn:
    role:      str 
    text:      str
    timestamp: float = field(default_factory=time.time)
    metadata:  dict  = field(default_factory=dict)   # emotion, intent, ner, etc.


class ConversationManager:
    def __init__(self, max_turns: int = 10) -> None:
        self.max_turns   = max_turns
        self._turns:     list[Turn] = []
        # Accumulated clinical context — grows throughout the session
        self._symptoms:  list[str] = []
        self._triggers:  list[str] = []
        self._topics:    list[str] = []
        self._suggestions: list[str] = []
        self._emotions:  list[str] = []
        self._turn_count = 0          # total including dropped turns

    def add_turn(
        self,
        role:       str,
        text:       str,
        ner:        Optional[dict] = None,
        emotion:    Optional[str]  = None,
        intent:     Optional[str]  = None,
        sources:    Optional[list] = None,
        suggestions: Optional[list[str]] = None,
    ) -> None:
        meta = {}
        if emotion:    meta["emotion"]  = emotion
        if intent:     meta["intent"]   = intent
        if sources:    meta["sources"]  = sources

        self._turns.append(Turn(role=role, text=text, metadata=meta))
        self._turn_count += 1

        if len(self._turns) > self.max_turns:
            self._turns = self._turns[-self.max_turns:]

        if role == "user":
            if ner:
                for sym in ner.get("symptoms", []):
                    if sym and sym not in self._symptoms:
                        self._symptoms.append(sym)
                for trig in ner.get("triggers", []):
                    if trig and trig not in self._triggers:
                        self._triggers.append(trig)

            if emotion and emotion not in ("unknown", "") and emotion not in self._emotions:
                self._emotions.append(emotion)

            if intent == "asking_mental_health_question":
                # Extract topic keywords from the text
                for kw in _extract_topic_keywords(text):
                    if kw not in self._topics:
                        self._topics.append(kw)

        if role == "assistant" and suggestions:
            for s in suggestions:
                if s not in self._suggestions:
                    self._suggestions.append(s)


    def get_context_for_prompt(self, max_chars: int = 600) -> str:
        parts = []
        if self._symptoms:
            parts.append(f"Symptoms mentioned: {', '.join(self._symptoms[:6])}")
        if self._triggers:
            parts.append(f"Triggers mentioned: {', '.join(self._triggers[:6])}")
        if self._emotions:
            parts.append(f"Emotions expressed: {', '.join(self._emotions[:4])}")
        if self._topics:
            parts.append(f"Topics discussed: {', '.join(self._topics[:6])}")

        if not parts:
            return ""

        context = "Previous session context — " + ". ".join(parts) + "."
        return context[:max_chars]

    def get_recent_history(self, n_turns: int = 4) -> list[dict]:
        recent = self._turns[-n_turns:] if len(self._turns) >= n_turns else self._turns
        return [{"role": t.role, "content": t.text} for t in recent]

    def get_session_summary(self) -> dict:
        narrative_parts = []
        if self._symptoms:
            narrative_parts.append(f"We discussed: {', '.join(self._symptoms)}")
        if self._triggers:
            narrative_parts.append(f"identified triggers including {', '.join(self._triggers)}")
        if self._suggestions:
            narrative_parts.append(f"and explored strategies such as {', '.join(self._suggestions[:4])}")
        narrative = ". ".join(narrative_parts) + "." if narrative_parts else "Session completed."

        return {
            "total_turns":         self._turn_count,
            "symptoms_discussed":  list(self._symptoms),
            "triggers_discussed":  list(self._triggers),
            "emotions_observed":   list(self._emotions),
            "topics_covered":      list(self._topics),
            "suggestions_made":    list(self._suggestions),
            "narrative":           narrative,
        }

    def get_accumulated_context(self) -> dict:
        return {
            "symptoms":    list(self._symptoms),
            "triggers":    list(self._triggers),
            "topics":      list(self._topics),
            "emotions":    list(self._emotions),
        }

    def mentioned_earlier(self, keyword: str) -> bool:
        kw = keyword.lower()
        return any(kw in t.text.lower() for t in self._turns)

    def reset(self) -> None:
        self.__init__(max_turns=self.max_turns)

    @property
    def turn_count(self) -> int:
        return self._turn_count

    @property
    def is_empty(self) -> bool:
        return self._turn_count == 0

    def __len__(self) -> int:
        return len(self._turns)

    def __repr__(self) -> str:
        return (f"ConversationManager(turns={self._turn_count}, "
                f"symptoms={self._symptoms}, triggers={self._triggers})")



_TOPIC_KEYWORDS = {
    "anxiety", "depression", "stress", "panic", "sleep", "insomnia",
    "trauma", "ptsd", "grief", "anger", "loneliness", "burnout",
    "self-harm", "suicide", "eating", "ocd", "phobia", "therapy",
    "cbt", "mindfulness", "medication", "relationship", "work",
}

def _extract_topic_keywords(text: str) -> list[str]:
    lower = text.lower()
    return [kw for kw in _TOPIC_KEYWORDS if kw in lower]