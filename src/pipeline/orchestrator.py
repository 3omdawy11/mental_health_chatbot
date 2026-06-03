"""
src/pipeline/orchestrator.py
==============================
Main pipeline orchestrator for the mental health chatbot.

Pipeline:
    Language Detection
    → Emotion Classification
    → NER
    → Intent Classification
    → Safety Check
    → Query Optimization
    → Retrieval (Qdrant primary, Hybrid fallback if available)
    → LLM Response Generation
    → Conversation Context Update
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent.parent
_KB_PATH = _ROOT / "data" / "processed" / "knowledge_base_combined.json"


_RAG_PROMPT = """You are a compassionate mental health support assistant.
Use ONLY the provided context to answer. Be empathetic and supportive.
Keep your response to 3-5 sentences. Do not diagnose.

{session_context}

Context from knowledge base:
{retrieved_context}

User message: {user_message}

Tone guidance: {tone_guidance}

Response:"""


_TONE_MAP = {
    "sadness":  "Warm, gentle, validating. Acknowledge pain before offering advice.",
    "fear":     "Calm, grounding, reassuring. Reduce alarm while staying present.",
    "anger":    "Non-judgmental, patient. Validate the feeling without amplifying it.",
    "joy":      "Warm and encouraging. Match their positive energy gently.",
    "love":     "Supportive and warm. Honour the emotional openness.",
    "surprise": "Clear and steady. Provide orientation and stability.",
    "unknown":  "Warm, professional, and empathetic.",
}


_DIRECT_RESPONSES = {
    "greeting": [
        "Hello! I'm here to support you. How are you feeling today?",
        "Hi there! I'm a mental health support assistant. What's on your mind?",
        "Hello! I'm glad you reached out. How can I support you today?",
    ],
    "goodbye": "Take care of yourself. Remember, support is always available when you need it.",
    "gratitude": "I'm really glad I could help. Please don't hesitate to reach out whenever you need support.",
    "out_of_scope": (
        "I'm specifically here to support with mental health and emotional wellbeing topics. "
        "Is there anything on your mind emotionally or mentally that I can help with?"
    ),
}


_SUGGESTION_KEYWORDS = [
    "breathing", "mindfulness", "therapy", "cbt", "medication", "exercise",
    "sleep hygiene", "journaling", "grounding", "meditation", "counselling",
    "self-care", "support group", "helpline", "crisis line",
]


def _extract_suggestions(text: str) -> list[str]:
    lower = (text or "").lower()
    return [kw for kw in _SUGGESTION_KEYWORDS if kw in lower]


def _normalize_chunk_for_hybrid(chunk: dict) -> dict:
    meta = chunk.get("metadata", {}) or {}
    return {
        "id": chunk.get("chunk_id") or chunk.get("id") or "",
        "text": chunk.get("text", ""),
        "source": meta.get("source", chunk.get("source", "")),
        "source_type": meta.get("source_type", chunk.get("source_type", "")),
        "section": meta.get("section", chunk.get("section", "")),
        "tokens": meta.get("tokens", chunk.get("tokens", 0)),
        "context_query": meta.get("context_query", ""),
        "original_question": meta.get("original_question", ""),
    }


class Orchestrator:
    """
    End-to-end pipeline orchestrator.

    Parameters
    ----------
    groq_api_key  : Groq API key (defaults to GROQ_API_KEY env var)
    use_hyde      : enable HyDE embeddings for retrieval queries
    use_hybrid    : enable in-memory hybrid fallback if KB is locally available
    vector_db_url : Qdrant URL override
    """

    def __init__(
        self,
        groq_api_key: Optional[str] = None,
        use_hyde: bool = False,
        use_hybrid: bool = True,
        vector_db_url: Optional[str] = None,
    ) -> None:
        self._groq_key = groq_api_key or os.getenv("GROQ_API_KEY", "")
        self._use_hyde = use_hyde
        self._use_hybrid = use_hybrid
        self._vdb_url = vector_db_url

        self._lang_detector = None
        self._emotion_clf = None
        self._ner = None
        self._intent_clf = None
        self._embedder = None
        self._vector_db = None
        self._query_optimizer = None
        self._safety = None

        self._hybrid_search = None
        self._hybrid_chunks = None

        from src.utils.conversation_manager import ConversationManager
        self.conversation = ConversationManager(max_turns=10)

        self._greeting_idx = 0

    # ── Lazy loaders ────────────────────────────────────────────────────────

    def _get_lang_detector(self):
        if self._lang_detector is None:
            from src.language_detector import LanguageDetector
            self._lang_detector = LanguageDetector()
        return self._lang_detector

    def _get_emotion_clf(self):
        if self._emotion_clf is None:
            from src.emotion_classifier import EmotionClassifier
            self._emotion_clf = EmotionClassifier()
        return self._emotion_clf

    def _get_ner(self):
        if self._ner is None:
            from src.ner_extractor import NERExtractor
            self._ner = NERExtractor(api_key=self._groq_key)
        return self._ner

    def _get_intent_clf(self):
        if self._intent_clf is None:
            from src.intent_classifier import IntentClassifier
            self._intent_clf = IntentClassifier(api_key=self._groq_key)
        return self._intent_clf

    def _get_embedder(self):
        if self._embedder is None:
            from src.utils.embedder import Embedder
            self._embedder = Embedder(groq_api_key=self._groq_key)
        return self._embedder

    def _get_vector_db(self):
        if self._vector_db is None:
            from src.utils.vector_db import VectorDBManager
            self._vector_db = VectorDBManager(url=self._vdb_url)
        return self._vector_db

    def _get_query_optimizer(self):
        if self._query_optimizer is None:
            from src.utils.query_optimizer import QueryOptimizer
            self._query_optimizer = QueryOptimizer(api_key=self._groq_key)
        return self._query_optimizer

    def _get_safety(self):
        if self._safety is None:
            from src.pipeline.safety_checks import SafetyChecker
            self._safety = SafetyChecker()
        return self._safety

    def _ensure_hybrid_ready(self):
        if not self._use_hybrid:
            return None
        if self._hybrid_search is not None:
            return self._hybrid_search
        if not _KB_PATH.exists():
            logger.warning(f"Hybrid fallback unavailable: KB file not found at {_KB_PATH}")
            return None

        try:
            from src.utils.hybrid_search import HybridSearch

            with open(_KB_PATH, encoding="utf-8") as f:
                raw_chunks = json.load(f)

            self._hybrid_chunks = [_normalize_chunk_for_hybrid(c) for c in raw_chunks if c.get("text")]
            if not self._hybrid_chunks:
                logger.warning("Hybrid fallback unavailable: no valid chunks in KB")
                return None

            self._hybrid_search = HybridSearch(self._hybrid_chunks, alpha=0.5)
            self._hybrid_search.build_corpus_vectors(self._get_embedder())
            logger.info(f"Hybrid fallback ready with {len(self._hybrid_chunks)} chunks")
            return self._hybrid_search

        except Exception as exc:
            logger.warning(f"Failed to initialize hybrid fallback: {exc}")
            self._hybrid_search = None
            return None

    # ── Safe wrappers ────────────────────────────────────────────────────────

    def _safe_language_detect(self, text: str) -> dict:
        try:
            return self._get_lang_detector().detect(text)
        except Exception as exc:
            logger.warning(f"Language detection failed: {exc}")
            return {
                "language": "en",
                "language_name": "English",
                "confidence": 0.0,
                "all_scores": {},
            }

    def _safe_emotion_classify(self, text: str) -> dict:
        try:
            return self._get_emotion_clf().classify(text)
        except Exception as exc:
            logger.warning(f"Emotion classification failed: {exc}")
            return {"emotion": "unknown", "confidence": 0.0, "all_scores": {}}

    def _safe_ner_extract(self, text: str) -> dict:
        try:
            return self._get_ner().extract(text)
        except Exception as exc:
            logger.warning(f"NER extraction failed: {exc}")
            return {"symptoms": [], "triggers": [], "duration": None, "severity": "low"}

    def _safe_intent_classify(self, text: str) -> dict:
        """
        Supports your actual IntentClassifier API:
        - predict(text) -> {"intent": ..., "is_crisis": ...}
        Also tolerates alternate future versions exposing classify(text).
        """
        try:
            clf = self._get_intent_clf()

            if hasattr(clf, "predict"):
                result = clf.predict(text)
            elif hasattr(clf, "classify"):
                result = clf.classify(text)
            else:
                raise AttributeError("IntentClassifier has neither predict() nor classify()")

            return {
                "intent": result.get("intent", "asking_mental_health_question"),
                "confidence": float(result.get("confidence", 0.0)),
                "is_crisis": bool(result.get("is_crisis", False)),
            }

        except Exception as exc:
            logger.warning(f"Intent classification failed: {exc}")
            return {
                "intent": "asking_mental_health_question",
                "confidence": 0.0,
                "is_crisis": False,
            }

    # ── Response generation ─────────────────────────────────────────────────

    def _generate_response(
        self,
        user_message: str,
        retrieved_chunks: list[dict],
        emotion: str,
        session_context: str,
    ) -> str:
        if not self._groq_key:
            if retrieved_chunks:
                top = retrieved_chunks[0].get("text", "")[:240]
                return (
                    f"It sounds like you're going through a difficult moment. "
                    f"One relevant idea is: {top} "
                    f"If these feelings are getting harder to manage, reaching out to a mental health professional could help."
                )
            return (
                "It sounds like you're carrying a lot right now. "
                "If you'd like, you can tell me a bit more, and if things feel overwhelming, "
                "reaching out to a mental health professional could help."
            )

        context_str = "\n\n".join(
            f"[{i+1}] {c.get('text', '')}" for i, c in enumerate(retrieved_chunks[:4])
        )
        tone = _TONE_MAP.get(emotion, _TONE_MAP["unknown"])

        prompt = _RAG_PROMPT.format(
            session_context=session_context or "No prior session context.",
            retrieved_context=context_str or "No specific context retrieved.",
            user_message=user_message,
            tone_guidance=tone,
        )

        try:
            from groq import Groq
            client = Groq(api_key=self._groq_key, timeout=15.0)
            resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.4,
            )
            text = (resp.choices[0].message.content or "").strip()
            if text:
                return text
        except Exception as exc:
            logger.warning(f"LLM generation failed: {exc}")

        if retrieved_chunks:
            return (
                f"{retrieved_chunks[0].get('text', '')[:180]}. "
                f"I recommend speaking with a mental health professional for personalised guidance."
            )

        return "I'm here to support you. Please consider speaking with a mental health professional."

    # ── Retrieval ───────────────────────────────────────────────────────────

    def _retrieve(self, query: str, optimized_query: str) -> list[dict]:
        emb = self._get_embedder()
        retrieval_query = (optimized_query or "").strip() or query

        # Primary: Qdrant semantic search
        try:
            db = self._get_vector_db()
            q_vec = emb.embed_hyde(retrieval_query) if self._use_hyde else emb.embed_text(retrieval_query)
            results = db.search(q_vec, limit=4, score_threshold=0.0)
            if results:
                return results
        except Exception as exc:
            logger.warning(f"Qdrant search failed: {exc}")

        # Fallback: local hybrid search
        hs = self._ensure_hybrid_ready()
        if hs is not None:
            try:
                results = hs.search(
                    retrieval_query,
                    emb,
                    k=4,
                    use_hyde=self._use_hyde,
                )
                mapped = []
                for r in results:
                    ch = r["chunk"]
                    mapped.append({
                        "text": ch.get("text", ""),
                        "source": ch.get("source", ""),
                        "source_type": ch.get("source_type", ""),
                        "section": ch.get("section", ""),
                        "tokens": ch.get("tokens", 0),
                        "chunk_id": ch.get("id", ""),
                        "score": r.get("combined_score", 0.0),
                    })
                return mapped
            except Exception as exc:
                logger.warning(f"Hybrid fallback search failed: {exc}")

        return []

    # ── Main process ────────────────────────────────────────────────────────

    def process(self, user_message: str) -> dict:
        text = (user_message or "").strip()
        if not text:
            return self._format(
                "I didn't catch that — could you tell me more about how you're feeling?",
                {}
            )

        # 1. Language
        lang_result = self._safe_language_detect(text)
        language = lang_result.get("language", "en")
        lang_conf = lang_result.get("confidence", 0.0)

        # 2. Emotion
        emo_result = self._safe_emotion_classify(text)
        emotion = emo_result.get("emotion", "unknown")
        emo_conf = emo_result.get("confidence", 0.0)

        # 3. NER
        ner_result = self._safe_ner_extract(text)

        # 4. Intent
        intent_result = self._safe_intent_classify(text)
        intent = intent_result.get("intent", "asking_mental_health_question")
        intent_conf = intent_result.get("confidence", 0.0)

        # 5. Safety
        safety = self._get_safety().check(
            text,
            emotion=emotion,
            emotion_confidence=emo_conf,
            language=language,
        )

        response = ""
        sources: list[dict] = []

        # 6. Routing
        if safety["is_crisis"] and safety["severity"] == "high":
            deesc = safety["deescalation_text"]
            res_str = self._get_safety().format_resources(safety["resources"])
            response = f"{deesc}{res_str}"

        elif intent == "greeting":
            idx = self._greeting_idx % len(_DIRECT_RESPONSES["greeting"])
            response = _DIRECT_RESPONSES["greeting"][idx]
            self._greeting_idx += 1

        elif intent == "goodbye":
            summary = self.conversation.get_session_summary()
            response = f"{_DIRECT_RESPONSES['goodbye']} {summary['narrative']}"

        elif intent == "gratitude":
            response = _DIRECT_RESPONSES["gratitude"]

        elif intent == "out_of_scope":
            response = _DIRECT_RESPONSES["out_of_scope"]

        else:
            opt_result = self._get_query_optimizer().optimize(text)
            optimized_query = opt_result.get("optimized", text)
            sources = self._retrieve(text, optimized_query)
            session_ctx = self.conversation.get_context_for_prompt()
            response = self._generate_response(text, sources, emotion, session_ctx)

            if safety["is_crisis"]:
                response += self._get_safety().format_resources(safety["resources"])

        # 7. Store conversation state
        suggestions = _extract_suggestions(response)

        self.conversation.add_turn(
            role="user",
            text=text,
            ner=ner_result,
            emotion=emotion,
            intent=intent,
        )
        self.conversation.add_turn(
            role="assistant",
            text=response,
            sources=sources,
            suggestions=suggestions,
        )

        # 8. Metadata
        metadata = {
            "language": language,
            "emotion": emotion,
            "intent": intent,
            "confidence_scores": {
                "language": round(lang_conf, 4),
                "emotion": round(emo_conf, 4),
                "intent": round(intent_conf, 4),
            },
            "sources": sources[:3],
            "is_crisis": safety["is_crisis"],
            "crisis_severity": safety["severity"],
            "crisis_signals": safety["signals"],
            "ner": ner_result,
            "context_accumulated": self.conversation.get_accumulated_context(),
        }

        return self._format(response, metadata)

    @staticmethod
    def _format(response: str, metadata: dict) -> dict:
        return {"response": response, "metadata": metadata}

    def reset_session(self) -> None:
        self.conversation.reset()

    def __repr__(self) -> str:
        return (
            f"Orchestrator(hyde={self._use_hyde}, hybrid={self._use_hybrid}, "
            f"turns={self.conversation.turn_count})"
        )