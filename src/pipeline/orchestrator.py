"""
src/pipeline/orchestrator.py
==============================
Main pipeline orchestrator for the mental health chatbot.

Wires together all Phase 2-6 modules in a single `process()` call:

    Language Detection → Emotion Classification → NER → Intent Classification
    → Safety Check → Routing → [RAG: Query Rewrite → Hybrid Search → LLM Generation]
    → Response Formatting

Usage
-----
    from src.pipeline.orchestrator import Orchestrator
    orch = Orchestrator()
    result = orch.process("I've been feeling anxious about work lately")
    print(result["response"])
    print(result["metadata"])
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, TYPE_CHECKING

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent.parent

# ── Prompt templates ──────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────

class Orchestrator:
    """
    End-to-end pipeline orchestrator.

    All modules are lazy-loaded on first use — construction is fast.

    Parameters
    ----------
    groq_api_key  : Groq API key (defaults to GROQ_API_KEY env var)
    use_hyde      : enable HyDE embeddings for RAG queries
    use_hybrid    : use hybrid BM25+semantic search (else pure semantic)
    vector_db_url : Qdrant URL override
    """

    def __init__(
        self,
        groq_api_key:  Optional[str] = None,
        use_hyde:      bool = False,
        use_hybrid:    bool = True,
        vector_db_url: Optional[str] = None,
    ) -> None:
        self._groq_key  = groq_api_key or os.getenv("GROQ_API_KEY", "")
        self._use_hyde  = use_hyde
        self._use_hybrid = use_hybrid
        self._vdb_url   = vector_db_url

        # Module instances — all lazy
        self._lang_detector   = None
        self._emotion_clf     = None
        self._ner             = None
        self._intent_clf      = None
        self._embedder        = None
        self._vector_db       = None
        self._hybrid_search   = None
        self._query_optimizer = None
        self._safety          = None

        # One ConversationManager per Orchestrator instance = one session
        from src.utils.conversation_manager import ConversationManager
        self.conversation = ConversationManager(max_turns=10)

        self._greeting_idx = 0

    # ── Lazy module loaders ───────────────────────────────────────────────────

    def _get_lang_detector(self):
        if self._lang_detector is None:
            from src.modules.language_detector import LanguageDetector
            self._lang_detector = LanguageDetector()
        return self._lang_detector

    def _get_emotion_clf(self):
        if self._emotion_clf is None:
            from src.modules.emotion_classifier import EmotionClassifier
            self._emotion_clf = EmotionClassifier()
        return self._emotion_clf

    def _get_ner(self):
        if self._ner is None:
            from src.modules.ner_extractor import NERExtractor
            self._ner = NERExtractor(api_key=self._groq_key)
        return self._ner

    def _get_intent_clf(self):
        if self._intent_clf is None:
            from src.modules.intent_classifier import IntentClassifier
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

    def _get_hybrid_search(self, chunks: list[dict]):
        """HybridSearch requires the chunk list at construction time."""
        if self._hybrid_search is None and chunks:
            from src.utils.hybrid_search import HybridSearch
            self._hybrid_search = HybridSearch(chunks, alpha=0.5)
            self._hybrid_search.build_corpus_vectors(self._get_embedder())
        return self._hybrid_search

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

    # ── LLM generation ────────────────────────────────────────────────────────

    def _generate_response(
        self,
        user_message:       str,
        retrieved_chunks:   list[dict],
        emotion:            str,
        session_context:    str,
    ) -> str:
        """Call Groq to generate a response grounded in retrieved chunks."""
        if not self._groq_key:
            # No API key — synthesise a template response from top chunk
            if retrieved_chunks:
                return (
                    f"Based on what you've shared, here's some relevant information: "
                    f"{retrieved_chunks[0].get('text','')[:200]}. "
                    f"Please consider speaking with a mental health professional for personalised support."
                )
            return "I hear you. Please consider reaching out to a mental health professional for personalised support."

        context_str = "\n\n".join(
            f"[{i+1}] {c.get('text','')}" for i, c in enumerate(retrieved_chunks[:4])
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
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            logger.warning(f"LLM generation failed: {exc}")
            if retrieved_chunks:
                return (
                    f"{retrieved_chunks[0].get('text','')[:180]}. "
                    f"I recommend speaking with a mental health professional for personalised guidance."
                )
            return "I'm here to support you. Please consider speaking with a mental health professional."

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def _retrieve(self, query: str, optimized_query: str) -> list[dict]:
        """
        Retrieve relevant chunks.  Tries Qdrant semantic search first,
        falls back to in-memory hybrid search if Qdrant unavailable.
        """
        emb = self._get_embedder()
        retrieval_query = optimized_query.strip() if optimized_query.strip() else query

        # Try Qdrant
        try:
            db      = self._get_vector_db()
            q_vec = emb.embed_hyde(retrieval_query) if self._use_hyde else emb.embed_text(retrieval_query)
            results = db.search(q_vec, limit=4, score_threshold=0.0)
            return results or []
        except Exception as exc:
            logger.warning(f"Qdrant search failed: {exc} — trying hybrid fallback")
            

        # Hybrid in-memory fallback (needs pre-built index)
        hs = self._hybrid_search
        if hs is not None:
            results = hs.search(optimized_query, emb, k=4)
            return [{"text": r["chunk"]["text"],
                     "source": r["chunk"].get("source",""),
                     "source_type": r["chunk"].get("source_type",""),
                     "section": r["chunk"].get("section",""),
                     "tokens": r["chunk"].get("tokens",0),
                     "chunk_id": r["chunk"].get("id",""),
                     "score": r["combined_score"]} for r in results]

        return []

    # ── Main process method ───────────────────────────────────────────────────

    def process(self, user_message: str) -> dict:
        """
        Process one user message through the full pipeline.

        Returns
        -------
        {
            "response": str,
            "metadata": {
                "language": str,
                "emotion": str,
                "intent": str,
                "confidence_scores": {language, emotion, intent},
                "sources": [...],
                "is_crisis": bool,
                "crisis_severity": str,
                "context_accumulated": {symptoms, triggers, topics, emotions},
            }
        }
        """
        text = (user_message or "").strip()
        if not text:
            return self._format("I didn't catch that — could you tell me more about how you're feeling?", {})

        # ── Step 1: Language detection ────────────────────────────────────────
        lang_result = self._get_lang_detector().detect(text)
        language    = lang_result.get("language", "en")
        lang_conf   = lang_result.get("confidence", 0.0)

        # ── Step 2: Emotion classification ───────────────────────────────────
        emo_result  = self._get_emotion_clf().classify(text)
        emotion     = emo_result.get("emotion", "unknown")
        emo_conf    = emo_result.get("confidence", 0.0)

        # ── Step 3: NER ───────────────────────────────────────────────────────
        ner_result  = self._get_ner().extract(text)

        # ── Step 4: Intent classification ────────────────────────────────────
        intent_result = self._get_intent_clf().classify(text)
        intent        = intent_result.get("intent", "out_of_scope")
        intent_conf   = intent_result.get("confidence", 0.0)

        # ── Step 5: Safety check ──────────────────────────────────────────────
        safety = self._get_safety().check(
            text,
            emotion=emotion,
            emotion_confidence=emo_conf,
            language=language,
        )

        # ── Step 6: Routing ───────────────────────────────────────────────────
        sources: list[dict] = []
        response: str       = ""

        if safety["is_crisis"] and safety["severity"] == "high":
            # Crisis override — skip RAG, return safety response immediately
            deesc   = safety["deescalation_text"]
            res_str = self._get_safety().format_resources(safety["resources"])
            response = f"{deesc}{res_str}"

        elif intent == "greeting":
            idx      = self._greeting_idx % len(_DIRECT_RESPONSES["greeting"])
            response = _DIRECT_RESPONSES["greeting"][idx]
            self._greeting_idx += 1

        elif intent == "goodbye":
            summary  = self.conversation.get_session_summary()
            response = (
                f"{_DIRECT_RESPONSES['goodbye']} "
                f"{summary['narrative']}"
            )

        elif intent == "gratitude":
            response = _DIRECT_RESPONSES["gratitude"]

        elif intent == "out_of_scope":
            response = _DIRECT_RESPONSES["out_of_scope"]

        else:
            # asking_mental_health_question → full RAG
            opt_result      = self._get_query_optimizer().optimize(text)
            optimized_query = opt_result["optimized"]
            sources         = self._retrieve(text, optimized_query)
            session_ctx     = self.conversation.get_context_for_prompt()
            response        = self._generate_response(text, sources, emotion, session_ctx)

            # Append crisis resources for medium/low severity even in RAG path
            if safety["is_crisis"]:
                response += self._get_safety().format_resources(safety["resources"])

        # ── Step 7: Accumulate context ────────────────────────────────────────
        suggestions = _extract_suggestions(response)
        self.conversation.add_turn(
            role="user", text=text, ner=ner_result,
            emotion=emotion, intent=intent,
        )
        self.conversation.add_turn(
            role="assistant", text=response,
            sources=sources, suggestions=suggestions,
        )

        # ── Step 8: Build structured output ──────────────────────────────────
        metadata = {
            "language":    language,
            "emotion":     emotion,
            "intent":      intent,
            "confidence_scores": {
                "language": round(lang_conf,  4),
                "emotion":  round(emo_conf,   4),
                "intent":   round(intent_conf, 4),
            },
            "sources":           sources[:3],    # top-3 only
            "is_crisis":         safety["is_crisis"],
            "crisis_severity":   safety["severity"],
            "crisis_signals":    safety["signals"],
            "ner":               ner_result,
            "context_accumulated": self.conversation.get_accumulated_context(),
        }

        return self._format(response, metadata)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _format(response: str, metadata: dict) -> dict:
        return {"response": response, "metadata": metadata}

    def reset_session(self) -> None:
        """Start a fresh conversation session."""
        self.conversation.reset()

    def __repr__(self) -> str:
        return (
            f"Orchestrator(hyde={self._use_hyde}, hybrid={self._use_hybrid}, "
            f"turns={self.conversation.turn_count})"
        )


# ── Helper ────────────────────────────────────────────────────────────────────

_SUGGESTION_KEYWORDS = [
    "breathing", "mindfulness", "therapy", "cbt", "medication", "exercise",
    "sleep hygiene", "journaling", "grounding", "meditation", "counselling",
    "self-care", "support group", "helpline", "crisis line",
]

def _extract_suggestions(text: str) -> list[str]:
    lower = text.lower()
    return [kw for kw in _SUGGESTION_KEYWORDS if kw in lower]