# src/api/routes/chat.py

import asyncio
import os
from fastapi import APIRouter, Depends, HTTPException
from src.api.schemas import ChatRequest, ChatResponse, NERResult
from src.api.dependencies import (
    get_language_detector,
    get_emotion_classifier,
    get_intent_classifier,
    get_translator,
    get_ner_extractor,
    get_query_optimizer,
    get_embedder,
)
from src.pipeline.safety_checks import SafetyChecker
from src.utils.conversation_manager import ConversationManager
from src.utils.vector_db import VectorDBManager
from functools import lru_cache

router = APIRouter()

# ── Tone map (mirrors orchestrator) ───────────────────────────────────────────
_TONE_MAP = {
    "sadness":  "Warm, gentle, validating. Acknowledge pain before offering advice.",
    "fear":     "Calm, grounding, reassuring. Reduce alarm while staying present.",
    "anger":    "Non-judgmental, patient. Validate the feeling without amplifying it.",
    "joy":      "Warm and encouraging. Match their positive energy gently.",
    "love":     "Supportive and warm. Honour the emotional openness.",
    "surprise": "Clear and steady. Provide orientation and stability.",
    "unknown":  "Warm, professional, and empathetic.",
}

_RAG_PROMPT = """You are a compassionate mental health support assistant.
Use ONLY the provided context to answer. Be empathetic and supportive.
Keep your response to 3-5 sentences. Do not diagnose.

{session_context}

Context from knowledge base:
{retrieved_context}

User message: {user_message}

Tone guidance: {tone_guidance}

Response:"""

_SUGGESTION_KEYWORDS = [
    "breathing", "mindfulness", "therapy", "cbt", "medication", "exercise",
    "sleep hygiene", "journaling", "grounding", "meditation", "counselling",
    "self-care", "support group", "helpline", "crisis line",
]


def _extract_suggestions(text: str) -> list[str]:
    lower = (text or "").lower()
    return [kw for kw in _SUGGESTION_KEYWORDS if kw in lower]


# ── Singletons ─────────────────────────────────────────────────────────────────
# One shared instance per process — keeps conversation history and
# avoids re-loading heavy components on every request.

@lru_cache(maxsize=1)
def get_safety_checker() -> SafetyChecker:
    return SafetyChecker()


@lru_cache(maxsize=1)
def get_conversation_manager() -> ConversationManager:
    return ConversationManager(max_turns=10)


@lru_cache(maxsize=1)
def get_vector_db() -> VectorDBManager:
    return VectorDBManager(url=os.getenv("QDRANT_URL"))


# ── Retrieval helper ──────────────────────────────────────────────────────────

async def _retrieve_chunks(
    query: str,
    optimized_query: str,
    embedder,
) -> list[dict]:
    """
    Embed the (optimized) query and run a semantic search against Qdrant.
    Returns an empty list on any failure so the pipeline can degrade gracefully.
    """
    retrieval_query = (optimized_query or "").strip() or query
    try:
        vec = await asyncio.to_thread(embedder.embed_text, retrieval_query)
        db  = get_vector_db()
        results = await asyncio.to_thread(db.search, vec, 4, 0.0)
        return results or []
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning(f"Qdrant retrieval failed: {exc}")
        return []


# ── LLM response generation ───────────────────────────────────────────────────

async def _generate_llm_response(
    user_message: str,
    retrieved_chunks: list[dict],
    emotion: str,
    session_context: str,
) -> str:
    """
    Call Groq with the RAG prompt. Falls back to a safe static reply if the
    API key is absent or the call fails.
    """
    groq_key = os.getenv("GROQ_API_KEY", "")
    #NER???
    if not groq_key:
        if retrieved_chunks:
            top = retrieved_chunks[0].get("text", "")[:240]
            return (
                f"It sounds like you're going through a difficult moment. "
                f"One relevant idea is: {top} "
                f"If these feelings are getting harder to manage, "
                f"reaching out to a mental health professional could help."
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
        client = Groq(api_key=groq_key, timeout=15.0)
        resp = await asyncio.to_thread(
            lambda: client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.4,
            )
        )
        text = (resp.choices[0].message.content or "").strip()
        if text:
            return text
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning(f"LLM generation failed: {exc}")

    if retrieved_chunks:
        return (
            f"{retrieved_chunks[0].get('text', '')[:180]}. "
            f"I recommend speaking with a mental health professional for personalised guidance."
        )
    return "I'm here to support you. Please consider speaking with a mental health professional."


# ── Route ─────────────────────────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse)
async def process_chat_message(
    request: ChatRequest,
    lang_detector      = Depends(get_language_detector),
    emotion_classifier = Depends(get_emotion_classifier),
    intent_classifier  = Depends(get_intent_classifier),
    translator         = Depends(get_translator),
    ner_extractor      = Depends(get_ner_extractor),
    query_optimizer    = Depends(get_query_optimizer),
    embedder           = Depends(get_embedder),
):
    user_text = request.message.strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    # Singleton helpers (not injected to keep the signature clean)
    safety      = get_safety_checker()
    conversation = get_conversation_manager()

    # ── Step 1: Language detection ─────────────────────────────────────────
    lang_result   = lang_detector.detect(user_text)
    detected_lang = lang_result.get("language", "en")
    language_name = lang_result.get("language_name", "English")

    # ── Step 2: Translate to English ───────────────────────────────────────
    translated = (
        translator.to_english(user_text, detected_lang)
        if detected_lang != "en"
        else user_text
    )

    # ── Step 3: Emotion + Intent in parallel ───────────────────────────────
    try:
        emotion_result, intent_result = await asyncio.gather(
            asyncio.to_thread(emotion_classifier.predict, translated),
            asyncio.to_thread(intent_classifier.predict,  translated),
        )
    except Exception:
        emotion_result = {"emotion": "neutral", "confidence": 0.0, "all_scores": {}}
        intent_result  = {"intent": "general", "is_crisis": False}

    emotion            = emotion_result.get("emotion", "neutral")
    emotion_confidence = emotion_result.get("confidence", 0.0)
    detected_intent    = intent_result.get("intent", "general")
    is_crisis_intent   = (
        intent_result.get("is_crisis", False)
        or detected_intent in ["suicide", "self_harm", "crisis"]
    )

    # ── Step 4: Safety check (enriches crisis detection) ──────────────────
    safety_result = safety.check(
        translated,
        emotion=emotion,
        emotion_confidence=emotion_confidence,
        language=detected_lang,
    )
    is_crisis = is_crisis_intent or (
        safety_result["is_crisis"] and safety_result["severity"] == "high"
    )

    # ── Step 5: Crisis guardrail ───────────────────────────────────────────
    if is_crisis:
        deesc    = safety_result.get("deescalation_text", "")
        res_str  = safety.format_resources(safety_result.get("resources", []))
        crisis_reply_en = f"{deesc}{res_str}".strip() or (
            "I'm really sorry you're feeling this way. "
            "Please reach out to someone who can help right now. "
            "You can call or text the Suicide & Crisis Lifeline at 988. "
            "You are not alone."
        )
        crisis_reply = translator.to_lang(crisis_reply_en, detected_lang)

        conversation.add_turn("user",      translated, emotion=emotion, intent=detected_intent)
        conversation.add_turn("assistant", crisis_reply)

        return ChatResponse(
            original_message=user_text,
            detected_lang=detected_lang,
            language_name=language_name,
            translated_message=translated,
            emotion=emotion,
            emotion_confidence=emotion_confidence,
            intent=detected_intent,
            is_crisis=True,
            ner=None,
            optimized_query=None,
            hypothetical_response=None,
            response=crisis_reply,
        )

    # ── Step 6: NER + Query optimisation (mental-health intents only) ──────
    ner_result            = None
    optimized_query       = None
    hypothetical_response = None

    if "mental_health" in detected_intent:
        try:
            ner_data, query_data = await asyncio.gather(
                asyncio.to_thread(ner_extractor.extract,    translated),
                asyncio.to_thread(query_optimizer.optimize, translated),
            )
            ner_result = NERResult(
                symptoms=ner_data.get("symptoms", []),
                triggers=ner_data.get("triggers", []),
                duration=ner_data.get("duration"),
                severity=ner_data.get("severity", "medium"),
            )
            optimized_query = query_data.get("optimized", translated)
        except Exception:
            optimized_query = translated

        # ── Step 7: HyDE — hypothetical counsellor response ───────────────
        # Used as the semantic query to the vector DB so retrieval bridges
        # colloquial user language to clinical KB language.
        try:
            hypothetical_response = await asyncio.to_thread(
                embedder.generate_hypothetical,
                optimized_query,
            )
        except Exception:
            hypothetical_response = None

    # ── Step 8: RAG retrieval ──────────────────────────────────────────────
    sources: list[dict] = []
    if "mental_health" in detected_intent:
        # Prefer the hypothetical response as the retrieval query when available
        retrieval_query = hypothetical_response or optimized_query or translated
        sources = await _retrieve_chunks(translated, retrieval_query, embedder)

    # ── Step 9: LLM response generation ───────────────────────────────────
    if "mental_health" in detected_intent:
        session_context = conversation.get_context_for_prompt()
        response = await _generate_llm_response(
            user_message=translated,
            retrieved_chunks=sources,
            emotion=emotion,
            session_context=session_context,
        )
        #NER?????

        # Append soft crisis resources if safety flagged a lower-severity signal
        if safety_result["is_crisis"]:
            response += safety.format_resources(safety_result.get("resources", []))

    elif detected_intent == "greeting":
        response = "Hello! I'm here to support you. How are you feeling today?"

    elif detected_intent == "goodbye":
        summary  = conversation.get_session_summary()
        response = f"Take care of yourself. Remember, SafeSpace is always there for you :)\n {summary['narrative']}"

    elif detected_intent == "gratitude":
        response = "I'm really glad I could help. Please don't hesitate to reach out whenever you need support. Thanks to god."

    else:  # out_of_scope, general, etc.
        response = (
            "I'm specifically here to support with mental health and emotional wellbeing topics. "
            "Is there anything on your mind emotionally or mentally that I can help with? "
        )

    # ── Step 10: Update conversation state ────────────────────────────────
    suggestions = _extract_suggestions(response)

    conversation.add_turn(
        role="user",
        text=translated,
        ner=ner_result.__dict__ if ner_result else None,
        emotion=emotion,
        intent=detected_intent,
    )
    conversation.add_turn(
        role="assistant",
        text=response,
        sources=sources,
        suggestions=suggestions,
    )

    # ── Step 11: Translate response back if needed ────────────────────────
    if detected_lang != "en":
        response = translator.to_lang(response, detected_lang)

    return ChatResponse(
        original_message=user_text,
        detected_lang=detected_lang,
        language_name=language_name,
        translated_message=translated,
        emotion=emotion,
        emotion_confidence=emotion_confidence,
        intent=detected_intent,
        is_crisis=False,
        ner=ner_result,
        optimized_query=optimized_query,
        hypothetical_response=hypothetical_response,
        response=response,
    )