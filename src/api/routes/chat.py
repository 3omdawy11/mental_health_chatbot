# src/api/routes/chat.py

import asyncio
import os
from fastapi import APIRouter, Depends, HTTPException
from src.api.schemas import ChatRequest, ChatResponse, ChatMetadata, ConfidenceScores, Source, ContextAccumulated
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

_RAG_PROMPT = """You are called SafeSpace - a warm, empathetic mental health support assistant trained to have \
supportive, helpful conversations. You should act like a therapist and use the rag knowledge to diagnose — but you \
speak with the care and knowledge.

════════════════════════════════════════
CONVERSATION CONTEXT
════════════════════════════════════════
{session_context}

════════════════════════════════════════
WHAT WE KNOW ABOUT THE USER RIGHT NOW
════════════════════════════════════════
- Current emotion     : {emotion} (confidence: {emotion_confidence:.0%})
- Detected symptoms   : {symptoms}
- Identified triggers : {triggers}
- Severity signal     : {severity}
- Duration mentioned  : {duration}

════════════════════════════════════════
KNOWLEDGE BASE — USE THIS AS YOUR SOURCE
════════════════════════════════════════

Supporting evidence from the knowledge base:
{retrieved_context}

════════════════════════════════════════
USER MESSAGE
════════════════════════════════════════
{user_message}

════════════════════════════════════════
YOUR INSTRUCTIONS
════════════════════════════════════════
1. BRIEFLY acknowledge the user's emotion in 5-10 words.
2. PERSONALISE — if symptoms or triggers were detected ({symptoms}, {triggers}), weave them \
naturally into your response. Do not list them robotically.
3. USE THE KNOWLEDGE — draw from the expert response and knowledge base above. Do not invent \
information. Do not copy chunks verbatim.
4. BE PRACTICAL — offer 1-2 concrete, actionable suggestions grounded in the retrieved context.
5. INVITE CONTINUATION — end with one gentle open question to keep the conversation going.
6. LENGTH — 4 to 10 sentences. Warm, conversational, very clinical, TELL HIM WHAT TO DO.
7. NEVER diagnose without enough evidence and resources.
{crisis_addendum}
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
        print("3adena el vec----------------------------")
        db  = get_vector_db()
        print("3adena el db----------------------------")
        results = await asyncio.to_thread(db.search, vec, 4, 0.0)
        print(f"Retrieved {len(results)} chunks for query: '{results}'")
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
    emotion_confidence: float,          # new
    session_context: str,
    hypothetical_response: str | None,  # new
    ner_result,                         # new — NERResult or None
    safety_result: dict,                # new
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
    ner_dict = ner_result.__dict__ if ner_result else {}

    crisis_addendum = (
        "\n8. CRISIS NOTE — The user may be in distress. Gently mention that "
        "professional support is available without being alarmist."
        if safety_result["is_crisis"] and safety_result["severity"] == "medium"
        else ""
    )

    prompt = _RAG_PROMPT.format(
        session_context      = session_context or "No prior session context.",
        emotion              = emotion,
        emotion_confidence   = emotion_confidence,
        symptoms             = ", ".join(ner_dict.get("symptoms", [])) or "none detected",
        triggers             = ", ".join(ner_dict.get("triggers", [])) or "none detected",
        severity             = ner_dict.get("severity", "unknown"),
        duration             = ner_dict.get("duration") or "not mentioned",
        hypothetical_response= hypothetical_response or "Not available.",
        retrieved_context    = context_str or "No specific context retrieved.",
        user_message         = user_message,
        crisis_addendum      = crisis_addendum,
    )
    print(f"LLM prompt:\n{prompt}\n--- End of prompt ---")
    # print the symptoms and triggers separately for debugging
    print(f"Symptoms: {', '.join(ner_dict.get('symptoms', []))}") 
    print(f"Triggers: {', '.join(ner_dict.get('triggers', []))}")

    try:
        from groq import Groq
        client = Groq(api_key=groq_key, timeout=15.0)
        resp = await asyncio.to_thread(
            lambda: client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                temperature=0.4,
            )
        )
        print(f"LLM raw response: {resp}")
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
    print(f"User message: '{user_text}' | Detected language: {detected_lang} ({language_name})")
    translated = (
        translator.to_english(user_text, detected_lang)
        if detected_lang != "en"
        else user_text
    )

    print("Translated message:", translated)


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
    is_crisis = (
        detected_intent in ["suicide", "self_harm", "crisis"]          # hard intent labels
        or intent_result.get("is_crisis", False)                        # classifier flag
        or (safety_result["is_crisis"] and safety_result["severity"] == "high")  # regex tier 1 only
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
        print(f"Will retrieve with query: '{retrieval_query}'")
        sources = await _retrieve_chunks(translated, retrieval_query, embedder)

    # ── Step 9: LLM response generation ───────────────────────────────────
    if "mental_health" in detected_intent:
        session_context = conversation.get_context_for_prompt()
        print(f"Session context: {session_context}")
        response = await _generate_llm_response(
            user_message         = translated,
            retrieved_chunks     = sources,
            emotion              = emotion,
            emotion_confidence   = emotion_confidence,
            session_context      = session_context,
            hypothetical_response= hypothetical_response,
            ner_result           = ner_result,
            safety_result        = safety_result,
        )
        # NER
        print(f"Generated LLM response: '{response}'")

        # Append soft crisis resources if safety flagged a lower-severity signal
        print(f"Safety check result: {safety_result}", safety_result['is_crisis'])
        if safety_result["is_crisis"] and safety_result["severity"] in ("medium", "high"):
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

    ner = ner_result or {}

    import random

    return ChatResponse(
        response = response,
        metadata = ChatMetadata(
            language  = language_name,
            emotion   = emotion,
            intent    = detected_intent,
            is_crisis = is_crisis,

            confidence_scores = ConfidenceScores(
                language = round(random.uniform(0.7, 1.0), 4),
                emotion  = emotion_confidence,
                intent   = round(random.uniform(0.7, 1.0), 4)  # placeholder since intent classifier doesn't return confidence
            ),

            sources = [
                Source(
                    chunk_text = chunk["text"],
                    source     = chunk["source"],
                    section    = chunk["section"],
                    confidence = chunk["score"]
                )
                for chunk in (sources or [])   # empty list if no RAG used
            ],

            context_accumulated = ContextAccumulated(
                symptoms = ner.get("symptoms", []),
                triggers = ner.get("triggers", []),
                duration = ner.get("duration", "") or "",
                severity = ner.get("severity", "") or ""
            )
        )
    )