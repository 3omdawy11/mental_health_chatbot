import asyncio
from fastapi import APIRouter, Depends, HTTPException
from src.api.schemas import ChatRequest, ChatResponse
from src.api.dependencies import (
    get_language_detector,
    get_emotion_classifier,
    get_intent_classifier,
    get_translator,
)

router = APIRouter()

CRISIS_RESPONSE = (
    "I'm really sorry you're feeling this way. "
    "Please reach out to someone who can help right now. "
    "You can call or text the Suicide & Crisis Lifeline at 988. "
    "You are not alone."
)

@router.post("/chat", response_model=ChatResponse)
async def process_chat_message(
    request: ChatRequest,
    lang_detector      = Depends(get_language_detector),
    emotion_classifier = Depends(get_emotion_classifier),
    intent_classifier  = Depends(get_intent_classifier),
    translator         = Depends(get_translator),
):
    user_text = request.message.strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    # ── Step 1: Language detection ─────────────────────────────────────
    lang_result   = lang_detector.detect(user_text)
    detected_lang = lang_result.get("language", "en")
    language_name = lang_result.get("language_name", "English")

    # ── Step 2: Translate to English if needed ─────────────────────────
    translated = translator.to_english(user_text, detected_lang)

    # ── Step 3: Emotion + Intent in parallel on English text ───────────
    try:
        emotion_task = asyncio.to_thread(emotion_classifier.predict, translated)
        intent_task  = asyncio.to_thread(intent_classifier.predict,  translated)
        emotion_result, intent_result = await asyncio.gather(emotion_task, intent_task)
    except Exception as e:
        emotion_result = {"emotion": "neutral", "confidence": 0.0, "all_scores": {}}
        intent_result  = {"intent": "general", "is_crisis": False}

    emotion            = emotion_result.get("emotion", "neutral")
    emotion_confidence = emotion_result.get("confidence", 0.0)
    detected_intent    = intent_result.get("intent", "general")
    is_crisis          = (
        intent_result.get("is_crisis", False)
        or detected_intent in ["suicide", "self_harm", "crisis"]
    )

    # ── Step 4: Crisis guardrail ───────────────────────────────────────
    if is_crisis:
        crisis_reply = translator.to_lang(CRISIS_RESPONSE, detected_lang)
        return ChatResponse(
            original_message=user_text,
            detected_lang=detected_lang,
            language_name=language_name,
            translated_message=translated,
            emotion=emotion,
            emotion_confidence=emotion_confidence,
            intent=detected_intent,
            is_crisis=True,
            response=crisis_reply,
        )

    # ── Step 5: RAG placeholder ────────────────────────────────────────
    # When RAG is ready, replace these lines:
    # rag_output = await rag_engine.generate(translated, emotion=emotion)
    # response   = translator.to_lang(rag_output, detected_lang)
    response = "RAG not integrated yet."

    return ChatResponse(
        original_message=user_text,
        detected_lang=detected_lang,
        language_name=language_name,
        translated_message=translated,
        emotion=emotion,
        emotion_confidence=emotion_confidence,
        intent=detected_intent,
        is_crisis=False,
        response=response,
    )