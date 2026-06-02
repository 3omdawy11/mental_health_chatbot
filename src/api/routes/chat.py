import asyncio
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
    ner_extractor      = Depends(get_ner_extractor),
    query_optimizer    = Depends(get_query_optimizer),
    embedder           = Depends(get_embedder),
):
    user_text = request.message.strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    # ── Step 1: Language detection ─────────────────────────────────────
    lang_result   = lang_detector.detect(user_text)
    detected_lang = lang_result.get("language", "en")
    language_name = lang_result.get("language_name", "English")

    # ── Step 2: Translate to English ───────────────────────────────────
    if detected_lang != "en":
        translated = translator.to_english(user_text, detected_lang)
    else:
        translated = user_text

    # ── Step 3: Emotion + Intent in parallel ───────────────────────────
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
            ner=None,
            optimized_query=None,
            hypothetical_response=None,
            response=crisis_reply,
        )

    # ── Step 5: NER + Query optimization (mental health only) ──────────
    ner_result            = None
    optimized_query       = None
    hypothetical_response = None

    if "mental_health" in detected_intent:
        try:
            ner_task   = asyncio.to_thread(ner_extractor.extract,    translated)
            query_task = asyncio.to_thread(query_optimizer.optimize, translated)
            ner_data, query_data = await asyncio.gather(ner_task, query_task)

            ner_result = NERResult(
                symptoms=ner_data.get("symptoms", []),
                triggers=ner_data.get("triggers", []),
                duration=ner_data.get("duration"),
                severity=ner_data.get("severity", "medium"),
            )
            optimized_query = query_data.get("optimized", translated)

        except Exception as e:
            optimized_query = translated

        # ── Step 6: Generate hypothetical counsellor response ──────────
        # Used as the actual RAG query — bridges user language to KB language
        try:
            hypothetical_response = await asyncio.to_thread(
                embedder.generate_hypothetical,
                optimized_query
            )
        except Exception as e:
            hypothetical_response = None

    # ── Step 7: RAG placeholder ────────────────────────────────────────
    # When RAG is ready, replace these lines:
    # if "mental_health" in detected_intent:
    #     query_for_retrieval = hypothetical_response or optimized_query or translated
    #     rag_output = await rag_engine.generate(
    #         query=query_for_retrieval,
    #         emotion=emotion,
    #         ner=ner_result,
    #     )
    #     response = translator.to_lang(rag_output, detected_lang)
    # else:
    #     response = await llm_responder.generate(translated)
    #     response = translator.to_lang(response, detected_lang)
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
        ner=ner_result,
        optimized_query=optimized_query,
        hypothetical_response=hypothetical_response,
        response=response,
    )