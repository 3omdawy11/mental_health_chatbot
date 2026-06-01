import asyncio
from fastapi import APIRouter, Depends, HTTPException
from src.api.schemas import ChatRequest, ChatResponse
from src.api.dependencies import (
    get_language_detector, 
    get_emotion_classifier, 
    get_intent_classifier
)

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def process_chat_message(
    request: ChatRequest,
    lang_detector = Depends(get_language_detector),
    emotion_classifier = Depends(get_emotion_classifier),
    intent_classifier = Depends(get_intent_classifier)
):
    user_text = request.message.strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="Message content cannot be empty.")

    # 1. Language Detection (Fast CPU bound)
    lang_result = lang_detector.detect(user_text)
    
    # 2. Parallelizing Emotion and Intent tasks to maximize performance
    # Running in threads prevents blocking FastAPI's main event loop
    try:
        emotion_task = asyncio.to_thread(emotion_classifier.predict, user_text)
        intent_task = asyncio.to_thread(intent_classifier.predict, user_text)
        
        emotion_result, intent_result = await asyncio.gather(emotion_task, intent_task)
    except Exception as e:
        # Fail-safe fallbacks if a classifier throws an error
        emotion_result = {"emotion": "neutral"}
        intent_result = {"intent": "general", "is_crisis": False}

    detected_intent = intent_result.get("intent", "general")
    is_crisis_event = intent_result.get("is_crisis", False) or detected_intent in ["suicide", "self_harm", "crisis"]

    # 3. Crisis Guardrail Interception
    if is_crisis_event:
        return ChatResponse(
            original_message=user_text,
            detected_lang=lang_result.get("language", "unknown"),
            emotion=emotion_result.get("emotion", "neutral"),
            intent=detected_intent,
            response="I'm really sorry you're feeling this way. Please reach out to someone who can help right now. You can call or text the Suicide & Crisis Lifeline at 988.",
            is_crisis=True
        )

    # 4. RAG Engine Integration Placeholder
    # Once your RAG module is fully developed, you'll inject it here:
    # rag_output = await rag_engine.generate(user_text, emotion=emotion_result.get("emotion"))
    rag_output = f"Mock RAG Response: This is where the mental health context gets injected."

    return ChatResponse(
        original_message=user_text,
        detected_lang=lang_result.get("language", "unknown"),
        emotion=emotion_result.get("emotion", "neutral"),
        intent=detected_intent,
        response=rag_output,
        is_crisis=False
    )