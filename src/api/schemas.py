from pydantic import BaseModel, Field
from typing import Optional

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000,
                         example="I feel very anxious lately")

class ChatResponse(BaseModel):
    original_message: str
    translated_message: str
    detected_lang: str
    emotion: str
    intent: str                        # mental_health | general
    response: str                      # placeholder until RAG is ready
    is_crisis: bool