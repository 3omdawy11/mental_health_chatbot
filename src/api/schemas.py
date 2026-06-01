from pydantic import BaseModel, Field
from typing import Optional, Dict

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000,
                         example="I feel very anxious lately")

class ChatResponse(BaseModel):
    original_message:   str
    detected_lang:      str
    language_name:      str
    translated_message: str
    emotion:            str
    emotion_confidence: float
    intent:             str
    is_crisis:          bool
    response:           str