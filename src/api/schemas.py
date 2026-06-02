from pydantic import BaseModel, Field
from typing import Optional, Dict, List

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000,
                         example="I feel very anxious lately")

class NERResult(BaseModel):
    symptoms: List[str]
    triggers: List[str]
    duration: Optional[str]
    severity: str

class ChatResponse(BaseModel):
    original_message:    str
    detected_lang:       str
    language_name:       str
    translated_message:  str
    emotion:             str
    emotion_confidence:  float
    intent:              str
    is_crisis:           bool
    ner:                 Optional[NERResult]    # None if intent == general
    hypothetical_response: Optional[str]          # None if generation failed
    optimized_query:     Optional[str]          # None if intent == general
    response:            str