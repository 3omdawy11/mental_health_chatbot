# src/api/schemas.py

from pydantic import BaseModel, Field
from typing import Optional, List

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000,
                         example="I feel very anxious lately")

# ── Nested metadata models ────────────────────────────────────────────────────

class Source(BaseModel):
    chunk_text : str
    source     : str
    section    : str
    confidence : float

class ContextAccumulated(BaseModel):
    symptoms : List[str]
    triggers : List[str]
    duration : str
    severity : str

class ConfidenceScores(BaseModel):
    language : float
    emotion  : float
    intent   : float

class ChatMetadata(BaseModel):
    language            : str
    emotion             : str
    intent              : str
    confidence_scores   : ConfidenceScores
    sources             : List[Source]
    is_crisis           : bool
    context_accumulated : ContextAccumulated

# ── Main response ─────────────────────────────────────────────────────────────

class ChatResponse(BaseModel):
    response : str
    metadata : ChatMetadata