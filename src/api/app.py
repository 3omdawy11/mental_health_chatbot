from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.language_detector import LanguageDetector
from src.emotion_classifier import EmotionClassifier
from src.intent_classifier import IntentClassifier
from src.language_translator import Translator

from src.api.routes import chat, health

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading NLP models into memory...")
    app.state.lang_detector       = LanguageDetector()
    app.state.emotion_classifier  = EmotionClassifier()
    app.state.intent_classifier   = IntentClassifier()
    app.state.translator          = Translator()
    print("All models loaded ✓")

    yield

    print("Shutting down...")
    del app.state.lang_detector
    del app.state.emotion_classifier
    del app.state.intent_classifier
    del app.state.translator

app = FastAPI(
    title="Mental Health Chatbot API",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router,   prefix="/v1", tags=["Chat"])
app.include_router(health.router, prefix="/v1", tags=["System Health"])