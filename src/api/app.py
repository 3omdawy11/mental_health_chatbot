from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import your wrappers
from src.language_detector import LanguageDetector
from src.emotion_classifier import EmotionClassifier
from src.intent_classifier import IntentClassifier

# Import your routes
from src.api.routes import chat, health

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup: Load all models into memory once ──
    print("Loading NLP models into memory...")
    app.state.lang_detector = LanguageDetector()
    app.state.emotion_classifier = EmotionClassifier()
    app.state.intent_classifier = IntentClassifier()
    print("All models successfully loaded!")
    
    yield
    
    # ── Shutdown: Clean up resources if needed ──
    print("Shutting down and clearing model states...")
    del app.state.lang_detector
    del app.state.emotion_classifier
    del app.state.intent_classifier

app = FastAPI(
    title="Mental Health Chatbot API Component",
    version="1.0.0",
    lifespan=lifespan
)

# Core Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers cleanly
app.include_router(chat.router, prefix="/v1", tags=["Chat"])
app.include_router(health.router, prefix="/v1", tags=["System Health"])