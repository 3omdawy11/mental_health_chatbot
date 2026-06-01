from fastapi import Request

def get_language_detector(request: Request):
    return request.app.state.language_detector

def get_emotion_classifier(request: Request):
    return request.app.state.emotion_classifier

def get_intent_classifier(request: Request):
    return request.app.state.intent_classifier
