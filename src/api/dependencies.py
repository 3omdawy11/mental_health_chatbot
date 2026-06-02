from fastapi import Request

def get_language_detector(request: Request):
    return request.app.state.lang_detector

def get_emotion_classifier(request: Request):
    return request.app.state.emotion_classifier

def get_intent_classifier(request: Request):
    return request.app.state.intent_classifier

def get_translator(request: Request):
    return request.app.state.translator

def get_ner_extractor(request: Request):
    return request.app.state.ner_extractor

def get_query_optimizer(request: Request):
    return request.app.state.query_optimizer

def get_embedder(request: Request):
    return request.app.state.embedder