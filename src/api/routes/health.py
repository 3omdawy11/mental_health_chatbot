from fastapi import APIRouter, Request

router = APIRouter()

@router.get("/health")
async def health_check():
    return {"status": "ok"}

@router.get("/health/models")
async def health_models(request: Request):
    results = {}
    sample_text = "I have been feeling anxious about work for 3 weeks"

    # Language detector
    try:
        r = request.app.state.lang_detector.detect(sample_text)
        results["language_detector"] = {
            "status": "ok",
            "sample": f"{r['language']} ({r['language_name']}) conf={r['confidence']:.2f}"
        }
    except Exception as e:
        results["language_detector"] = {"status": "error", "detail": str(e)}

    # Translator
    try:
        r = request.app.state.translator.to_english("أنا حزين", "ar")
        results["translator"] = {"status": "ok", "sample": r}
    except Exception as e:
        results["translator"] = {"status": "error", "detail": str(e)}

    # Emotion classifier
    try:
        r = request.app.state.emotion_classifier.predict(sample_text)
        results["emotion_classifier"] = {
            "status": "ok",
            "sample": f"{r['emotion']} conf={r['confidence']:.2f}"
        }
    except Exception as e:
        results["emotion_classifier"] = {"status": "error", "detail": str(e)}

    # Intent classifier
    try:
        r = request.app.state.intent_classifier.predict(sample_text)
        results["intent_classifier"] = {
            "status": "ok",
            "sample": f"intent={r['intent']} crisis={r['is_crisis']}"
        }
    except Exception as e:
        results["intent_classifier"] = {"status": "error", "detail": str(e)}

    # NER extractor
    try:
        r = request.app.state.ner_extractor.extract(sample_text)
        results["ner_extractor"] = {
            "status": "ok",
            "sample": f"symptoms={r['symptoms']} severity={r['severity']}"
        }
    except Exception as e:
        results["ner_extractor"] = {"status": "error", "detail": str(e)}

    # Query optimizer
    try:
        r = request.app.state.query_optimizer.optimize(sample_text)
        results["query_optimizer"] = {
            "status": "ok",
            "sample": f"[{r['method']}] {r['optimized'][:60]}..."
        }
    except Exception as e:
        results["query_optimizer"] = {"status": "error", "detail": str(e)}

    overall = "healthy" if all(v["status"] == "ok" for v in results.values()) else "degraded"
    return {"status": overall, "models": results}