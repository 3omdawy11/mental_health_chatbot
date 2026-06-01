# import os
# from dotenv import load_dotenv
# from src.intent_classifier import IntentClassifier

# def test_intent_classifier():
#     print("Initializing Intent Classifier...")
    
#     # Ensure environment variables are loaded
#     load_dotenv()
    
#     # Double check API Key presence before making the call
#     if not os.getenv("GROQ_API_KEY"):
#         print("❌ Error: GROQ_API_KEY is missing from your environment or .env file.")
#         return

#     try:
#         # Instantiate the class
#         classifier = IntentClassifier()
#         print("✅ Classifier initialized successfully!\n")
#     except Exception as e:
#         print(f"❌ Failed to initialize classifier: {e}")
#         return

#     # A list of diverse test phrases to see how the LLM splits the intents
#     test_phrases = [
#         "Hey! Good morning chatbot.",
#         "I'm feeling incredibly anxious about my exams and can't breathe.",
#         "Can you explain what cognitive behavioral therapy is?",
#         "What is the current standings table for the Premier League?",
#         "Thank you so much, this makes me feel a bit better.",
#         "I can't handle this pain anymore, I want to end it all."
#     ]

#     print("--- Running Test Phrases ---")
#     for phrase in test_phrases:
#         print(f"\nUser Message: \"{phrase}\"")
        
#         # Call the predict method we built
#         result = classifier.predict(phrase)
        
#         # Color or highlight the output in terminal for clarity
#         print(f"  └─ 🎯 Detected Intent: {result['intent']}")
#         print(f"  └─ 🚨 Crisis Flag:     {result['is_crisis']}")
    
#     print("\n----------------------------")
#     print("Test Suite Complete!")

# if __name__ == "__main__":
#     test_intent_classifier()


# import os
# from src.language_detector import LanguageDetector

# def test_language_detector():
#     print("Initializing Language Detector...")
    
#     try:
#         # Instantiating the class wrapper
#         detector = LanguageDetector()
#         print(f"✅ Initialization success: {detector}\n")
#     except FileNotFoundError as e:
#         print(f"❌ Path Error: Could not locate your saved pickle files.")
#         print(e)
#         return
#     except Exception as e:
#         print(f"❌ Error during initialization: {e}")
#         return

#     # Check out the property lists 
#     print(f"Supported Languages in your model: {detector.supported_languages}\n")

#     # Sample texts across languages to check mapping consistency
#     samples = [
#         "I feel completely isolated and tired.",
#         "Me siento muy ansioso y no puedo dormir.",
#         "Je me sens très stressé ces derniers temps.",
#         ""  # Empty edge case test
#     ]

#     print("--- Running Language Test Samples ---")
#     for sample in samples:
#         print(f"\nText Input: \"{sample}\"")
#         result = detector.detect(sample)
#         print(f"  └─ 🌐 Language Code: {result['language']} ({result['language_name']})")
#         print(f"  └─ 📈 Confidence:    {result['confidence']}")

#     print("\n-----------------------------")
#     print("Language Detector Test Suite Complete!")

# if __name__ == "__main__":
#     test_language_detector()



import os
from dotenv import load_dotenv

# Absolute imports enabled by your editable installation (pip install -e .)
from src.intent_classifier import IntentClassifier
from src.language_detector import LanguageDetector
from src.emotion_classifier import EmotionClassifier

def test_complete_pipeline():
    print("====================================================")
    print("🚀 Bootstrapping Chatbot Core Model Test Suite")
    print("====================================================\n")
    
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        print("❌ Error: GROQ_API_KEY missing from environment setup.")
        return

    # ──────────────────────────────────────────────────────────────────
    # 1. Initialize All Components
    # ──────────────────────────────────────────────────────────────────
    try:
        print("📥 Loading Intent Classifier (Groq)...")
        intent_clf = IntentClassifier()
        
        print("📥 Loading Language Detector (TF-IDF + LR)...")
        lang_det = LanguageDetector()
        
        print("📥 Loading Emotion Classifier (BiLSTM)...")
        emotion_clf = EmotionClassifier()
        
        print("\n✅ All models successfully instantiated and ready!\n")
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # ──────────────────────────────────────────────────────────────────
    # 2. Test Cases (Mixed Intent, Language, and Emotions)
    # ──────────────────────────────────────────────────────────────────
    test_inputs = [
        "Hey there! Good morning, how are you?", 
        "I have been feeling really anxious and overwhelmed lately, I can't sleep.",
        "Me siento completamente solo y desesperado hoy, nadie me entiende.",
        "What is the total distance of a marathon race?",
        "Thank you so much, this conversation really helped calm me down.",
        "I want to end my life, please help me."
    ]

    # ──────────────────────────────────────────────────────────────────
    # 3. Execution Sweep
    # ──────────────────────────────────────────────────────────────────
    print("--- Running Pipeline Evaluation Sweep ---")
    for idx, text in enumerate(test_inputs, 1):
        print(f"\n📝 Test #{idx}: \"{text}\"")
        print("-" * 50)
        
        # 🎯 Step A: Intent Prediction
        try:
            intent_res = intent_clf.predict(text)
            print(f"  🔹 Intent   : {intent_res['intent'].upper()} (Crisis Flag: {intent_res['is_crisis']})")
        except Exception as e:
            print(f"  ❌ Intent Detection Failed: {e}")

        # 🌐 Step B: Language Detection
        try:
            lang_res = lang_det.detect(text)
            print(f"  🔹 Language : {lang_res['language']} ({lang_res['language_name']}) | Conf: {lang_res['confidence']:.2f}")
        except Exception as e:
            print(f"  ❌ Language Detection Failed: {e}")

        # 🎭 Step C: Emotion Classification (BiLSTM Forward Pass)
        try:
            emotion_res = emotion_clf.predict(text)
            print(f"  🔹 Emotion  : {emotion_res['emotion'].upper()} | Conf: {emotion_res['confidence']:.2f}")
            if emotion_res['all_scores']:
                top_3 = list(emotion_res['all_scores'].items())[:3]
                print(f"  └─ Top Scores: {', '.join([f'{k}: {v}' for k, v in top_3])}")
        except Exception as e:
            print(f"  ❌ Emotion Classification Failed: {e}")

    print("\n====================================================")
    print("✅ Pipeline Evaluation Sweep Complete!")
    print("====================================================")

if __name__ == "__main__":
    test_complete_pipeline()

# from src.emotion_classifier import EmotionClassifier
# from src.utils.preprocessor import clean_text

# def test_isolated_emotion_classifier():
#     print("====================================================")
#     print("🧠 Isolated Evaluation: BiLSTM Emotion Classifier")
#     print("====================================================\n")
    
#     # ──────────────────────────────────────────────────────────────────
#     # 1. Initialize the Classifier
#     # ──────────────────────────────────────────────────────────────────
#     try:
#         print("📥 Initializing and loading your fine-tuned BiLSTM model weights...")
#         # Auto-detects device type (CUDA / CPU) internally
#         clf = EmotionClassifier()
#         print(f"✅ Class instance ready: {clf}")
#     except FileNotFoundError as e:
#         print("\n❌ Path Error: Missing required model artifacts!")
#         print("Verify your folder 'models/emotion_classifier/' contains:")
#         print("  - emotion_classifier_best_model.pt")
#         print("  - emotion_config.yaml")
#         print("  - tokenizer.pkl")
#         print(f"\nDetails: {e}")
#         return
#     except Exception as e:
#         print(f"❌ Failed during setup initialization: {e}")
#         import traceback
#         traceback.print_exc()
#         return

#     # ──────────────────────────────────────────────────────────────────
#     # 2. Targeted Test Expressions (Evaluating your 6 emotion labels)
#     # ──────────────────────────────────────────────────────────────────
#     test_phrases = [
#         "I feel completely isolated and hopeless today, like nothing will ever get better.", # sadness
#         "Get out of my face! I am absolutely furious at how unfairly I am being treated.", # anger
#         "I am over the moon right now! I just found out I passed my hardest exams!",       # joy
#         "I heard a creepy shattering noise downstairs in the dark and my chest is tight.",# fear
#         "Wow! I completely did not expect to see you walk through that door today!",     # surprise
#         "I appreciate you listening to me so deeply. I feel incredibly safe here."        # love
#     ]

#     # ──────────────────────────────────────────────────────────────────
#     # 3. Running Prediction Sweep
#     # ──────────────────────────────────────────────────────────────────
#     print("\n--- Starting Text Inference Sweep ---")
#     for idx, text in enumerate(test_phrases, 1):
#         print(f"\n📝 Test #{idx}: \"{text}\"")
#         print("-" * 55)
        
#         try:
#             cleaned = clean_text(text)
#             if cleaned != text.strip():
#                 print(f"  🧹 Cleaned text: \"{cleaned}\"")
#             result = clf.predict(cleaned)
            
#             print(f"  🎯 Predicted Emotion : {result['emotion'].upper()}")
#             print(f"  📈 Confidence Score   : {result['confidence']:.4f}")
            
#             if result['all_scores']:
#                 print("  📊 Full Sorted Probability Distribution:")
#                 for emotion, score in result['all_scores'].items():
#                     # Format as a clean visual bar representation for easier terminal debugging
#                     bar = "█" * int(score * 20)
#                     print(f"    └─ {emotion:<9}: {score:.4f} {bar}")
                    
#         except Exception as e:
#             print(f"  ❌ Inference Processing Failed: {e}")
#             import traceback
#             traceback.print_exc()

#     print("\n====================================================")
#     print("✅ Emotion Classifier Test Evaluation Complete!")
#     print("====================================================")

# if __name__ == "__main__":
#     test_isolated_emotion_classifier()