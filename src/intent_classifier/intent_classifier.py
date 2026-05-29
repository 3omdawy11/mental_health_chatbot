# src/intent_classifier/intent_classifier.py

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ── 1. Intent Definitions & Few-Shot Examples ─────────────────────────────────

SYSTEM_PROMPT = """
You are an intent classifier for a mental health support chatbot.
Your job is to classify the user's message into exactly one of these intents:

INTENTS:
- greeting        : The user is saying hello or starting a conversation
- goodbye         : The user is ending the conversation
- gratitude       : The user is thanking or expressing appreciation
- asking_mental_health_question : The user is asking about or describing a mental health concern, emotion, or seeking support
- out_of_scope    : The user is asking about something unrelated to mental health

EXAMPLES:
Message: "Hey there!"                                          → greeting
Message: "Hello, I need some help"                            → greeting
Message: "Hi, good morning"                                   → greeting

Message: "Bye, take care"                                     → goodbye
Message: "Thanks, I'll be going now"                          → goodbye
Message: "See you later"                                      → goodbye

Message: "Thank you so much, this really helped"             → gratitude
Message: "I appreciate your support"                          → gratitude
Message: "Thanks a lot"                                       → gratitude

Message: "I've been feeling really anxious lately"            → asking_mental_health_question
Message: "I can't stop feeling sad and I don't know why"      → asking_mental_health_question
Message: "How do I deal with panic attacks?"                  → asking_mental_health_question
Message: "I feel overwhelmed and can't sleep"                 → asking_mental_health_question
Message: "I'm not doing okay"                                 → asking_mental_health_question
Message: "What is depression?"                                → asking_mental_health_question

Message: "What's the weather today?"                          → out_of_scope
Message: "Can you help me with my math homework?"             → out_of_scope
Message: "Who won the football match?"                        → out_of_scope
Message: "What's the best restaurant near me?"                → out_of_scope

IMPORTANT RULES:
- If the user expresses any negative emotion or mental health concern, always classify as asking_mental_health_question
- "I'm not okay", "I'm struggling", "I feel lost" are all asking_mental_health_question
- If the message contains multiple intents, pick the SINGLE most dominant one
- NEVER return more than one label
- NEVER return two words
- Respond with ONLY one of these exact labels, nothing else:
  greeting, goodbye, gratitude, asking_mental_health_question, out_of_scope
"""

# ── 2. Classify Function ──────────────────────────────────────────────────────

VALID_INTENTS = {
    "greeting",
    "goodbye",
    "gratitude",
    "asking_mental_health_question",
    "out_of_scope"
}

def classify_intent(user_message: str) -> str:
    """
    Takes a raw user message and returns one of the 5 intent labels.
    Falls back to 'out_of_scope' if the LLM returns something unexpected.
    """
    response = client.chat.completions.create(
        model    = "llama-3.1-8b-instant",   # fast and free on Groq
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Message: {user_message}"}
        ],
        temperature = 0.0,   # no randomness — we want deterministic classification
        max_tokens  = 10,    # intent label is at most a few words
    )

    intent = response.choices[0].message.content.strip().lower()

    # Validate — if LLM returns something unexpected fall back to out_of_scope
    if intent not in VALID_INTENTS:
        print(f"[INTENT] Unexpected response: '{intent}' — falling back to out_of_scope")
        return "out_of_scope"

    return intent