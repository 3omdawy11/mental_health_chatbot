# scripts/test_intent_classifier.py


from src.intent_classifier import classify_intent

test_messages = [
    "Hey, how are you?",
    "I've been feeling really anxious lately",
    "Thank you so much for your help",
    "Bye, take care",
    "What's the best pizza place near me?",
    "I can't sleep and feel hopeless",    # edge case
    "I'm not okay",                        # edge case
    "Thanks, goodbye!",                    # mixed intent
]

print(f"{'Message':<45} {'Intent'}")
print("-" * 70)
for msg in test_messages:
    intent = classify_intent(msg)
    print(f"{msg:<45} {intent}")