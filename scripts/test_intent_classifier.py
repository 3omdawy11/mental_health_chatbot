from src.intent_classifier import IntentClassifier

test_messages = [
    "Hey, how are you?",
    "I've been feeling really anxious lately",
    "Thank you so much for your help",
    "Bye, take care",
    "What's the best pizza place near me?",
    "I can't sleep and feel hopeless",
    "I'm not okay",            
    "Thanks, goodbye!",   
]

print(f"{'Message':<45} {'Intent'}")
print("-" * 70)
for msg in test_messages:
    intent = IntentClassifier().predict(msg)
    print(f"{msg:<45} {intent}")