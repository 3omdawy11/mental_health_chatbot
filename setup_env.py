import os
import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    """Sets the seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"[*] Global random seed set to {seed}")

def create_directories():
    """Creates necessary directories for datasets and models."""
    directories = [
        "data/raw",
        "data/processed",
        "data/splits",
        "models/emotion_classifier",
        "models/language_detector",
        "vector_store",
        "api",
        "notebooks" # For Kaggle/Colab training scripts
    ]
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"[+] Directory ready: {directory}")
        except Exception as e:
            print(f"[-] Error creating directory {directory}: {e}")

if __name__ == "__main__":
    print("Initializing Mental Health Chatbot Environment...\n" + "-"*50)
    set_seed(42)
    create_directories()
    
    print("-" * 50)
    print("[*] Ensuring spaCy English model is installed for NER...")
    try:
        import spacy
        if not spacy.util.is_package("en_core_web_sm"):
            os.system("python -m spacy download en_core_web_sm")
            print("[+] spaCy model 'en_core_web_sm' downloaded successfully.")
        else:
            print("[+] spaCy model 'en_core_web_sm' already installed.")
    except Exception as e:
        print(f"[-] Failed to setup spaCy: {e}")
        
    print("-" * 50)
    print("Environment setup complete. Ready for data processing.")