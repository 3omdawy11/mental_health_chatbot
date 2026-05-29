import os
import json
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset

# Local utilities (ensuring paths resolve correctly on Kaggle)
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.preprocessor import clean_text
from src.utils.docling_utils import extract_pdf_directory

def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def ensure_directories():
    paths = ["data/processed", "data/splits"]
    for path in paths:
        os.makedirs(path, exist_ok=True)

def generate_stratified_splits(df: pd.DataFrame, target_col: str, prefix: str):
    """Generates precise 80/10/10 stratified train/val/test splits."""
    try:
        train_df, temp_df = train_test_split(
            df, test_size=0.20, stratify=df[target_col], random_state=42
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.50, stratify=temp_df[target_col], random_state=42
        )
        
        train_df.to_csv(f"data/splits/{prefix}_train.csv", index=False)
        val_df.to_csv(f"data/splits/{prefix}_val.csv", index=False)
        test_df.to_csv(f"data/splits/{prefix}_test.csv", index=False)
        print(f"[+] Exported splits for {prefix} (Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)})")
    except Exception as e:
        print(f"[-] Stratification failed for {prefix}: {e}. Falling back to standard split.")
        train_df, temp_df = train_test_split(df, test_size=0.20, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42)
        train_df.to_csv(f"data/splits/{prefix}_train.csv", index=False)
        val_df.to_csv(f"data/splits/{prefix}_val.csv", index=False)
        test_df.to_csv(f"data/splits/{prefix}_test.csv", index=False)

def main():
    print("="*60)
    print("PHASE 1 KAGGLE DATA PROCESSING PIPELINE")
    print("="*60)
    
    set_all_seeds(42)
    ensure_directories()
    
    # -------------------------------------------------------------
    # Language Identification Dataset (Deduplicate by text only)
    # -------------------------------------------------------------
    print("\n[*] Processing Language Identification Dataset...")
    try:
        lang_dataset = load_dataset("papluca/language-identification")
        lang_df = pd.concat([pd.DataFrame(lang_dataset[k]) for k in lang_dataset.keys()], ignore_index=True)
        
        lang_df['text'] = lang_df['text'].fillna("").astype(str).apply(clean_text)
        lang_df = lang_df.drop_duplicates(subset=['text'])
        lang_df = lang_df[lang_df['text'].apply(lambda x: len(x.split())) >= 5]
        
        lang_df.to_csv("data/processed/language_identification_cleaned.csv", index=False)
        generate_stratified_splits(lang_df, target_col='labels', prefix='language')
    except Exception as e:
        print(f"[-] Error parsing language identification dataset: {e}")

    # -------------------------------------------------------------
    # Emotion Dataset (Deduplicate by text only)
    # -------------------------------------------------------------
    print("\n[*] Processing Emotion Dataset...")
    try:
        emotion_dataset = load_dataset("dair-ai/emotion")
        emotion_df = pd.concat([pd.DataFrame(emotion_dataset[k]) for k in emotion_dataset.keys()], ignore_index=True)
        
        emotion_df['text'] = emotion_df['text'].fillna("").astype(str).apply(clean_text)
        emotion_df = emotion_df.drop_duplicates(subset=['text'])
        emotion_df = emotion_df[emotion_df['text'].apply(lambda x: len(x.split())) >= 5]
        
        emotion_df.to_csv("data/processed/emotion_cleaned.csv", index=False)
        generate_stratified_splits(emotion_df, target_col='label', prefix='emotion')
    except Exception as e:
        print(f"[-] Error parsing emotion dataset: {e}")

    # -------------------------------------------------------------
    # Mental Health Counseling Dataset (FIXED DEDUPLICATION)
    # -------------------------------------------------------------
    print("\n[*] Processing Mental Health Counseling Dataset...")
    try:
        mh_dataset = load_dataset("Amod/mental_health_counseling_conversations")
        mh_df = pd.DataFrame(mh_dataset['train'])
        
        initial_rows = len(mh_df)
        mh_df['Context'] = mh_df['Context'].fillna("").astype(str).apply(clean_text)
        mh_df['Response'] = mh_df['Response'].fillna("").astype(str).apply(clean_text)
        
        # FIX: Deduplicate using BOTH columns so we preserve unique counselor answers
        mh_df = mh_df.drop_duplicates(subset=['Context', 'Response'])
        
        # Filter short low-quality context/responses
        mh_df = mh_df[mh_df['Context'].apply(lambda x: len(x.split())) >= 5]
        mh_df = mh_df[mh_df['Response'].apply(lambda x: len(x.split())) >= 5]
        
        mh_df.to_csv("data/processed/mental_health_cleaned.csv", index=False)
        print(f"[+] Cleaned mental health conversations. Retained {len(mh_df)} out of {initial_rows} samples (Preserved alternative responses!)")
    except Exception as e:
        print(f"[-] Error parsing counseling dataset: {e}")
        mh_df = pd.DataFrame()

    # -------------------------------------------------------------
    # PDF Extraction via Docling (FIXED PATH FOR KAGGLE INPUT)
    # -------------------------------------------------------------
    kaggle_pdf_path = "/kaggle/input/datasets/ziadmahmoudamr/mental-health-chatbot/mental_health_chatbot/data/raw/mental_health_books"
    print(f"\n[*] Parsing PDFs from Kaggle input path: {kaggle_pdf_path}")
    
    pdf_chunks = []
    if os.path.exists(kaggle_pdf_path):
        pdf_chunks = extract_pdf_directory(kaggle_pdf_path)
    else:
        print("[-] Kaggle dataset path not found. Falling back to local check.")
        pdf_chunks = extract_pdf_directory("data/raw/mental_health_books")
    
    with open("data/processed/knowledge_base_chunks.json", "w", encoding="utf-8") as f:
        json.dump(pdf_chunks, f, ensure_ascii=False, indent=2)

    # -------------------------------------------------------------
    # Combine Knowledge Base Elements
    # -------------------------------------------------------------
    print("\n[*] Integrating Knowledge Base Framework...")
    combined_knowledge_base = []
    chunk_counter = 1
    
    for chunk in pdf_chunks:
        combined_knowledge_base.append({
            "chunk_id": f"chunk_{chunk_counter:05d}",
            "text": chunk["text"],
            "metadata": chunk["metadata"]
        })
        chunk_counter += 1
        
    if not mh_df.empty:
        for idx, row in mh_df.iterrows():
            combined_knowledge_base.append({
                "chunk_id": f"chunk_{chunk_counter:05d}",
                "text": row["Response"],
                "metadata": {
                    "source": "counseling_dataset",
                    "context_query": row["Context"],
                    "type": "qa_response"
                }
            })
            chunk_counter += 1
            
    with open("data/processed/knowledge_base_combined.json", "w", encoding="utf-8") as f:
        json.dump(combined_knowledge_base, f, ensure_ascii=False, indent=2)
        
    print("\n" + "="*60)
    print("PIPELINE PROCESSING COMPLETE")
    print("="*60)
    print(f"Total compiled chunks in Knowledge Base: {len(combined_knowledge_base)}")
    print(f"  - From PDF Documents: {len(pdf_chunks)}")
    print(f"  - From Q&A Responses: {len(combined_knowledge_base) - len(pdf_chunks)}")

if __name__ == "__main__":
    main()