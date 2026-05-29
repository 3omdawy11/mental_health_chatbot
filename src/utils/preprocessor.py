import re
import pandas as pd
import numpy as np

def clean_text(text: str) -> str:
    """
    Cleans text by removing URLs, emails, and normalizing whitespaces.
    Preserves unicode alphanumeric characters (accents) and emojis.
    """
    if not isinstance(text, str):
        return ""
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove emails
    text = re.sub(r'\S+@\S+\.\S+', '', text)
    # Normalize multiple whitespaces/newlines into a single space
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def preprocess_dataframe(df: pd.DataFrame, text_column: str, min_words: int = 5) -> pd.DataFrame:
    """
    Applies text cleaning, drops duplicates, and filters out short texts.
    Returns the cleaned DataFrame along with processing metrics.
    """
    initial_shape = df.shape[0]
    
    # 1. Fill missing values
    df = df.copy()
    df[text_column] = df[text_column].fillna("").astype(str)
    
    # 2. Clean text
    df[text_column] = df[text_column].apply(clean_text)
    
    # 3. Drop empty strings after cleaning
    df = df[df[text_column] != ""]
    
    # 4. Remove duplicates based on text content
    df = df.drop_duplicates(subset=[text_column])
    after_dedup_shape = df.shape[0]
    
    # 5. Filter out short rows (< min_words)
    df['word_count'] = df[text_column].apply(lambda x: len(x.split()))
    df = df[df['word_count'] >= min_words]
    df = df.drop(columns=['word_count'])
    
    final_shape = df.shape[0]
    
    print(f"[Info] Preprocessing Stats for column '{text_column}':")
    print(f"  - Initial Rows: {initial_shape}")
    print(f"  - Removed Duplicates/Empties: {initial_shape - after_dedup_shape}")
    print(f"  - Removed Short Texts (< {min_words} words): {after_dedup_shape - final_shape}")
    print(f"  - Final Rows: {final_shape}")
    
    return df