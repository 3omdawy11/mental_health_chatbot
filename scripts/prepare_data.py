"""
==========================
Phase 1 automated pipeline:
  1. Load datasets
  2. Clean & preprocess text
  3. Create train/val/test splits
  4. Extract & chunk PDFs (if any exist)
  5. Build combined knowledge base
"""
import json
import re
from pathlib import Path
from typing import Optional, Callable  # Add this line
from datetime import datetime
import logging
import pandas as pd
import shutil
import sys

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from src.utils.preprocessor import apply_cleaning, print_cleaning_stats
from src.utils.docling_utils import process_pdf_directory, load_chunks, save_chunks

# Paths
IN_KAGGLE = "/kaggle" in str(Path.cwd())
if IN_KAGGLE:
    print("run 12 ")
try:
    DATA_RAW = Path("/kaggle/input")
    DATA_PROC = Path("/kaggle/working/data/processed")
    DATA_SPLITS = Path("/kaggle/working/data/splits")
    PDF_DIR = DATA_RAW / "datasets/ziadmahmoudamr/mental-health-chatbot/mental_health_chatbot/data/raw/mental_health_books"
    
    if not PDF_DIR.exists():
        candidates = list(DATA_RAW.glob("**/mental_health_books"))
        PDF_DIR = candidates[0] if candidates else PDF_DIR

except:
    DATA_RAW = ROOT / "data" / "raw"
    DATA_PROC = ROOT / "data" / "processed"
    DATA_SPLITS = ROOT / "data" / "splits"
    PDF_DIR = DATA_RAW / "mental_health_books"

for d in [DATA_PROC, DATA_SPLITS]:
    d.mkdir(parents=True, exist_ok=True)

if PDF_DIR.exists():
    print(f"  Found PDF directory: {PDF_DIR}")
else:
    print(f"  PDF directory not found: {PDF_DIR}")



# Load datasets
def load_language_dataset():
    print("\n[1/5] Loading Language Identification dataset …")
    ds = load_dataset("papluca/language-identification")
    # Combine all splits so we can re-split ourselves
    frames = []
    for split_name in ds.keys():
        df = ds[split_name].to_pandas()
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={"labels": "language"})
    print(f"  Loaded {len(df):,} rows | {df['language'].nunique()} languages")
    print(f"  Class distribution:\n{df['language'].value_counts().to_string()}")
    return df


def load_emotion_dataset():
    print("\n[1/5] Loading Emotion dataset …")
    ds = load_dataset("dair-ai/emotion")
    frames = []
    for split_name in ds.keys():
        df = ds[split_name].to_pandas()
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    label_map = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}
    df["emotion"] = df["label"].map(label_map)
    # df = df.drop(columns=["label"]) I guess I'll keep the label for now
    print(f"  Loaded {len(df):,} rows | {df['emotion'].nunique()} emotions")
    print(f"  Class distribution:\n{df['emotion'].value_counts().to_string()}")
    return df


def load_mental_health_dataset():
    print("\n[1/5] Loading Mental Health Counseling dataset …")
    ds = load_dataset("Amod/mental_health_counseling_conversations")
    frames = []
    for split_name in ds.keys():
        df = ds[split_name].to_pandas()
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={"Context": "context", "Response": "response"})
    print(f"  Loaded {len(df):,} rows")
    return df


# Clean datasets

def clean_language(df: pd.DataFrame):
    print("\n[2/5] Cleaning Language Identification …")
    df, stats = apply_cleaning(df, text_col="text", min_words=3)
    print_cleaning_stats("Language ID", stats)
    return df


def clean_emotion(df: pd.DataFrame):
    print("\n[2/5] Cleaning Emotion dataset …")
    df, stats = apply_cleaning(df, text_col="text", min_words=3)
    print_cleaning_stats("Emotion", stats)
    return df


def clean_mental_health(df: pd.DataFrame):
    # since multiple answers to the same question are valuable for RAG, then I've decided to preserve diverse responses
    print("\n[2/5] Cleaning Mental Health dataset …")
    
    # preserve duplicates that have different responses
    df, stats_ctx = apply_cleaning(
        df,
        text_col="context",
        min_words=5,
        pair_col="response",
    )
    print_cleaning_stats("Mental Health — context", stats_ctx)
    from src.utils.preprocessor import clean_text
    df["response"] = df["response"].fillna("").apply(clean_text)

    df = df[df["response"].str.split().apply(len) >= 5].reset_index(drop=True)
    print(f"  After response filter: {len(df):,} rows remain")
    
    context_counts = df["context"].value_counts()
    duplicated_questions = (context_counts > 1).sum()
    total_duplicate_responses = context_counts[context_counts > 1].sum()
    
    print(f"\n  Duplicate Context Info (PRESERVED for RAG):")
    print(f"    Questions with multiple answers: {duplicated_questions:,}")
    print(f"    Total responses for duplicated questions: {total_duplicate_responses:,}")
    print(f"    Avg responses per duplicated question: {total_duplicate_responses / max(duplicated_questions, 1):.2f}")
    
    return df


# Train / val /test splits

def stratified_split(
    df: pd.DataFrame,
    label_col: str,
    val_size: float = 0.10,
    test_size: float = 0.10,
    random_state: int = 42,):
    """80 / 10 / 10 stratified split."""
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df[label_col], random_state=random_state
    )
    # Then split val from train_val
    relative_val = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=relative_val,
        stratify=train_val[label_col],
        random_state=random_state,
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def verify_split_distribution(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, label_col: str, name: str
):
    all_labels = set(train[label_col].unique())
    val_labels = set(val[label_col].unique())
    test_labels = set(test[label_col].unique())
    missing_val = all_labels - val_labels
    missing_test = all_labels - test_labels
    if missing_val or missing_test:
        print(f"  * {name}: missing in val={missing_val} | missing in test={missing_test}")
    else:
        print(f"  **  {name}: all {len(all_labels)} classes present in every split")
    print(f"     train={len(train):,}  val={len(val):,}  test={len(test):,}")


def make_splits(
    lang_df: pd.DataFrame, emotion_df: pd.DataFrame
):
    print("\n[3/5] Creating train/val/test splits …")

    lang_train, lang_val, lang_test = stratified_split(lang_df, "language")
    verify_split_distribution(lang_train, lang_val, lang_test, "language", "Language ID")

    em_train, em_val, em_test = stratified_split(emotion_df, "emotion")
    verify_split_distribution(em_train, em_val, em_test, "emotion", "Emotion")

    return {
        "language": (lang_train, lang_val, lang_test),
        "emotion": (em_train, em_val, em_test),
    }


def save_splits(splits: dict):
    for name, (train, val, test) in splits.items():
        train.to_csv(DATA_SPLITS / f"{name}_train.csv", index=False)
        val.to_csv(DATA_SPLITS / f"{name}_val.csv", index=False)
        test.to_csv(DATA_SPLITS / f"{name}_test.csv", index=False)
        print(f"  Saved: {name}_{{train,val,test}}.csv")


# PDF extraction

def extract_pdfs(embedding_model: Optional[Callable] = None):
    print("\n[4/5] Extracting PDFs …")
    
    if not PDF_DIR.exists():
        print(f"  PDF directory not found: {PDF_DIR}")
        return []
    
    pdf_files = list(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"  No PDFs found in {PDF_DIR}. Skipping PDF extraction.")
        return []
    
    print(f"  Found {len(pdf_files)} PDF files")
    
    chunks = process_pdf_directory(
        pdf_dir=PDF_DIR,
        output_path=DATA_PROC / "knowledge_base_chunks.json",
        embedding_model=embedding_model,
    )
    return chunks


# Build combined knowledge base

def _calculate_counseling_quality_score(context: str, response: str):
    """
    i'll calculate quality score based on response length, comprehensiveness, and empathy indicators.
    """
    response_tokens = len(response) // 4
    context_tokens = len(context) // 4
    
    if response_tokens < 50:
        length_score = response_tokens / 50 * 0.4  # Too short
    elif response_tokens < 100:
        length_score = 0.7
    elif response_tokens < 400:
        length_score = 1.0  # Sweet spot for counseling
    else:
        # Penalize overly long responses slightly
        length_score = max(0.8, 1.0 - (response_tokens - 400) / 2000)
    
    # Empathy indicators
    empathy_keywords = [
        "understand", "feel", "support", "help", "important", "valid",
        "appreciate", "listen", "compassion", "concern", "care",
        "recognize", "acknowledge", "respect", "hear you"
    ]
    empathy_count = sum(
        response.lower().count(keyword) for keyword in empathy_keywords
    )
    empathy_score = min(1.0, empathy_count / 3)  # Normalize
    
    relevance_score = min(1.0, context_tokens / 100)
    
    quality_score = (
        length_score * 0.5 +
        empathy_score * 0.35 +
        relevance_score * 0.15
    )
    
    return round(quality_score, 3)


def _generate_counseling_context_query(context: str, max_length: int = 150):
    # For counseling, we use the question directly as it's the retrieval key.
    query = context.strip()
    if len(query) > max_length:
        # Truncate at word boundary
        query = query[:max_length].rsplit(' ', 1)[0] + "..."
    return query


def build_knowledge_base(
    mh_df: pd.DataFrame,
    pdf_chunks: list[dict],
    embedding_model: Optional[Callable] = None,
):
    """
    Build combined RAG-optimized knowledge base.
    Tracks response diversity for better relevance scoring.
    """
    print("\n[5/5] Building combined knowledge base …")
    
    from collections import Counter
    from datetime import datetime, timezone
    created_date = datetime.now(timezone.utc).isoformat()
    
    # Count how many responses exist for each context (question)
    context_response_count = mh_df.groupby("context").size()
    
    counseling_chunks = []
    for i, row in mh_df.iterrows():
        question = str(row['context']).strip()
        answer = str(row['response']).strip()
        
        tokens = len(answer) // 4
        quality_score = _calculate_counseling_quality_score(question, answer)
        context_query = _generate_counseling_context_query(question)
        
        # Track response diversity
        num_responses_for_this_question = context_response_count.get(question, 1)
        
        embedding_vector = None
        if embedding_model is not None:
            try:
                embedding_vector = embedding_model(answer)
            except Exception as exc:
                logger.warning(f"Failed to generate embedding for counseling chunk {i}: {exc}")
        
        counseling_chunks.append(
            {
                "chunk_id": f"counseling_{i:05d}",
                "text": answer,
                "metadata": {
                    "source": "counseling_dataset",
                    "source_type": "counseling_qa",
                    "section": "Mental Health Counseling",
                    "tokens": tokens,
                    "context_query": context_query,
                    "quality_rating": quality_score,
                    "created_date": created_date,
                    "embedding_vector": embedding_vector,
                    "original_question": question,
                    "response_diversity": num_responses_for_this_question, 
                    "is_unique_answer": num_responses_for_this_question == 1,  
                }
            }
        )
    
    combined = counseling_chunks + pdf_chunks
    
    for i, chunk in enumerate(combined):
        chunk["chunk_id"] = f"kb_{i:05d}"
    
    save_chunks(combined, DATA_PROC / "knowledge_base_combined.json")
    
    source_counts: dict[str, int] = {}
    quality_stats: dict[str, list[float]] = {}
    diversity_stats = []
    total_tokens = 0
    
    for chunk in combined:
        source_type = chunk.get("metadata", {}).get("source_type", "unknown")
        quality = chunk.get("metadata", {}).get("quality_rating", 0.0)
        tokens = chunk.get("metadata", {}).get("tokens", 0)
        diversity = chunk.get("metadata", {}).get("response_diversity", 1)
        
        source_counts[source_type] = source_counts.get(source_type, 0) + 1
        if source_type not in quality_stats:
            quality_stats[source_type] = []
        quality_stats[source_type].append(quality)
        
        if source_type == "counseling_qa":
            diversity_stats.append(diversity)
        
        total_tokens += tokens
    
    avg_quality_by_source = {
        src: round(sum(scores) / len(scores), 3)
        for src, scores in quality_stats.items()
    }
    
    print(f"\n  Knowledge Base Summary")
    print(f"  {'─' * 60}")
    print(f"  Total chunks              : {len(combined):,}")
    print(f"\n  Breakdown by source:")
    for src, count in sorted(source_counts.items()):
        avg_quality = avg_quality_by_source.get(src, 0.0)
        print(f"    {src:<28}: {count:>6,}  (avg quality: {avg_quality:.3f})")
    
    if diversity_stats:
        print(f"\n  Response Diversity (Counseling):")
        print(f"    Questions with 1 answer  : {sum(1 for d in diversity_stats if d == 1):,}")
        print(f"    Questions with 2-3 answers: {sum(1 for d in diversity_stats if 2 <= d <= 3):,}")
        print(f"    Questions with 4+ answers : {sum(1 for d in diversity_stats if d >= 4):,}")
        print(f"    Max answers for one Q     : {max(diversity_stats)}")
        print(f"    Avg answers per Q        : {sum(diversity_stats) / len(diversity_stats):.2f}")
    
    print(f"\n  Token Statistics")
    print(f"  Total estimated tokens    : {total_tokens:,}")
    print(f"  Average tokens/chunk      : {total_tokens // max(len(combined), 1):,}")
    
    all_quality_scores = [
        chunk.get("metadata", {}).get("quality_rating", 0.0)
        for chunk in combined
    ]
    if all_quality_scores:
        print(f"\n  Quality Rating Distribution:")
        print(f"    Min  : {min(all_quality_scores):.3f}")
        print(f"    Avg  : {round(sum(all_quality_scores) / len(all_quality_scores), 3):.3f}")
        print(f"    Max  : {max(all_quality_scores):.3f}")
    
    high_quality = sum(1 for q in all_quality_scores if q >= 0.7)
    medium_quality = sum(1 for q in all_quality_scores if 0.5 <= q < 0.7)
    low_quality = sum(1 for q in all_quality_scores if q < 0.5)
    print(f"\n  High Quality (≥0.7)       : {high_quality:,}")
    print(f"  Medium Quality (0.5-0.7)  : {medium_quality:,}")
    print(f"  Low Quality (<0.5)        : {low_quality:,}")
    
    print(f"\n  Output saved to: {DATA_PROC / 'knowledge_base_combined.json'}")
    print(f"  {'─' * 60}\n")
    
    return combined

def download_data_to_output():
    OUTPUT_DIR = Path("/kaggle/output")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Copy processed datasets
    print("\n Copying data to output folder for download...")
    
    source_files = [
        DATA_PROC / "language_identification_cleaned.csv",
        DATA_PROC / "emotion_cleaned.csv",
        DATA_PROC / "mental_health_cleaned.csv",
        DATA_PROC / "knowledge_base_chunks.json",
        DATA_PROC / "knowledge_base_combined.json",
    ]
    
    split_files = list(DATA_SPLITS.glob("*.csv"))
    
    all_files = source_files + split_files
    
    for src in all_files:
        if src.exists():
            dst = OUTPUT_DIR / src.name
            shutil.copy2(src, dst)
            size_mb = src.stat().st_size / 1024 / 1024
            print(f"  ok {src.name} ({size_mb:.2f} MB)")
        else:
            print(f"  ✗ {src.name} (NOT FOUND)")
    
    print(f"  Total files: {len(list(OUTPUT_DIR.glob('*')))}")
# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main(embedding_model: Optional[Callable] = None):
    print("=" * 60)
    print("  Mental Health Chatbot — Phase 1: Data Preparation")
    print("=" * 60)

    lang_df = load_language_dataset()
    emotion_df = load_emotion_dataset()
    mh_df = load_mental_health_dataset()

    lang_df = clean_language(lang_df)
    emotion_df = clean_emotion(emotion_df)
    mh_df = clean_mental_health(mh_df)

    lang_df.to_csv(DATA_PROC / "language_identification_cleaned.csv", index=False)
    emotion_df.to_csv(DATA_PROC / "emotion_cleaned.csv", index=False)
    mh_df.to_csv(DATA_PROC / "mental_health_cleaned.csv", index=False)
    print(f"\n  Saved processed datasets → {DATA_PROC}")

    splits = make_splits(lang_df, emotion_df)
    save_splits(splits)

    pdf_chunks = extract_pdfs(embedding_model=embedding_model)

    kb = build_knowledge_base(mh_df, pdf_chunks, embedding_model=embedding_model)

    print("\n" + "=" * 60)
    print("   Phase 1 Complete — Output Summary")
    print("=" * 60)
    output_files = [
        DATA_PROC / "language_identification_cleaned.csv",
        DATA_PROC / "emotion_cleaned.csv",
        DATA_PROC / "mental_health_cleaned.csv",
        DATA_PROC / "knowledge_base_combined.json",
        DATA_SPLITS / "language_train.csv",
        DATA_SPLITS / "language_val.csv",
        DATA_SPLITS / "language_test.csv",
        DATA_SPLITS / "emotion_train.csv",
        DATA_SPLITS / "emotion_val.csv",
        DATA_SPLITS / "emotion_test.csv",
    ]
    if pdf_chunks:
        output_files.insert(3, DATA_PROC / "knowledge_base_chunks.json")

    for f in output_files:
        size = f"{f.stat().st_size / 1024:.1f} KB" if f.exists() else "MISSING"
        try:
            display_path = f.relative_to(ROOT)
        except ValueError:
            display_path = f
        print(f"  {'ok' if f.exists() else '✗'}  {display_path}  ({size})")
    
    print("=" * 60)


if __name__ == "__main__":
    embedding_model = None
    try:
        from sentence_transformers import SentenceTransformer
        print("Loading embedding model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embedding_model = lambda text: model.encode(text).tolist()
    except ImportError:
        print("sentence-transformers not available. Skipping embeddings.")
    
    main(embedding_model=embedding_model)
    if IN_KAGGLE:
        download_data_to_output()