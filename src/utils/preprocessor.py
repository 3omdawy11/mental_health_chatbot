import re
import pandas as pd
from typing import Optional



def remove_urls(text: str) -> str:
    """Remove http/https URLs and bare www.* links."""
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"www\.\S+", "", text)
    return text


def remove_emails(text: str) -> str:
    return re.sub(r"\S+@\S+\.\S+", "", text)


def normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces/newlines into a single space and strip."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)  # keep max 2 newlines
    return text.strip()


def remove_special_characters(text: str, keep_punctuation: bool = True) -> str:
    """
    Remove unwanted special characters.
    - Keeps letters (including accented), digits, whitespace.
    - Optionally keeps common punctuation.
    - Keeps emojis (useful for emotion dataset).
    """
    if keep_punctuation:
        # keep letters, digits, whitespace, basic punctuation
        pattern = r"[^\w\s.,!?;:'\"\-–—()\[\]@#&%\U0001F300-\U0001FAFF]"
    else:
        pattern = r"[^\w\s\U0001F300-\U0001FAFF]"
    return re.sub(pattern, " ", text, flags=re.UNICODE)


def clean_text(
    text: str,
    remove_urls_flag: bool = True,
    remove_emails_flag: bool = True,
    remove_special: bool = True,
    keep_punctuation: bool = True,
) -> str:
    if not isinstance(text, str):
        return ""
    if remove_urls_flag:
        text = remove_urls(text)
    if remove_emails_flag:
        text = remove_emails(text)
    if remove_special:
        text = remove_special_characters(text, keep_punctuation)
    text = normalize_whitespace(text)
    return text



def word_count(text: str) -> int:
    return len(str(text).split())

def apply_cleaning(
    df: pd.DataFrame,
    text_col: str,
    min_words: int = 5,
    pair_col: Optional[str] = None,
    **clean_kwargs,
) -> tuple[pd.DataFrame, dict]:
    """
    Apply cleaning to a DataFrame column and return (cleaned_df, stats).
    
    For Q&A datasets, handles duplicates intelligently:
    - Removes duplicate pairs (same context + same response)
    - Preserves different responses for same context (valuable for RAG)
    """
    stats: dict = {
        "original_rows": len(df),
        "duplicate_contexts_retained": 0,
    }

    #  Drop nulls
    df = df.dropna(subset=[text_col]).copy()
    stats["removed_nulls"] = stats["original_rows"] - len(df)

    #  Clean text
    df[text_col] = df[text_col].apply(lambda t: clean_text(t, **clean_kwargs))

    #  Remove very short texts
    mask_short = df[text_col].apply(word_count) < min_words
    stats["removed_short"] = mask_short.sum()
    df = df[~mask_short].copy()

    #  Remove duplicate PAIRS (not just text_col)
    # For Q&A: keep different responses for same question
    before_dedup = len(df)
    if pair_col is not None:
        # For Q&A datasets: drop duplicate (context, response) pairs
        # This preserves different answers to the same question
        subset_cols = [text_col, pair_col]
        df = df.drop_duplicates(subset=subset_cols).reset_index(drop=True)
    else:
        # Fallback: just deduplicate on text_col
        df = df.drop_duplicates(subset=[text_col]).reset_index(drop=True)
    
    stats["removed_duplicate_pairs"] = before_dedup - len(df)
    
    # Count how many contexts have multiple responses (useful for RAG)
    if pair_col is not None:
        response_counts = df[text_col].value_counts()
        duplicated_contexts = (response_counts > 1).sum()
        stats["duplicate_contexts_retained"] = duplicated_contexts

    stats["final_rows"] = len(df)
    stats["after_clean"] = len(df)
    
    return df, stats

def print_cleaning_stats(title: str, stats: dict) -> None:
    print(f"  Original rows    : {stats.get('original_rows', 0):,}")
    print(f"  Removed (null)   : {stats.get('removed_nulls', 0):,}")
    print(f"  Removed (short)  : {stats.get('removed_short', 0):,}")
    
    removed_dedup = stats.get(
        'removed_duplicate_pairs',  
        stats.get('removed_duplicates', 0)
    )
    print(f"  Removed (dedup)  : {removed_dedup:,}")
    
    # Q&A specific stats
    dup_contexts = stats.get('duplicate_contexts_retained', 0)
    if dup_contexts > 0:
        print(f"  ok Preserved diverse Q&A : {dup_contexts:,} questions")
        print(f"    (with multiple valid answers)")
    
    final = stats.get('final_rows', stats.get('after_clean', 0))
    print(f"  Final rows       : {final:,}")
    print()