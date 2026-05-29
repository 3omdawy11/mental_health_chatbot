"""
PDF extraction and semantic chunking using Docling.
Falls back to basic extraction if Docling is not installed.
Generates RAG-optimized chunks with enhanced metadata structure.
"""
import json
import re
from pathlib import Path
from datetime import datetime, timezone
import logging
from typing import Optional, Callable
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


# ── Chunking helpers (used regardless of extraction method) ─────────────────

def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return max(1, len(text) // 4)


def _generate_context_query(section: str, text: str, max_length: int = 150) -> str:
    """
    Generate a context query from section header and text excerpt.
    Useful for semantic matching in RAG retrieval.
    """
    # Use section as base
    if section and section != "Introduction":
        query = f"About {section}: "
    else:
        query = ""
    
    # Add first few sentences from text
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    excerpt = " ".join(sentences[:2])
    if len(excerpt) > max_length:
        excerpt = excerpt[:max_length].rsplit(' ', 1)[0] + "..."
    
    query += excerpt
    return query.strip()


def _split_by_paragraphs(text: str, max_chars: int = 2000) -> list[str]:
    """Split a long section into paragraph-sized chunks."""
    paragraphs = re.split(r"\n{2,}", text.strip())
    chunks, current = [], ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current) + len(para) + 2 <= max_chars:
            current = (current + "\n\n" + para).strip()
        else:
            if current:
                chunks.append(current)
            # If single paragraph is already too big, hard-split it
            if len(para) > max_chars:
                for i in range(0, len(para), max_chars):
                    chunks.append(para[i : i + max_chars])
            else:
                current = para
    if current:
        chunks.append(current)
    return chunks

def _calculate_quality_score(text: str, tokens: int) -> float:
    """
    Calculate a quality/relevance score for a chunk (0.0 to 1.0).
    Considers length appropriateness, sentence structure, and information density.
    """
    if tokens == 0:
        return 0.0
    
    # Penalize very short chunks, reward medium-length ones
    if tokens < 50:
        length_score = tokens / 50 * 0.5
    elif tokens < 300:
        length_score = 1.0
    else:
        # Penalize overly long chunks
        length_score = max(0.6, 1.0 - (tokens - 300) / 1000)
    
    # Check for sentence structure (simple heuristic)
    sentences = len(re.split(r'[.!?]+', text))
    if sentences == 0:
        structure_score = 0.3
    else:
        avg_sentence_length = tokens / sentences
        if 15 <= avg_sentence_length <= 30:
            structure_score = 1.0
        else:
            structure_score = 0.8
    
    # Overall score
    quality_score = (length_score * 0.6 + structure_score * 0.4)
    return round(quality_score, 3)

# ─────────────────────────────────────────────────────────────────────────────


def _split_sentences(text: str) -> list[str]:
    """
    Split text into sentences using a simple but robust regex.
    Handles abbreviations, decimals, and common edge-cases.
    """
    # Protect common abbreviations
    abbrev = re.compile(
        r"\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|e\.g|i\.e|Fig|Eq|Vol|No)\.",
        re.IGNORECASE,
    )
    protected = abbrev.sub(lambda m: m.group(0).replace(".", "<<<DOT>>>"), text)

    # Split on sentence-ending punctuation followed by whitespace + capital
    raw = re.split(r"(?<=[.!?])\s+(?=[A-Z\"\'\(])", protected)

    sentences = []
    for s in raw:
        restored = s.replace("<<<DOT>>>", ".")
        restored = restored.strip()
        if restored:
            sentences.append(restored)
    return sentences


def _tfidf_embedder(sentences: list[str]) -> np.ndarray:
    """
    Fit a TF-IDF vectorizer on all sentences and return dense L2-normalised
    vectors. Used as the fallback when no external embedding_model is provided.
    """
    if len(sentences) == 1:
        # Edge-case: single sentence → unit vector of length 1
        return np.ones((1, 1), dtype=np.float32)

    vec = TfidfVectorizer(min_df=1, sublinear_tf=True)
    X = vec.fit_transform(sentences).toarray().astype(np.float32)

    # L2-normalise rows so cosine_similarity == dot product
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


def _compute_breakpoints(
    sentences: list[str],
    embedding_model: Optional[Callable],
    percentile_threshold: float = 80.0,
) -> list[int]:
    """
    Core semantic chunking logic (Kamradt / "percentile" method).

    1. Embed every sentence.
    2. Compute cosine similarity between each consecutive sentence pair.
    3. Wherever the similarity DROPS below the (100 - percentile_threshold)
       percentile it is a *semantic boundary* → split there.

    Returns the indices (into `sentences`) just AFTER each boundary.
    """
    if len(sentences) <= 1:
        return []

    # Build embedding matrix  (n_sentences × dim)
    if embedding_model is not None:
        try:
            matrix = np.array([embedding_model(s) for s in sentences], dtype=np.float32)
        except Exception as exc:
            logger.warning(f"embedding_model failed, falling back to TF-IDF: {exc}")
            matrix = _tfidf_embedder(sentences)
    else:
        matrix = _tfidf_embedder(sentences)

    # Pairwise cosine similarities between consecutive sentences
    # cosine_similarity returns shape (n, n); we only need the off-diagonal band
    adjacent_sims = np.array(
        [
            float(cosine_similarity(matrix[i : i + 1], matrix[i + 1 : i + 2])[0][0])
            for i in range(len(sentences) - 1)
        ]
    )

    # Low similarity → high "distance"
    distances = 1.0 - adjacent_sims

    # Breakpoints wherever distance exceeds the chosen percentile
    threshold = float(np.percentile(distances, percentile_threshold))
    breakpoints = [i + 1 for i, d in enumerate(distances) if d >= threshold]
    return breakpoints


def _merge_short_chunks(
    chunks: list[str],
    min_chunk_chars: int,
    max_section_chars: int,
) -> list[str]:
    """
    After splitting, merge any chunk that is too short into its neighbour.
    Also guard against chunks that somehow exceeded max_section_chars.
    """
    merged: list[str] = []
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        if merged and len(merged[-1]) < min_chunk_chars:
            # Absorb this chunk into the previous one
            candidate = merged[-1] + " " + chunk
            if len(candidate) <= max_section_chars:
                merged[-1] = candidate
                continue
        merged.append(chunk)

    # Final pass: drop any still-too-short orphan
    return [c for c in merged if len(c) >= min_chunk_chars]


def _semantic_split(
    text: str,
    max_section_chars: int,
    min_chunk_chars: int,
    embedding_model: Optional[Callable],
    percentile_threshold: float = 80.0,
) -> list[str]:
    """
    True semantic splitting for a single block of text.

    Steps
    -----
    1. Sentence-tokenise.
    2. Find semantic breakpoints via cosine-distance percentile.
    3. Group sentences into candidate chunks.
    4. Hard-cap any chunk > max_section_chars by naively splitting on the
       nearest sentence boundary.
    5. Merge orphan (too-short) chunks.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return []

    breakpoints = _compute_breakpoints(sentences, embedding_model, percentile_threshold)

    # Build raw chunk strings
    raw_chunks: list[str] = []
    prev = 0
    for bp in breakpoints:
        raw_chunks.append(" ".join(sentences[prev:bp]))
        prev = bp
    raw_chunks.append(" ".join(sentences[prev:]))

    # Hard-cap: if any chunk is still too long, fall back to sentence-boundary split
    capped: list[str] = []
    for chunk in raw_chunks:
        if len(chunk) <= max_section_chars:
            capped.append(chunk)
        else:
            # Re-split by walking sentences until we hit the cap
            chunk_sentences = _split_sentences(chunk)
            buf: list[str] = []
            buf_len = 0
            for sent in chunk_sentences:
                if buf and buf_len + len(sent) + 1 > max_section_chars:
                    capped.append(" ".join(buf))
                    buf = [sent]
                    buf_len = len(sent)
                else:
                    buf.append(sent)
                    buf_len += len(sent) + 1
            if buf:
                capped.append(" ".join(buf))

    return _merge_short_chunks(capped, min_chunk_chars, max_section_chars)


# ════════════════════════════════════════════════════════════════════════
# Public api
# ══════════════════════════════════

def semantic_chunk(
    text: str,
    source_file: str,
    max_section_chars: int = 2000,
    min_chunk_chars: int = 100,
    embedding_model: Optional[Callable] = None,
) -> list[dict]:
    """
    Split text into semantic chunks optimised for RAG using true semantic
    similarity rather than recursive character splitting.

    Algorithm
    ---------
    1. Split the document on markdown headers (##, ###, etc.) to keep
       structural boundaries intact and preserve ``current_section`` context.
    2. For each section, sentence-tokenise the content and compute cosine
       similarity between every consecutive sentence pair.
    3. Wherever similarity drops below the configurable percentile threshold,
       insert a chunk boundary (Kamradt percentile method).
    4. Hard-cap any chunk exceeding *max_section_chars* at the nearest sentence
       boundary, then merge any orphan chunk shorter than *min_chunk_chars*.
    5. If *embedding_model* is provided, use it to embed sentences; otherwise
       fall back to TF-IDF vectors (no external dependencies required).

    The output dict structure and all field names are identical to the original
    implementation so this function is a drop-in replacement.

    Args:
        text: The document text to chunk.
        source_file: Source filename for tracking.
        max_section_chars: Hard cap on characters per chunk.
        min_chunk_chars: Minimum characters to retain a chunk.
        embedding_model: Optional callable ``(str) -> vector`` for embeddings.

    Returns:
        list of dicts with RAG-optimised structure including metadata.
    """
    # ── 1. Split on markdown headers ─────────────────────────────────────────
    header_pattern = re.compile(r"(^#{1,4}\s.+$)", re.MULTILINE)
    parts = header_pattern.split(text)

    chunks: list[dict] = []
    current_section = "Introduction"
    chunk_id = 0
    created_date = datetime.now(timezone.utc).isoformat()

    i = 0
    while i < len(parts):
        part = parts[i].strip()
        if not part:
            i += 1
            continue

        # Is this part a header?
        if header_pattern.match(part):
            current_section = re.sub(r"^#+\s*", "", part).strip()
            i += 1
            continue

        # ── 2–4. True semantic splitting ──────────────────────────────────────
        sub_chunks = _semantic_split(
            part,
            max_section_chars=max_section_chars,
            min_chunk_chars=min_chunk_chars,
            embedding_model=embedding_model,
        )

        for sub in sub_chunks:
            tokens = _estimate_tokens(sub)
            quality_score = _calculate_quality_score(sub, tokens)
            context_query = _generate_context_query(current_section, sub)

            # Generate embedding if model provided, otherwise None
            embedding_vector = None
            if embedding_model is not None:
                try:
                    embedding_vector = embedding_model(sub)
                except Exception as exc:
                    logger.warning(
                        f"Failed to generate embedding for chunk {chunk_id}: {exc}"
                    )

            chunk_dict = {
                "chunk_id": f"{Path(source_file).stem}_chunk_{chunk_id:04d}",
                "text": sub,
                "metadata": {
                    "source": source_file,
                    "source_type": "pdf_file",
                    "section": current_section,
                    "tokens": tokens,
                    "context_query": context_query,
                    "quality_rating": quality_score,
                    "created_date": created_date,
                    "embedding_vector": embedding_vector,
                },
            }
            chunks.append(chunk_dict)
            chunk_id += 1

        i += 1

    return chunks


# ── smoke-test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    SAMPLE = """
## Introduction

This document explores the history of artificial intelligence.
The field started in the 1950s when Alan Turing proposed the famous Turing Test.
Early researchers were optimistic about achieving human-level AI within decades.
However, progress was slower than expected due to hardware limitations.

## Deep Learning Era

The rise of deep learning transformed the field completely.
Neural networks with many layers could learn representations directly from raw data.
GPUs made it possible to train much larger models on much larger datasets.
Convolutional neural networks achieved superhuman performance on image tasks.
Recurrent networks and later Transformers revolutionised natural language processing.

These advances led to practical applications in translation, speech recognition, and recommendation systems.
Companies began investing heavily in AI research and talent.
Academic benchmarks were surpassed one after another at an accelerating pace.

## Current Challenges

Despite the rapid progress, many challenges remain unsolved.
Models still struggle with robust reasoning and generalisation outside their training distribution.
Hallucination in large language models remains a significant reliability concern.
The computational cost of training frontier models has grown exponentially.
Questions around safety, alignment, and societal impact have gained urgency.

Researchers are actively working on interpretability to understand what models actually learn.
Alignment techniques aim to ensure models act in accordance with human values.
Efficient architectures try to reduce the energy footprint of AI training.
"""

    results = semantic_chunk(SAMPLE, "test_document.pdf", max_section_chars=600, min_chunk_chars=80)
    print(f"Total chunks: {len(results)}\n")
    for r in results:
        print(f"[{r['chunk_id']}] section={r['metadata']['section']!r:30s} "
              f"chars={len(r['text']):4d}  tokens={r['metadata']['tokens']:4d}  "
              f"quality={r['metadata']['quality_rating']:.3f}")
        print(f"  text preview: {r['text'][:90].replace(chr(10),' ')}...")
        print()
# ── Docling extraction ───────────────────────────────────────────────────────

def _extract_with_docling(pdf_path: Path) -> str:
    """Use Docling to convert PDF → markdown text."""
    from docling.document_converter import DocumentConverter  # type: ignore

    converter = DocumentConverter()
    result = converter.convert(str(pdf_path))
    return result.document.export_to_markdown()


def _extract_with_pypdf(pdf_path: Path) -> str:
    """Fallback: plain text extraction with pypdf."""
    try:
        import pypdf  # type: ignore

        reader = pypdf.PdfReader(str(pdf_path))
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        return "\n\n".join(pages)
    except ImportError:
        logger.warning("pypdf not installed. Trying pdfminer.")
        from pdfminer.high_level import extract_text  # type: ignore

        return extract_text(str(pdf_path))


def extract_pdf(pdf_path: Path, use_docling: bool = True) -> str:
    """Extract text from a PDF, preferring Docling."""
    try:
        if use_docling:
            return _extract_with_docling(pdf_path)
    except Exception as exc:
        logger.warning(f"Docling failed for {pdf_path.name}: {exc}. Falling back.")
    return _extract_with_pypdf(pdf_path)


# ── Public API ───────────────────────────────────────────────────────────────

def process_pdf_directory(
    pdf_dir: str | Path,
    output_path: str | Path,
    use_docling: bool = True,
    max_section_chars: int = 2000,
    embedding_model: Optional[callable] = None,
) -> list[dict]:
    """
    Process all PDFs in a directory:
      1. Extract text (Docling or fallback)
      2. Semantic chunk with RAG optimization
      3. Save to JSON with enhanced metadata
    
    Args:
        pdf_dir: Directory containing PDF files
        output_path: Output JSON file path
        use_docling: Whether to use Docling for extraction
        max_section_chars: Maximum section length for chunking
        embedding_model: Optional callable for embedding generation
    
    Returns:
        Combined list of chunk dicts with metadata
    """
    pdf_dir = Path(pdf_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDF files found in {pdf_dir}")
        return []

    all_chunks: list[dict] = []
    for pdf_path in pdf_files:
        print(f"  Processing: {pdf_path.name}")
        try:
            text = extract_pdf(pdf_path, use_docling=use_docling)
            chunks = semantic_chunk(
                text,
                source_file=pdf_path.name,
                max_section_chars=max_section_chars,
                embedding_model=embedding_model,
            )
            all_chunks.extend(chunks)
            print(f"    → {len(chunks)} chunks extracted")
        except Exception as exc:
            logger.error(f"Failed to process {pdf_path.name}: {exc}")

    # Re-index IDs globally and maintain consistency
    for i, chunk in enumerate(all_chunks):
        chunk["chunk_id"] = f"pdf_chunk_{i:05d}"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"\n  Saved {len(all_chunks)} PDF chunks → {output_path}")
    return all_chunks


def load_chunks(path: str | Path) -> list[dict]:
    """Load chunks from a JSON file."""
    path = Path(path)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)



class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy and pandas types."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, np.generic):
            return obj.item()
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)


def save_chunks(chunks: list[dict], path: str | Path) -> None:
    """
    Save chunks to a JSON file with proper serialization.
    Uses custom encoder to handle numpy/pandas types.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
    
    logger.info(f"Saved {len(chunks)} chunks to {path}")

def filter_chunks_by_quality(chunks: list[dict], min_rating: float = 0.5) -> list[dict]:
    """Filter chunks by quality rating threshold."""
    return [
        c for c in chunks
        if c.get("metadata", {}).get("quality_rating", 0) >= min_rating
    ]


def filter_chunks_by_date(
    chunks: list[dict],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> list[dict]:
    """Filter chunks by creation date range (ISO format strings)."""
    filtered = chunks
    if start_date:
        filtered = [
            c for c in filtered
            if c.get("metadata", {}).get("created_date", "") >= start_date
        ]
    if end_date:
        filtered = [
            c for c in filtered
            if c.get("metadata", {}).get("created_date", "") <= end_date
        ]
    return filtered