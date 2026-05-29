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

logger = logging.getLogger(__name__)


# ── Chunking helpers (used regardless of extraction method) ─────────────────

def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return len(text) // 4


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


def semantic_chunk(
    text: str,
    source_file: str,
    max_section_chars: int = 2000,
    min_chunk_chars: int = 100,
    embedding_model: Optional[Callable] = None,
) -> list[dict]:
    """
    Split text into semantic chunks optimized for RAG:
      1. Split on markdown headers (##, ###, etc.)
      2. If a section > max_section_chars → split by paragraphs
      3. Filter out very short chunks
      4. Generate enhanced metadata with context queries and quality scores
    
    Args:
        text: The document text to chunk
        source_file: Source filename for tracking
        max_section_chars: Maximum characters per section
        min_chunk_chars: Minimum characters to keep a chunk
        embedding_model: Optional callable that takes text and returns embedding vector
    
    Returns:
        list of dicts with RAG-optimized structure including metadata
    """
    # Split on headers
    header_pattern = re.compile(r"(^#{1,4}\s.+$)", re.MULTILINE)
    parts = header_pattern.split(text)

    chunks: list[dict] = []
    current_section = "Introduction"
    chunk_id = 0
    # created_date = datetime.utcnow().isoformat()
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

        # It's content — decide whether to split further
        if len(part) > max_section_chars:
            sub_chunks = _split_by_paragraphs(part, max_section_chars)
        else:
            sub_chunks = [part]

        for sub in sub_chunks:
            sub = sub.strip()
            if len(sub) < min_chunk_chars:
                continue
            
            tokens = _estimate_tokens(sub)
            quality_score = _calculate_quality_score(sub, tokens)
            context_query = _generate_context_query(current_section, sub)
            
            # Generate embedding if model provided, otherwise None
            embedding_vector = None
            if embedding_model is not None:
                try:
                    embedding_vector = embedding_model(sub)
                except Exception as exc:
                    logger.warning(f"Failed to generate embedding for chunk {chunk_id}: {exc}")
            
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
                }
            }
            chunks.append(chunk_dict)
            chunk_id += 1
        i += 1

    return chunks


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