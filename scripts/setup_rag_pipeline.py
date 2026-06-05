"""
1. Load knowledge_base_combined.json
2. Initialise Embedder
3. Batch-embed all chunks
4. Create Qdrant collection with HNSW
5. Upload chunks + embeddings
6. Verify count and run test queries
"""

import sys, argparse, json, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from dotenv import load_dotenv
load_dotenv()

from src.utils.embedder   import Embedder
from src.utils.vector_db  import VectorDBManager

KB_PATH = ROOT / "data" / "processed" / "knowledge_base_combined.json"

def load_knowledge_base(limit: int | None = None) :
    print(f"\n[1/5] Loading knowledge base from {KB_PATH.name} …")
    if not KB_PATH.exists():
        print(f"   {KB_PATH} not found.")
        return _demo_chunks()

    with open(KB_PATH, encoding="utf-8") as f:
        chunks = json.load(f)

    if limit:
        chunks = chunks[:limit]

    sources: dict[str, int] = {}
    for c in chunks:
        meta = c.get("metadata", {}) or {}
        st = meta.get("source_type", c.get("source_type", "unknown"))
        sources[st] = sources.get(st, 0) + 1
        
    print(f"  Total chunks  : {len(chunks):,}")
    for src, cnt in sources.items():
        print(f"    {src:<30}: {cnt:,}")
    nulls = sum(1 for c in chunks if not c.get("text", "").strip())
    print(f"  Empty texts   : {nulls}")
    return chunks


def _demo_chunks():
    """20 representative chunks for smoke-testing without the KB file."""
    items = [
        ("Anxiety disorders involve persistent and excessive worry. CBT and medication are first-line treatments.",  "counseling", "Anxiety"),
        ("Depression is characterised by persistent low mood, loss of interest, fatigue and hopelessness.",          "counseling", "Depression"),
        ("Panic attacks involve sudden intense fear, racing heartbeat and shortness of breath. They are treatable.", "counseling", "Panic"),
        ("Cognitive Behavioral Therapy helps restructure negative thoughts and develop coping strategies.",           "pdf",        "CBT"),
        ("Insomnia and sleep disorders frequently accompany anxiety and depression. Sleep hygiene techniques help.",  "counseling", "Sleep"),
        ("Mindfulness meditation reduces stress, improves focus and promotes emotional regulation.",                  "pdf",        "Mindfulness"),
        ("Workplace stress and occupational burnout are common triggers for anxiety and depression.",                 "pdf",        "Work"),
        ("Relationship difficulties and interpersonal conflict are major stressors linked to depression.",            "counseling", "Relationships"),
        ("Trauma and PTSD cause flashbacks, hypervigilance and emotional numbing. Trauma therapy is effective.",     "pdf",        "Trauma"),
        ("Self-compassion and self-care practices are essential components of mental health recovery.",               "pdf",        "Self-care"),
        ("Social anxiety disorder involves intense fear of social situations and judgement from others.",             "counseling", "Social Anxiety"),
        ("Grief and bereavement naturally involve sadness, denial, anger and acceptance in varying order.",           "counseling", "Grief"),
        ("Exercise and physical activity significantly reduce symptoms of anxiety and depression.",                   "pdf",        "Exercise"),
        ("Eating disorders involve distorted body image and unhealthy eating patterns requiring specialist care.",    "counseling", "Eating Disorders"),
        ("ADHD affects attention, impulsivity and emotional regulation across the lifespan.",                         "counseling", "ADHD"),
        ("Breathing exercises like diaphragmatic breathing rapidly reduce acute anxiety and panic symptoms.",         "pdf",        "Breathing"),
        ("Building social support networks is a key protective factor against depression and loneliness.",            "pdf",        "Social Support"),
        ("Medication such as SSRIs is often used alongside therapy for moderate to severe depression.",               "counseling", "Medication"),
        ("Boundaries in relationships protect emotional health and reduce resentment and burnout.",                   "pdf",        "Boundaries"),
        ("Crisis intervention and safety planning are essential for people experiencing suicidal thoughts.",          "counseling", "Crisis"),
    ]
    return [
        {
            "id":          f"demo_{i:03d}",
            "text":        text,
            "source":      src,
            "source_type": src,
            "section":     section,
            "tokens":      len(text.split()),
        }
        for i, (text, src, section) in enumerate(items)
    ]

def get_meta(chunk: dict):
    return chunk.get("metadata", {}) or {}

def build_retrieval_text(chunk: dict) :
    meta = get_meta(chunk)

    section = meta.get("section", "") or chunk.get("section", "")
    source_type = meta.get("source_type", "") or chunk.get("source_type", "")
    question = meta.get("original_question", "") or meta.get("context_query", "")
    answer = chunk.get("text", "")

    parts = []
    if section:
        parts.append(f"Section: {section}")

    if source_type in {"counseling_qa", "qa", "faq"}:
        if question:
            parts.append(f"Question: {question}")
        if answer:
            parts.append(f"Answer: {answer}")
    elif source_type == 'pdf_file':
        if section:
            parts.append(f"Question: {section}")
        else:
            parts.append(f"Question: {question}")
        if answer:
            parts.append(f"Answer: {answer}")
    else:
        if answer:
            parts.append(f"Content: {answer}")

    return "\n".join(parts).strip()


def embed_chunks(chunks: list[dict]):
    print(f"\n[2/5] Embedding {len(chunks):,} chunks …")
    emb  = Embedder()
    t0   = time.time()
    texts = [c.get("text", "") for c in chunks]
    vecs  = emb.embed_batch(texts, batch_size=64, show_progress=True)
    elapsed = time.time() - t0
    print(f"  Shape    : {vecs.shape}")
    print(f"  Dtype    : {vecs.dtype}")
    print(f"  Elapsed  : {elapsed:.1f}s  ({len(chunks)/elapsed:.0f} chunks/s)")
    return vecs, emb


def setup_collection(db: VectorDBManager, recreate: bool) :
    print(f"\n[3/5] Setting up Qdrant collection '{db.collection_name}' …")
    created = db.create_collection(recreate=recreate)
    if created:
        print("   Collection created")
    else:
        print("   Collection already exists (use --recreate to rebuild)")
    info = db.collection_info()
    if "error" not in info:
        print(f"  Points: {info.get('count', 0):,}  Status: {info.get('status','?')}")


def index_and_verify(db: VectorDBManager, chunks: list[dict], vecs: np.ndarray) :
    print(f"\n[4/5] Indexing {len(chunks):,} chunks …")
    summary = db.index_chunks(chunks, vecs, show_progress=True)
    print(f"  Uploaded : {summary['uploaded']:,} / {summary['total']:,}")
    print(f"  Elapsed  : {summary['elapsed']}s  ({summary['rate']} chunks/s)")

    count = db.verify_count()
    ok    = count >= len(chunks)
    print(f"  Verified : {count:,} points in Qdrant  {'Ok' if ok else '** mismatch'}")


def test_queries(db: VectorDBManager, emb: Embedder) :
    print(f"\n[5/5] Test queries …")
    queries = [
        ("I feel anxious about work",           0.0),
        ("How do I manage depression?",          0.0),
        ("I can't sleep, sleep issues",          0.0),
        ("panic attack symptoms and treatment",  0.0),
    ]
    all_ok = True
    for query, _ in queries:
        t0      = time.time()
        results = db.search_by_text(query, emb, limit=3, score_threshold=0.0)
        ms      = (time.time() - t0) * 1000
        ok      = len(results) > 0 and ms < 500
        all_ok  = all_ok and ok
        mark    = "Ok" if ok else "**"
        print(f"\n  {mark} [{ms:.0f}ms] \"{query}\"")
        for i, r in enumerate(results, 1):
            print(f"     {i}. score={r['score']:.3f}  [{r['source_type']}]  {r['text'][:65]}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--recreate",   action="store_true", help="Delete and rebuild collection")
    p.add_argument("--limit",      type=int, default=None, help="Index only first N chunks")
    p.add_argument("--no-verify",  action="store_true",   help="Skip test queries")
    return p.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print("  vector Database Setup")
    print("=" * 60)

    db = VectorDBManager()
    print(f"\n  {db}")

    chunks          = load_knowledge_base(limit=args.limit)
    vecs, emb       = embed_chunks(chunks)
    setup_collection(db, recreate=args.recreate)
    index_and_verify(db, chunks, vecs)

    if not args.no_verify:
        test_queries(db, emb)

    count = db.verify_count()
    print("\n" + "=" * 60)
    print(f"  Collection : {db.collection_name}")
    print(f"  Mode       : {'cloud' if db.is_cloud else 'local (set QDRANT_URL for cloud)'}")
    print(f"  Points     : {count:,}")
    print("=" * 60)


if __name__ == "__main__":
    main()