# SafeSpace AI

**RAG-Based Mental Health Support Chatbot** — NLP Final Project

> SafeSpace AI is an end-to-end, retrieval-augmented generation chatbot designed to provide grounded, empathetic, and context-aware support for users dealing with anxiety, depression, stress, and related mental health topics.

---

## Repositories

| Component | Repository |
|-----------|------------|
| AI + FastAPI Backend | https://github.com/3omdawy11/SafeSpaceAI |
| React Frontend | https://github.com/ZeyadMahmoudAmrMohamed/SafeSpaceAI |

The backend contains the full NLP pipeline, RAG engine, REST API, and all four module implementations. The frontend provides a conversational UI that connects to the FastAPI backend in real time.

---

## Project Overview

SafeSpace AI integrates four distinct NLP modules into a single, tightly coupled pipeline. A user message flows through language detection, emotion classification, and intent routing before reaching the RAG engine, which retrieves relevant mental health knowledge and generates a final response — all returned in the user's original language.

---

## System Architecture

```
                        User Message
                             |
                             v
              +-----------------------------+
              |    Module 1: Language        |
              |    Detection                 |
              |  TF-IDF + Logistic Reg.      |
              |  (20 languages)              |
              +-----------------------------+
                             |
               Non-English --+--> English (pass-through)
                             |
                    [Translate to English]
                             |
                             v
              +-----------------------------+
              |    Module 2: Emotion         |
              |    Classification            |
              |  BiLSTM (fine-tuned)         |
              |  joy | sadness | anger |     |
              |  fear | love | surprise      |
              +-----------------------------+
                             |
                             v
              +-----------------------------+
              |    Module 3: Intent          |
              |    Classification            |
              |  Zero/Few-shot LLM (Groq)    |
              +-----------------------------+
                    |               |
          greeting /         asking_mental_
          goodbye /          health_question
          gratitude /               |
          out_of_scope              v
                |       +----------------------+
                |       |  Module 4: RAG        |
                |       |  Pipeline             |
                |       |  BM25 + Semantic      |
                |       |  Search (Qdrant)      |
                |       |  HyDE Query Expansion |
                |       |  NER Extraction       |
                |       |  Groq LLM Generation  |
                |       +----------------------+
                |               |
                +-------+-------+
                        |
               [Translate back if non-English]
                        |
                        v
                  Final Response
```

---

## Module Details

### Module 1: Language Detection

A traditional NLP multi-class classifier that identifies the language of an incoming user query across 20 languages. This module gates the translation step and ensures all downstream modules operate on English text.

| Property | Detail |
|----------|--------|
| Dataset | Language Identification Dataset (~90k samples, HuggingFace) |
| Vectorization | TF-IDF on character n-grams |
| Classifier | Logistic Regression |
| Output | Detected language code (e.g. `en`, `ar`) + confidence score |

**Confidence-based fallback pipeline.** When the model's top prediction falls below a confidence threshold, the system scans the top-N predicted languages and returns the highest-priority one found among them. Priority order reflects approximate global speaker population and web prevalence, making the detector robust under uncertainty rather than defaulting blindly to the top softmax output.

Non-English queries are translated to English before continuing to the next module, and the final response is translated back to the original language before delivery.

---

### Module 2: Emotion Classification

A deep learning multi-class classifier that identifies the emotional state expressed in the user's query. The detected emotion directly shapes how the RAG module frames its final response, enabling empathetic and tone-appropriate replies.

| Property | Detail |
|----------|--------|
| Base Dataset | Emotion Dataset (~6k Twitter messages, HuggingFace) |
| Model | BiLSTM, fine-tuned on Kaggle GPU |
| Classes | joy, sadness, anger, fear, love, surprise |
| Output | Emotion label |

**Handling class imbalance with synthetic data generation.** The original dataset was heavily skewed toward joy and sadness. To address this, we synthesized an additional 16,112 entries using `llama-3.3-70b-versatile`, targeting the underrepresented classes:

| Emotion | Original | Synthetic | Total |
|---------|----------|-----------|-------|
| joy | 5,286 | 0 | 5,286 |
| sadness | 4,512 | 774 | 5,286 |
| anger | 2,108 | 3,178 | 5,286 |
| fear | 1,850 | 3,436 | 5,286 |
| love | 1,291 | 3,995 | 5,286 |
| surprise | 557 | 4,729 | 5,286 |

---

### Module 3: Intent Classification

Classifies the user's intent into one of five categories and routes the system to the correct response path — without any additional model training. This avoids unnecessary RAG calls for simple conversational turns.

| Property | Detail |
|----------|--------|
| Method | Zero-shot / few-shot prompting via Groq LLM |
| Classes | `greeting`, `goodbye`, `gratitude`, `asking_mental_health_question`, `out_of_scope` |

**Routing logic:**
- `greeting`, `goodbye`, `gratitude` — direct LLM response, no RAG invoked
- `asking_mental_health_question` — triggers the full RAG pipeline
- `out_of_scope` — polite decline with redirection

---

### Module 4: RAG Pipeline

The core knowledge-retrieval and generation module. When a query is classified as a mental health question, this pipeline retrieves the most relevant passages from the knowledge base and synthesizes a grounded, empathetic response conditioned on both the retrieved chunks and the detected emotion from Module 2.

| Property | Detail |
|----------|--------|
| Dataset | Mental Health Counseling Conversations (~17k professional Q&A pairs, HuggingFace) + curated domain PDFs |
| Embeddings | `all-MiniLM-L6-v2` (Sentence Transformers) |
| Vector DB | Qdrant Cloud (HNSW indexing) |
| LLM | Groq free tier (`gpt-oss-120b` / `gpt-oss-20b`) |

**PDF extraction with Docling.** Curated mental health PDFs are parsed using Docling, which handles complex layouts, multi-column text, and embedded structure more reliably than basic text extractors. Extracted content is then passed into the chunking stage.

**Semantic chunking.** Documents are chunked using cosine similarity between sentence embeddings rather than fixed token windows. Semantically coherent passages are grouped together, producing chunks that preserve meaning and improve retrieval precision.

**Hybrid search.** Retrieval combines BM25 keyword search with dense semantic similarity search over Qdrant. This ensures both exact-term matches and conceptually related passages are surfaced, complementing each other's weaknesses.

**HyDE query expansion.** Before retrieval, the system generates a Hypothetical Document Embedding (HyDE) — a synthetic passage that a relevant answer might look like — and uses it alongside the original query to improve recall for abstractly-phrased questions.

**NER extraction.** Named entity recognition identifies symptoms and emotional triggers mentioned in the query, which are used to refine retrieval and response generation.

---

## Technology Stack

| Technology | Role |
|------------|------|
| Python 3.13+ | Core language for all modules and scripts |
| FastAPI + uvicorn | REST API server |
| React | Frontend conversational UI |
| PyTorch | Deep learning backend (BiLSTM) |
| Hugging Face | Transformers, datasets, model hub |
| Sentence Transformers | `all-MiniLM-L6-v2` embeddings |
| Qdrant | Cloud vector database (HNSW indexing) |
| Groq | Free-tier LLM inference API |
| scikit-learn | TF-IDF, Logistic Regression, preprocessing |
| Weights & Biases | Experiment tracking during training |
| Docling | PDF extraction from curated mental health documents |
| pandas / numpy | Data manipulation and analysis |

---

## Datasets

| Dataset | Size | Type | Source |
|---------|------|------|--------|
| Language Identification | ~90,000 samples | 20 languages | HuggingFace (auto-downloaded) |
| Emotion Detection | ~6,000 original + 16,112 synthetic | 6 emotion classes | HuggingFace (Twitter messages) |
| Mental Health Counseling | ~17,000 Q&A pairs | Professional counseling | HuggingFace — core RAG knowledge |
| Mental Health PDFs | Custom | Curated books and guides | Manually placed in `data/raw/` |

---

## Setup and Usage

### Environment Variables

Create a `.env` file in the project root:

```
GROQ_API_KEY       # https://console.groq.com/
QDRANT_URL         # https://cloud.qdrant.io/
QDRANT_API_KEY     # Generated from Qdrant dashboard
WANDB_API_KEY      # wandb login locally, or set in Kaggle secrets
```

### Running the Pipeline

Run the following from the backend repository root, in order:

```bash
pip install -r requirements.txt

# Download datasets and chunk PDFs
python scripts/00_prepare_data.py

# Train the TF-IDF language detector
python scripts/01_train_language_detector.py

# Index the knowledge base to Qdrant
python scripts/03_setup_rag.py

# Start the FastAPI server on localhost:8000
uvicorn app.main:app --reload

# Run full pipeline smoke tests
python scripts/04_test_pipeline.py
```

---

## API Reference

### `POST /chat`

Main inference endpoint. Accepts a user message and returns a full pipeline response.

**Response fields:**

| Field | Description |
|-------|-------------|
| `response` | Generated text response (translated back to original language if needed) |
| `emotion` | Detected emotion label (e.g. `fear`, `sadness`) |
| `language` | Detected language code (e.g. `en`, `ar`) |
| `intent` | Classified intent (e.g. `asking_mental_health_question`) |
| `confidence_scores` | Per-module confidence values (language, emotion, intent) |
| `sources` | Retrieved knowledge chunks with source attribution |

---

## Limitations

- **Not a therapist.** Responses are AI-generated from training data, not from licensed mental health professionals. Users in genuine distress should always be directed to real professional resources.
- **Knowledge base bounds.** RAG output quality is directly bounded by the quality and coverage of the indexed PDFs and counseling dataset.
- **English-biased training.** The emotion classifier and LLM are trained and prompted primarily in English. Translation quality for low-resource languages may vary.
- **No persistent user state.** This is a proof-of-concept; there is no cross-session conversation history or user profile.
- **API rate limits.** The free Groq tier (~30 requests/min) is sufficient for development but not for production workloads.

---

## Future Work

- Expand the knowledge base with additional peer-reviewed mental health resources and clinical guidelines.
- Fine-tune the LLM on domain-specific mental health data for more nuanced responses.
- Implement persistent conversation history and user context across sessions.
- Extend multi-language support end-to-end, including multilingual emotion classification.
- Integrate a feedback-driven fine-tuning loop using collected user ratings.
