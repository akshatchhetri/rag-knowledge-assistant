# RAG Project Assistant

A retrieval-augmented generation (RAG) system that answers questions grounded
in a local knowledge base, with a relevance guardrail to prevent
hallucinated answers when the knowledge base doesn't actually contain the
answer.

## How it works

```
sample_docs/*.txt --> ingest.py --> chunk --> embed --> chroma_db/ (vector store)

user question --> rag_chain.py --> embed question --> retrieve top-k chunks
                --> guardrail check (similarity threshold)
                --> if relevant: build prompt --> call LLM --> grounded answer
                --> if not relevant: refuse instead of guessing
```

## Setup

```bash
pip install chromadb fastapi "uvicorn[standard]" anthropic scikit-learn

# 1. Ingest your documents (chunks + embeds + stores them)
python3 ingest.py
```

### Option A — Free, runs locally (recommended)

Uses [Ollama](https://ollama.com/download) to run an open-source LLM on your
own machine. No API key, no cost, no rate limits.

```bash
# 1. Install Ollama from https://ollama.com/download, then:
ollama pull llama3.2

# 2. rag_chain.py already defaults to the "ollama" backend, so just run:
python3 rag_chain.py "What ROC-AUC did the insider threat model achieve?"
```

### Option B — Anthropic API (requires funded account)

```bash
export ANTHROPIC_API_KEY=sk-...
export GENERATION_BACKEND=anthropic
python3 rag_chain.py "What ROC-AUC did the insider threat model achieve?"
```

### Running the API server (either backend)

```bash
uvicorn api:app --reload
# then open http://127.0.0.1:8000/docs
```

## Files

| File | Purpose |
|---|---|
| `sample_docs/` | Source documents to ingest — swap in your own `.txt` files |
| `embeddings.py` | Pluggable embedding backend (TF-IDF by default; see note below) |
| `ingest.py` | Chunks documents, embeds them, stores in Chroma |
| `rag_chain.py` | Retrieval + guardrail + LLM generation |
| `api.py` | FastAPI wrapper exposing `/ask` |

## A note on the embedding backend

This project uses TF-IDF (via scikit-learn) as the default embedding
method, because it runs fully locally with zero setup — no model download,
no API key needed just to try retrieval. It's a real, classic embedding
technique, but it matches on word overlap rather than semantic meaning
(it won't know "car" and "automobile" are related).

**To upgrade to real semantic embeddings** once you're on a machine with
normal internet access, swap in a neural embedding model — e.g.
`sentence-transformers` (`all-MiniLM-L6-v2`) for a free local model, or an
API-based embedding model (OpenAI, Voyage, etc.). Only `embeddings.py`
needs to change; `ingest.py` and `rag_chain.py` stay the same, since they
just call whatever embedding function they're given. That's the value of
keeping ingestion/retrieval decoupled from the embedding implementation.

## The guardrail

`rag_chain.py` checks the cosine distance of the best-retrieved chunk
against `MAX_DISTANCE`. If nothing retrieved is close enough to be
genuinely relevant, the system returns a refusal instead of asking the LLM
to answer from a weak or irrelevant context. This is a small but real
example of an AI reliability control — the kind of thing production RAG
systems need so they don't confidently make things up.

## Talking points for interviews

- **Chunking trade-offs:** smaller chunks = more precise retrieval but less
  surrounding context; overlap prevents cutting an idea in half at a chunk
  boundary.
- **Why a guardrail matters:** retrieval isn't just "extra context" — it's
  a gate on whether the system should answer at all. Without a relevance
  check, RAG systems will retrieve *something* even for irrelevant
  questions and the LLM will often use it anyway.
- **Distance metric calibration:** I originally used L2 distance with
  un-normalized TF-IDF vectors and got meaningless thresholds; switching to
  cosine distance with L2-normalized vectors fixed it. Small detail, but it
  is the kind of thing that silently breaks a guardrail in production if
  you don't check it empirically.
- **Decoupled design:** the embedding backend is swappable without
  touching ingestion or retrieval logic — same idea as a repository
  pattern, useful when you want to swap TF-IDF for a real neural embedding
  model later.
