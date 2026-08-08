"""
ingest.py — Step 1 of the RAG pipeline: load documents, chunk them,
embed them, and store them in a local Chroma vector database.

Run: python3 ingest.py
"""

import os
import glob
import chromadb
from embeddings import TfidfEmbeddingFunction

DOCS_DIR = "sample_docs"
DB_DIR = "chroma_db"
COLLECTION_NAME = "resume_projects"

# --- Chunking config ---
# CHUNK_SIZE: how many characters per chunk. Smaller = more precise retrieval
#             but less surrounding context per chunk.
# CHUNK_OVERLAP: chunks overlap so we don't accidentally cut a sentence/idea
#                in half at a chunk boundary and lose its meaning.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Split text into overlapping chunks by character count.

    This is the simplest possible chunking strategy. In production you'd
    usually chunk by sentence/paragraph boundaries (e.g. LangChain's
    RecursiveCharacterTextSplitter) so you don't cut mid-sentence, but
    building this by hand first is how you actually understand what
    that abstraction is doing for you.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap  # step forward, leaving overlap
    return [c.strip() for c in chunks if c.strip()]


def main():
    doc_paths = glob.glob(os.path.join(DOCS_DIR, "*.txt"))
    print(f"Found {len(doc_paths)} document(s) to ingest.")

    all_chunks, all_ids, all_metadatas = [], [], []

    for path in doc_paths:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = chunk_text(text)
        filename = os.path.basename(path)
        print(f"  {filename}: {len(text)} chars -> {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"{filename}::chunk_{i}")
            all_metadatas.append({"source": filename, "chunk_index": i})

    # Fit the embedding backend on the whole corpus BEFORE creating the
    # collection, so the vector space is defined consistently across all
    # chunks (and reused later for queries).
    embed_fn = TfidfEmbeddingFunction()
    embed_fn.fit(all_chunks)

    # Persistent client = data survives between runs (stored on disk in DB_DIR)
    client = chromadb.PersistentClient(path=DB_DIR)
    # Fresh collection each ingest run, so re-running never duplicates data.
    client.delete_collection(name=COLLECTION_NAME) if COLLECTION_NAME in [
        c.name for c in client.list_collections()
    ] else None
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},  # cosine distance, not default L2
    )

    # This is where embedding actually happens: each chunk of text is
    # converted into a vector and stored alongside its original text + metadata.
    collection.add(documents=all_chunks, ids=all_ids, metadatas=all_metadatas)

    print(f"\nIngested {len(all_chunks)} chunks into collection '{COLLECTION_NAME}'.")
    print(f"Vector DB persisted at: {os.path.abspath(DB_DIR)}")


if __name__ == "__main__":
    main()
