"""
rag_chain.py — Step 2 of the RAG pipeline: retrieval + generation.

Given a user question:
  1. Embed the question (same vector space as the ingested chunks).
  2. Retrieve the top-k most similar chunks from Chroma.
  3. GUARDRAIL: if none of the retrieved chunks are actually relevant
     (similarity below a threshold), refuse to answer instead of letting
     the LLM improvise an answer with no real grounding. This is the
     single most important idea in this project: retrieval isn't optional
     context, it's a check on whether the system is even allowed to answer.
  4. Stuff the relevant chunks into a prompt and call the LLM to generate
     a grounded answer, citing which source file it came from.
"""

import os
import chromadb
from embeddings import TfidfEmbeddingFunction

DB_DIR = "chroma_db"
COLLECTION_NAME = "resume_projects"

TOP_K = 3
# Chroma returns cosine *distance* (0 = identical, 2 = opposite) since the
# collection is configured with hnsw:space="cosine". Anything above this
# threshold is treated as "not actually relevant" — this is the guardrail.
# Tune it empirically: print distances for a few real queries against your
# own corpus and pick a cutoff that separates good matches from noise.
MAX_DISTANCE = 0.75

# Which LLM to use for generation: "ollama" (free, local, no API key) or
# "anthropic" (requires ANTHROPIC_API_KEY + funded account). Can also be
# overridden per-run with: GENERATION_BACKEND=ollama python3 rag_chain.py ...
GENERATION_BACKEND = "ollama"
OLLAMA_MODEL = "llama3.2"


def get_collection():
    embed_fn = TfidfEmbeddingFunction()  # loads the already-fitted vectorizer
    client = chromadb.PersistentClient(path=DB_DIR)
    return client.get_collection(name=COLLECTION_NAME, embedding_function=embed_fn)


def retrieve(question: str, top_k: int = TOP_K):
    collection = get_collection()
    results = collection.query(query_texts=[question], n_results=top_k)

    chunks = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    retrieved = [
        {"text": c, "source": m["source"], "distance": d}
        for c, m, d in zip(chunks, metadatas, distances)
    ]
    return retrieved


def build_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    context = "\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in retrieved_chunks
    )
    return f"""You are answering questions using ONLY the context below.
If the context doesn't contain enough information to answer, say so honestly
instead of guessing. Always mention which source file(s) you used.

Context:
{context}

Question: {question}

Answer:"""


def answer_question(question: str) -> dict:
    """Full RAG pipeline: retrieve -> guardrail check -> generate."""
    retrieved = retrieve(question)

    # --- Guardrail ---
    relevant = [c for c in retrieved if c["distance"] <= MAX_DISTANCE]
    if not relevant:
        return {
            "answer": (
                "I don't have relevant information in my knowledge base to "
                "answer that question confidently, so I won't guess."
            ),
            "sources": [],
            "guardrail_triggered": True,
        }

    prompt = build_prompt(question, relevant)

    # --- Generation ---
    # Two backends supported:
    #   "ollama"    -> free, runs entirely on your machine, no API key needed.
    #                  Requires the Ollama app running locally (ollama.com).
    #   "anthropic" -> requires ANTHROPIC_API_KEY and a funded account.
    # Set GENERATION_BACKEND below (or override with an env var) to choose.
    backend = os.environ.get("GENERATION_BACKEND", GENERATION_BACKEND)

    if backend == "ollama":
        answer_text = _generate_with_ollama(prompt)
    else:
        answer_text = _generate_with_anthropic(prompt)

    return {
        "answer": answer_text,
        "sources": [c["source"] for c in relevant],
        "guardrail_triggered": False,
    }


def _generate_with_ollama(prompt: str) -> str:
    """Call a local Ollama server. Free, no API key, runs on your machine.

    Setup (one-time):
      1. Install Ollama: https://ollama.com/download
      2. Pull a model:   ollama pull llama3.2
      3. Ollama runs a local server automatically after install.
    """
    import urllib.request
    import json as jsonlib

    try:
        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=jsonlib.dumps(
                {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = jsonlib.loads(resp.read())
            return data.get("response", "[Ollama returned no response field]")
    except Exception as e:
        return (
            f"[Ollama call failed: {e}]\n\n"
            f"Make sure Ollama is installed and running (https://ollama.com/download), "
            f"and that you've pulled the model with: ollama pull {OLLAMA_MODEL}\n\n"
            f"Prompt that would have been sent:\n\n{prompt}"
        )


def _generate_with_anthropic(prompt: str) -> str:
    """Call the Anthropic API. Requires ANTHROPIC_API_KEY and API credits."""
    try:
        import anthropic

        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        text_blocks = [b.text for b in response.content if b.type == "text"]
        return text_blocks[0] if text_blocks else "[No text content in response]"
    except anthropic.APIStatusError as e:
        print(f"\n[API ERROR {e.status_code}] {e.message}\n")
        return f"[LLM call failed — see error above. {e.message}]"
    except Exception as e:
        # Falls back to showing the constructed prompt so you can still see
        # the pipeline working end-to-end without an API key configured.
        return (
            f"[No LLM call made — set ANTHROPIC_API_KEY to enable generation. "
            f"Reason: {e}]\n\nThis is the prompt that would have been sent:\n\n{prompt}"
        )


if __name__ == "__main__":
    # Quick manual test from the command line
    import sys

    q = sys.argv[1] if len(sys.argv) > 1 else "What ROC-AUC did the insider threat model achieve?"
    result = answer_question(q)
    print("QUESTION:", q)
    print("\nSOURCES USED:", result["sources"])
    print("\nANSWER:\n", result["answer"])
