"""
embeddings.py — pluggable embedding backends for the RAG pipeline.

Why this file exists:
Chroma's default embedding function downloads a small neural model
(all-MiniLM-L6-v2) the first time you use it. That works fine on a normal
laptop with open internet access. In network-restricted environments
(like this sandbox) that download can be blocked.

For learning/demo purposes here, we use a TF-IDF embedding (classic,
pure-local, no download required) as the DEFAULT. TF-IDF isn't a neural
embedding — it can't capture meaning/synonyms the way MiniLM or an
OpenAI/Claude embedding model can — but it's a real, legitimate embedding
technique and lets you run and understand the full pipeline end-to-end
right now.

When you run this on your own machine with normal internet access, switch
EMBEDDING_BACKEND to "minilm" below (or plug in an API-based embedding
model) for proper semantic search. That upgrade is a single-line change.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from chromadb import EmbeddingFunction, Documents, Embeddings
import numpy as np
import pickle
import os

EMBEDDING_BACKEND = "tfidf"  # switch to "minilm" on a machine with open internet


class TfidfEmbeddingFunction(EmbeddingFunction):
    """A Chroma-compatible embedding function backed by scikit-learn's TF-IDF.

    Chroma expects embeddings to be a fixed-size vector per text. TF-IDF
    vectors are as wide as the vocabulary, which works fine for a small,
    fixed corpus like this one — the vectorizer is fit once on the corpus
    at ingest time and reused (loaded from disk) at query time so the
    vector space stays consistent.
    """

    def __init__(self, vectorizer_path: str = "chroma_db/tfidf_vectorizer.pkl"):
        self.vectorizer_path = vectorizer_path
        self.vectorizer: TfidfVectorizer | None = None
        if os.path.exists(vectorizer_path):
            with open(vectorizer_path, "rb") as f:
                self.vectorizer = pickle.load(f)

    def fit(self, corpus: list[str]):
        """Fit the vectorizer on the full chunk corpus and persist it."""
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=2000)
        self.vectorizer.fit(corpus)
        os.makedirs(os.path.dirname(self.vectorizer_path), exist_ok=True)
        with open(self.vectorizer_path, "wb") as f:
            pickle.dump(self.vectorizer, f)

    def __call__(self, input: Documents) -> Embeddings:
        if self.vectorizer is None:
            raise RuntimeError(
                "Vectorizer not fitted yet. Run ingest.py first, "
                "or call .fit() before embedding queries."
            )
        vectors = self.vectorizer.transform(input)
        # L2-normalize so cosine distance behaves predictably (range 0-2,
        # with near-0 meaning near-identical). Without this, raw TF-IDF
        # vector magnitudes vary a lot by chunk length and distance
        # thresholds become meaningless.
        vectors = normalize(vectors)
        return [np.array(v, dtype=np.float32).tolist() for v in vectors.toarray()]
