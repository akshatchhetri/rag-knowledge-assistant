"""
api.py — FastAPI wrapper around the RAG pipeline.

Run: uvicorn api:app --reload
Then open http://127.0.0.1:8000/docs for an interactive UI to test it.
"""

from fastapi import FastAPI
from pydantic import BaseModel
from rag_chain import answer_question

app = FastAPI(title="RAG Project Assistant")


class Question(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(payload: Question):
    """Ask a question grounded in the ingested document knowledge base."""
    result = answer_question(payload.question)
    return result
