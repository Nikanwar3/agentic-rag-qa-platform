"""
Local RAG vector store: SentenceTransformers embeddings + a per-document
FAISS index, persisted to disk so it survives an app restart (no external
vector DB / API key required to run this project).
"""

import json
import os

import faiss
import numpy as np

from app.config import settings

os.makedirs(settings.index_dir, exist_ok=True)

_model = None


def _get_model():
    """Lazy-loaded singleton — avoids paying the model-load cost on import
    (e.g. when just running unit tests that don't touch embeddings)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(settings.embedding_model)
    return _model


def _paths(document_id: str) -> tuple[str, str]:
    base = os.path.join(settings.index_dir, document_id)
    return f"{base}.faiss", f"{base}.chunks.json"


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    return vectors / norms


def build_and_persist_index(document_id: str, chunks: list[str]) -> None:
    model = _get_model()
    embeddings = _normalize(np.asarray(model.encode(chunks), dtype="float32"))

    index = faiss.IndexFlatIP(embeddings.shape[1])  # cosine similarity via normalized inner product
    index.add(embeddings)

    index_path, chunks_path = _paths(document_id)
    faiss.write_index(index, index_path)
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f)


def query_top_chunks(document_id: str, query: str, top_k: int = 3) -> str:
    index_path, chunks_path = _paths(document_id)
    if not os.path.exists(index_path):
        raise ValueError(f"No index found for document_id={document_id!r}. Ingest it first.")

    index = faiss.read_index(index_path)
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    model = _get_model()
    query_vec = _normalize(np.asarray(model.encode([query]), dtype="float32"))
    top_k = min(top_k, len(chunks))
    _, indices = index.search(query_vec, top_k)

    return "\n\n".join(chunks[i] for i in indices[0] if 0 <= i < len(chunks))
