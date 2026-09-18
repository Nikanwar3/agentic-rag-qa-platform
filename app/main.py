"""
FastAPI application: ingest a document, ask agentic RAG questions against
it, and inspect SQL-backed history/analytics. Every /query call is also
logged as an MLflow run (see app/mlflow_tracker.py).
"""

import os
import time

from fastapi import FastAPI, HTTPException

from app import mlflow_tracker
from app.agent import answer_question
from app.config import settings
from app.database import (
    get_analytics,
    get_recent_queries,
    init_db,
    log_document,
    log_query,
)
from app.document_processor import (
    chunk_text,
    download_to_tempfile,
    extract_text_from_document,
)
from app.schemas import IngestRequest, IngestResponse, QueryRequest, QueryResponse
from app.vector_store import build_and_persist_index

app = FastAPI(title="Agentic RAG QA Platform")
init_db()


@app.post("/documents/ingest", response_model=IngestResponse)
async def ingest_document(req: IngestRequest):
    if req.source_url:
        tmp_path = download_to_tempfile(req.source_url)
        try:
            text = extract_text_from_document(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        source_type, source_ref = "url", req.source_url
    else:
        text = req.text
        source_type, source_ref = "text", "inline"

    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="No text could be extracted from the document")

    chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        raise HTTPException(status_code=400, detail="No chunks produced from document text")

    document_id = log_document(source_type, source_ref, len(chunks))
    build_and_persist_index(document_id, chunks)

    return IngestResponse(document_id=document_id, num_chunks=len(chunks))


@app.post("/query", response_model=QueryResponse)
async def query_document(req: QueryRequest):
    start = time.time()

    with mlflow_tracker.start_run(run_name=f"query-{req.document_id[:8]}"):
        mlflow_tracker.log_params({
            "document_id": req.document_id,
            "embedding_model": settings.embedding_model,
            "llm_model": settings.llm_model,
            "chunk_size": settings.chunk_size,
            "top_k": req.top_k or settings.top_k,
        })
        try:
            result = await answer_question(req.document_id, req.question, req.top_k)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))

        latency_ms = (time.time() - start) * 1000
        mlflow_tracker.log_metrics({
            "latency_ms": latency_ms,
            "retrieval_attempts": result["retrieval_attempts"],
            "query_rewritten": int(result["query_rewritten"]),
        })

    log_query(
        document_id=req.document_id,
        question=req.question,
        answer=result["answer"],
        retrieval_attempts=result["retrieval_attempts"],
        query_rewritten=result["query_rewritten"],
        latency_ms=latency_ms,
    )

    return QueryResponse(
        answer=result["answer"],
        retrieval_attempts=result["retrieval_attempts"],
        query_rewritten=result["query_rewritten"],
        latency_ms=latency_ms,
    )


@app.get("/history")
async def history(limit: int = 20):
    return {"queries": get_recent_queries(limit=limit)}


@app.get("/analytics")
async def analytics():
    return get_analytics()


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
