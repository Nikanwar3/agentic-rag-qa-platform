# Agentic RAG QA Platform

A self-correcting, retrieval-augmented document Q&A service with SQL-backed
analytics and MLflow-tracked experiments — built end-to-end in Python.

Ingest any document (PDF, DOCX, or raw text), ask it questions, and get
answers from an **agentic RAG loop**: retrieve → grade relevance → (rewrite
the query and retry, or answer). Every query and every offline evaluation
run is logged to SQL and MLflow, so both live performance and model/config
choices are measurable, not anecdotal.

-----

## Why "agentic"

Most RAG demos do one retrieve + one generate, and answer confidently even
when the retrieved chunk doesn't actually contain the answer. This one
doesn't:

```
question
   │
   ▼
retrieve top-k chunks ──▶ LLM grades: "does this context answer the question?"
   ▲                              │
   │ rewrite query           NO ──┤── YES
   └────────────────────────      ▼
   (capped at MAX_AGENT_RETRIES)  generate final answer
```

`retrieval_attempts` and `query_rewritten` are returned with every answer so
you can see when — and how often — the agent had to self-correct.

-----

## Tech Stack

| Skill | Where |
| --- | --- |
| **Python** | Entire codebase — FastAPI app, agent loop, evaluation harness |
| **SQL** | SQLAlchemy ORM over SQLite (`app/database.py`) — documents, query logs, evaluation runs; real aggregate queries (`COUNT`, `AVG`, `MIN/MAX`, `GROUP BY`) power `/analytics` |
| **LLM** | Groq (Llama3-8B) via the OpenAI-compatible client, with a deterministic mock fallback when no API key is set |
| **RAG** | SentenceTransformers embeddings + a per-document FAISS index (`app/vector_store.py`), no external vector DB required |
| **AI/ML** | Embedding-based semantic retrieval; `scripts/run_evaluation.py` sweeps chunking/top-k configs and scores retrieval hit-rate + answer accuracy against a benchmark QA set |
| **MLflow** | Every `/query` call *and* every evaluation sweep run is logged as an MLflow run (params + metrics), backed by a SQLite tracking store — `app/mlflow_tracker.py` |

-----

## Project Layout

```
agentic-rag-qa-platform/
├── app/
│   ├── main.py                # FastAPI app: /documents/ingest, /query, /history, /analytics, /health
│   ├── config.py               # Env-driven settings (all optional — sane local defaults)
│   ├── database.py              # SQL (SQLAlchemy): schema, writes, aggregate analytics queries
│   ├── schemas.py                 # Pydantic request/response models
│   ├── document_processor.py       # PDF/DOCX/text extraction + sliding-window chunking
│   ├── vector_store.py              # SentenceTransformers embeddings + per-document FAISS index
│   ├── llm_client.py                 # Groq LLM calls + mock-mode fallback (grade/rewrite/generate)
│   ├── agent.py                       # The retrieve → grade → rewrite/retry → generate loop
│   └── mlflow_tracker.py               # MLflow run helpers shared by the API and the eval script
├── scripts/
│   └── run_evaluation.py         # Offline sweep over chunk_size × top_k, logged to MLflow + SQL
├── data/
│   ├── sample_document.txt        # Synthetic HR policy doc used by the evaluation sweep
│   └── eval_qa.json                # Benchmark question set with expected keywords
├── tests/
│   ├── test_pipeline.py            # Unit tests: chunking, vector store, mock LLM, agent loop
│   └── test_api.py                  # FastAPI TestClient integration tests
├── requirements.txt
└── .env.example
```

-----

## Quickstart

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Optional: set GROQ_API_KEY for real LLM answers. Without it, the app runs
# in mock mode — fully functional, deterministic, zero external calls.

uvicorn app.main:app --reload
```

The first request that touches embeddings downloads the
`all-MiniLM-L6-v2` model (~90 MB, one-time, cached locally by
SentenceTransformers).

### Try it

```bash
# 1. Ingest a document
curl -X POST http://localhost:8000/documents/ingest \
  -H "Content-Type: application/json" \
  -d '{"text": "Employees may work remotely up to five days a week."}'
# => {"document_id": "...", "num_chunks": 1}

# 2. Ask it a question
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"document_id": "<id from step 1>", "question": "How many days can employees work remotely?"}'

# 3. Inspect SQL-backed history & analytics
curl http://localhost:8000/history
curl http://localhost:8000/analytics
```

-----

## API

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/documents/ingest` | POST | `{source_url}` or `{text}` → extract, chunk, embed, index. Returns `document_id`. |
| `/query` | POST | `{document_id, question, top_k?}` → agentic RAG answer + `retrieval_attempts` + `query_rewritten`. |
| `/history` | GET | Recent question/answer pairs, straight from `query_logs`. |
| `/analytics` | GET | SQL-aggregated stats: totals, avg/min/max latency, avg retrieval attempts, rewrite rate, per-document breakdown. |
| `/health` | GET | Liveness check. |

Interactive docs (Swagger UI) at `/docs` once the server is running.

-----

## Evaluation & Experiment Tracking (AI/ML + MLflow)

```bash
python scripts/run_evaluation.py
```

This rebuilds the index over `data/sample_document.txt` for four
`(chunk_size, top_k)` configurations, runs all 8 benchmark questions from
`data/eval_qa.json` through the real retrieval + generation pipeline, and
for each config logs to MLflow:

- **params**: `chunk_size`, `top_k`, `embedding_model`, `llm_model`
- **metrics**: `hit_rate` (did retrieval surface the right passage?),
  `answer_accuracy` (did the generated answer contain the expected fact?),
  `avg_latency_ms`, `num_chunks`

Each run is also persisted to the SQL `evaluation_runs` table alongside its
`mlflow_run_id`, so the two systems cross-reference each other. Compare
configs visually:

```bash
mlflow ui --backend-store-uri sqlite:///./data/mlflow.db
```

-----

## Testing

```bash
pytest -v
```

Tests run fully offline against isolated temp SQLite/FAISS/MLflow stores
(see `tests/conftest.py`) with `GROQ_API_KEY` unset, exercising the same
mock-mode code path used in CI and local dev without credentials.

-----

## Design notes

- **Mock-mode LLM fallback** (`app/llm_client.py`): every grading/rewriting/
  generation call degrades to a keyword-overlap heuristic when no
  `GROQ_API_KEY` is set, instead of crashing or requiring a key just to run
  tests. Production traffic with a real key gets the real model
  transparently — same call sites, same return shapes.
- **No external vector DB dependency**: FAISS indexes are built per-document
  and persisted to `data/indexes/`, so the whole RAG pipeline runs locally
  with no Pinecone/Weaviate account needed.
- **SQLite by default, swappable**: `DATABASE_URL` and `MLFLOW_TRACKING_URI`
  both point at SQLite out of the box; pointing either at Postgres is a
  one-line env var change (SQLAlchemy/MLflow both support it natively).
