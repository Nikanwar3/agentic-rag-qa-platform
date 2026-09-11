"""
Point every test at an isolated, throwaway SQLite DB / FAISS index dir /
MLflow store, and make sure GROQ_API_KEY is unset so the LLM client runs in
its deterministic mock mode. This module-level code runs once, before any
`app.*` module is imported by a test file, because pytest always imports
conftest.py ahead of test collection.
"""

import os
import tempfile

_TEST_DIR = tempfile.mkdtemp(prefix="agentic_rag_test_")
os.environ["DATABASE_URL"] = f"sqlite:///{_TEST_DIR}/test_app.db"
os.environ["INDEX_DIR"] = f"{_TEST_DIR}/indexes"
os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{_TEST_DIR}/test_mlflow.db"
os.environ.pop("GROQ_API_KEY", None)
