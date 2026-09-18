import os

from dotenv import load_dotenv

load_dotenv()


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


class Settings:
    """Central, env-driven configuration. Every value has a sensible local
    default so the app runs out of the box with zero required env vars —
    GROQ_API_KEY is the only thing that upgrades it from mock mode to a real
    LLM (see app/llm_client.py)."""

    # --- LLM (Groq, OpenAI-compatible API) ---
    groq_api_key: str | None = os.getenv("GROQ_API_KEY")
    llm_model: str = os.getenv("LLM_MODEL", "llama3-8b-8192")

    # --- Embeddings / RAG ---
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    chunk_size: int = _int("CHUNK_SIZE", 500)
    chunk_overlap: int = _int("CHUNK_OVERLAP", 80)
    top_k: int = _int("TOP_K", 3)
    max_agent_retries: int = _int("MAX_AGENT_RETRIES", 2)

    # --- Storage ---
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./data/app.db")
    index_dir: str = os.getenv("INDEX_DIR", "./data/indexes")

    # --- MLflow ---
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///./data/mlflow.db")
    mlflow_experiment_name: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "agentic-rag-qa")


settings = Settings()
