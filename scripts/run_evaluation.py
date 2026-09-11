"""
Offline evaluation sweep: for each (chunk_size, top_k) config, rebuild the
index over data/sample_document.txt, run every question in data/eval_qa.json
through the real retrieval + generation pipeline, and log the results to
MLflow (one run per config) and to the SQL evaluation_runs table.

Run with:
    python scripts/run_evaluation.py

Then compare configs with:
    mlflow ui --backend-store-uri sqlite:///./data/mlflow.db
"""

import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import llm_client, mlflow_tracker  # noqa: E402
from app.config import settings  # noqa: E402
from app.database import init_db, log_evaluation_run  # noqa: E402
from app.document_processor import chunk_text  # noqa: E402
from app.vector_store import build_and_persist_index, query_top_chunks  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

CONFIGS = [
    {"chunk_size": 300, "top_k": 2},
    {"chunk_size": 500, "top_k": 3},
    {"chunk_size": 800, "top_k": 3},
    {"chunk_size": 500, "top_k": 5},
]


def load_eval_set() -> list[dict]:
    with open(os.path.join(DATA_DIR, "eval_qa.json"), encoding="utf-8") as f:
        return json.load(f)


def load_sample_text() -> str:
    with open(os.path.join(DATA_DIR, "sample_document.txt"), encoding="utf-8") as f:
        return f.read()


def contains_any(text: str, keywords: list[str]) -> bool:
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in keywords)


def evaluate_config(config: dict, text: str, eval_set: list[dict]) -> dict:
    document_id = f"eval-{config['chunk_size']}-{config['top_k']}"
    chunks = chunk_text(text, chunk_size=config["chunk_size"], overlap=settings.chunk_overlap)
    build_and_persist_index(document_id, chunks)

    hits = correct = 0
    latencies = []
    for item in eval_set:
        start = time.time()
        context = query_top_chunks(document_id, item["question"], top_k=config["top_k"])
        answer = llm_client.generate_answer(item["question"], context)
        latencies.append((time.time() - start) * 1000)

        if contains_any(context, item["expected_keywords"]):
            hits += 1
        if contains_any(answer, item["expected_keywords"]):
            correct += 1

    n = len(eval_set)
    return {
        "hit_rate": hits / n,
        "answer_accuracy": correct / n,
        "avg_latency_ms": statistics.mean(latencies),
        "num_chunks": len(chunks),
    }


def main() -> None:
    init_db()
    text = load_sample_text()
    eval_set = load_eval_set()

    print(f"Evaluating {len(CONFIGS)} configs over {len(eval_set)} questions...")
    if not settings.groq_api_key:
        print("NOTE: GROQ_API_KEY not set — running in mock-LLM mode (heuristic answers).\n")

    results = []
    for config in CONFIGS:
        run_name = f"eval-chunk{config['chunk_size']}-top{config['top_k']}"
        with mlflow_tracker.start_run(run_name=run_name):
            mlflow_tracker.log_params({
                "chunk_size": config["chunk_size"],
                "top_k": config["top_k"],
                "embedding_model": settings.embedding_model,
                "llm_model": settings.llm_model,
            })
            metrics = evaluate_config(config, text, eval_set)
            mlflow_tracker.log_metrics(metrics)
            mlflow_run_id = mlflow_tracker.active_run_id()

        log_evaluation_run(
            run_name=run_name,
            chunk_size=config["chunk_size"],
            top_k=config["top_k"],
            embedding_model=settings.embedding_model,
            hit_rate=metrics["hit_rate"],
            answer_accuracy=metrics["answer_accuracy"],
            avg_latency_ms=metrics["avg_latency_ms"],
            mlflow_run_id=mlflow_run_id,
        )
        results.append({**config, **metrics, "run_name": run_name})
        print(
            f"  {run_name}: hit_rate={metrics['hit_rate']:.2f} "
            f"answer_accuracy={metrics['answer_accuracy']:.2f} "
            f"avg_latency_ms={metrics['avg_latency_ms']:.1f}"
        )

    best = max(results, key=lambda r: (r["answer_accuracy"], r["hit_rate"]))
    print(
        f"\nBest config: chunk_size={best['chunk_size']} top_k={best['top_k']} "
        f"(answer_accuracy={best['answer_accuracy']:.2f}, hit_rate={best['hit_rate']:.2f})"
    )
    print(f"View full comparison with: mlflow ui --backend-store-uri {settings.mlflow_tracking_uri}")


if __name__ == "__main__":
    main()
