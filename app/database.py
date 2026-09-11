"""
SQL persistence layer (SQLAlchemy ORM over SQLite by default; swap
DATABASE_URL for Postgres/MySQL without touching any other module).

Three tables:
  - documents        one row per ingested document
  - query_logs       one row per question answered (latency, agent behavior)
  - evaluation_runs  one row per offline evaluation sweep (see scripts/run_evaluation.py)

get_analytics() demonstrates real SQL aggregation (COUNT/AVG/MIN/MAX, GROUP BY)
via the ORM's func.* helpers rather than hand-rolled Python aggregation.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import create_engine, func, String, Integer, Float, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker, relationship

from app.config import settings


class Base(DeclarativeBase):
    pass


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    source_type: Mapped[str] = mapped_column(String)  # "url" | "text" | "file"
    source_ref: Mapped[str] = mapped_column(String)
    num_chunks: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    queries: Mapped[list["QueryLog"]] = relationship(back_populates="document")


class QueryLog(Base):
    __tablename__ = "query_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id"))
    question: Mapped[str] = mapped_column(String)
    answer: Mapped[str] = mapped_column(String)
    retrieval_attempts: Mapped[int] = mapped_column(Integer, default=1)
    query_rewritten: Mapped[bool] = mapped_column(Boolean, default=False)
    latency_ms: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    document: Mapped["Document"] = relationship(back_populates="queries")


class EvaluationRun(Base):
    __tablename__ = "evaluation_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_name: Mapped[str] = mapped_column(String)
    chunk_size: Mapped[int] = mapped_column(Integer)
    top_k: Mapped[int] = mapped_column(Integer)
    embedding_model: Mapped[str] = mapped_column(String)
    hit_rate: Mapped[float] = mapped_column(Float)
    answer_accuracy: Mapped[float] = mapped_column(Float)
    avg_latency_ms: Mapped[float] = mapped_column(Float)
    mlflow_run_id: Mapped[str] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)


engine = create_engine(settings.database_url, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


def init_db() -> None:
    Base.metadata.create_all(engine)


def log_document(source_type: str, source_ref: str, num_chunks: int) -> str:
    with SessionLocal() as db:
        doc = Document(source_type=source_type, source_ref=source_ref, num_chunks=num_chunks)
        db.add(doc)
        db.commit()
        return doc.id


def log_query(document_id: str, question: str, answer: str, retrieval_attempts: int,
              query_rewritten: bool, latency_ms: float) -> None:
    with SessionLocal() as db:
        db.add(QueryLog(
            document_id=document_id,
            question=question,
            answer=answer,
            retrieval_attempts=retrieval_attempts,
            query_rewritten=query_rewritten,
            latency_ms=latency_ms,
        ))
        db.commit()


def log_evaluation_run(run_name: str, chunk_size: int, top_k: int, embedding_model: str,
                        hit_rate: float, answer_accuracy: float, avg_latency_ms: float,
                        mlflow_run_id: str | None = None) -> None:
    with SessionLocal() as db:
        db.add(EvaluationRun(
            run_name=run_name, chunk_size=chunk_size, top_k=top_k,
            embedding_model=embedding_model, hit_rate=hit_rate,
            answer_accuracy=answer_accuracy, avg_latency_ms=avg_latency_ms,
            mlflow_run_id=mlflow_run_id,
        ))
        db.commit()


def get_recent_queries(limit: int = 20) -> list[dict]:
    with SessionLocal() as db:
        rows = (
            db.query(QueryLog)
            .order_by(QueryLog.created_at.desc())
            .limit(limit)
            .all()
        )
        return [
            {
                "id": r.id, "document_id": r.document_id, "question": r.question,
                "answer": r.answer, "retrieval_attempts": r.retrieval_attempts,
                "query_rewritten": r.query_rewritten, "latency_ms": r.latency_ms,
                "created_at": r.created_at.isoformat(),
            }
            for r in rows
        ]


def get_analytics() -> dict:
    """SQL-aggregated stats: overall summary + per-document breakdown."""
    with SessionLocal() as db:
        summary_row = db.query(
            func.count(QueryLog.id).label("total_queries"),
            func.avg(QueryLog.latency_ms).label("avg_latency_ms"),
            func.min(QueryLog.latency_ms).label("min_latency_ms"),
            func.max(QueryLog.latency_ms).label("max_latency_ms"),
            func.avg(QueryLog.retrieval_attempts).label("avg_retrieval_attempts"),
            func.sum(func.cast(QueryLog.query_rewritten, Integer)).label("num_rewrites"),
        ).one()

        by_document = (
            db.query(
                QueryLog.document_id,
                func.count(QueryLog.id).label("num_queries"),
                func.avg(QueryLog.latency_ms).label("avg_latency_ms"),
            )
            .group_by(QueryLog.document_id)
            .order_by(func.count(QueryLog.id).desc())
            .all()
        )

        doc_count = db.query(func.count(Document.id)).scalar()

        return {
            "summary": {
                "total_documents": doc_count,
                "total_queries": summary_row.total_queries or 0,
                "avg_latency_ms": summary_row.avg_latency_ms,
                "min_latency_ms": summary_row.min_latency_ms,
                "max_latency_ms": summary_row.max_latency_ms,
                "avg_retrieval_attempts": summary_row.avg_retrieval_attempts,
                "num_rewrites": summary_row.num_rewrites or 0,
            },
            "by_document": [
                {"document_id": row.document_id, "num_queries": row.num_queries,
                 "avg_latency_ms": row.avg_latency_ms}
                for row in by_document
            ],
        }
