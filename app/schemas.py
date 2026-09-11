from pydantic import BaseModel, model_validator


class IngestRequest(BaseModel):
    source_url: str | None = None
    text: str | None = None

    @model_validator(mode="after")
    def _one_source_required(self):
        if not self.source_url and not self.text:
            raise ValueError("Provide either 'source_url' or 'text'")
        return self


class IngestResponse(BaseModel):
    document_id: str
    num_chunks: int


class QueryRequest(BaseModel):
    document_id: str
    question: str
    top_k: int | None = None


class QueryResponse(BaseModel):
    answer: str
    retrieval_attempts: int
    query_rewritten: bool
    latency_ms: float
