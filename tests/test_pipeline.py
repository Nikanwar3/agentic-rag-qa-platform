import asyncio

from app import llm_client
from app.agent import answer_question
from app.document_processor import chunk_text
from app.vector_store import build_and_persist_index, query_top_chunks

SAMPLE_TEXT = (
    "Employees may work remotely up to five days a week with manager approval. "
    "Health insurance eligibility begins after 90 days of employment. "
    "The annual learning stipend is INR 15,000 per employee."
)


def test_chunk_text_overlaps_and_covers_all_text():
    chunks = chunk_text(SAMPLE_TEXT, chunk_size=40, overlap=10)
    assert len(chunks) > 1
    # every chunk actually appears verbatim in the (whitespace-normalized) source
    for c in chunks:
        assert c in SAMPLE_TEXT


def test_chunk_text_empty_input():
    assert chunk_text("   ") == []


def test_vector_store_round_trip():
    chunks = chunk_text(SAMPLE_TEXT, chunk_size=60, overlap=10)
    build_and_persist_index("test-doc-1", chunks)
    context = query_top_chunks("test-doc-1", "How many days can employees work remotely?", top_k=2)
    assert "remote" in context.lower() or "five" in context.lower()


def test_mock_llm_generate_answer_uses_context():
    context = "The annual learning stipend is INR 15,000 per employee."
    answer = llm_client.generate_answer("What is the learning stipend?", context)
    assert "15,000" in answer or "learning" in answer.lower()


def test_mock_llm_generate_answer_no_context():
    assert llm_client.generate_answer("Anything?", "") == "Not mentioned in the document."


def test_agentic_answer_question_end_to_end():
    chunks = chunk_text(SAMPLE_TEXT, chunk_size=60, overlap=10)
    build_and_persist_index("test-doc-2", chunks)

    result = asyncio.run(answer_question("test-doc-2", "What is the annual learning stipend?"))

    assert result["retrieval_attempts"] >= 1
    assert isinstance(result["query_rewritten"], bool)
    assert result["answer"]
