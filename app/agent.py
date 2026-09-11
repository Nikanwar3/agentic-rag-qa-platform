"""
Agentic retrieval loop: retrieve -> grade -> (rewrite & retry | generate).

This is the "corrective RAG" behavior that separates the /query endpoint
from a plain one-shot retrieve-then-generate pipeline — if the LLM judges
the first retrieval irrelevant, it rewrites the query and searches again
(bounded by MAX_AGENT_RETRIES) before answering.
"""

import asyncio

from app import llm_client
from app.config import settings
from app.vector_store import query_top_chunks


async def answer_question(document_id: str, question: str, top_k: int | None = None) -> dict:
    top_k = top_k or settings.top_k
    query = question
    query_rewritten = False
    context = ""
    attempts = 0

    while True:
        attempts += 1
        context = await asyncio.to_thread(query_top_chunks, document_id, query, top_k)
        relevant = await asyncio.to_thread(llm_client.grade_relevance, question, context)
        if relevant or attempts > settings.max_agent_retries:
            break
        new_query = await asyncio.to_thread(llm_client.rewrite_query, question, context)
        if new_query and new_query.strip() and new_query.strip() != query:
            query = new_query.strip()
            query_rewritten = True
        else:
            break  # rewrite didn't produce anything new — stop retrying

    answer = await asyncio.to_thread(llm_client.generate_answer, question, context)

    return {
        "answer": answer,
        "retrieval_attempts": attempts,
        "query_rewritten": query_rewritten,
        "context": context,
    }
