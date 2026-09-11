"""
LLM client (Groq, via the OpenAI-compatible API) with a deterministic mock
fallback when no GROQ_API_KEY is configured. This keeps tests, the
evaluation script, and local demos runnable with zero API keys, while
production use gets the real model.
"""

import re

from app.config import settings

_client = None


def _get_client():
    global _client
    if _client is None and settings.groq_api_key:
        from openai import OpenAI
        _client = OpenAI(api_key=settings.groq_api_key, base_url="https://api.groq.com/openai/v1")
    return _client


def _chat(prompt: str, max_tokens: int = 150, temperature: float = 0.0) -> str | None:
    client = _get_client()
    if client is None:
        return None  # signals "use mock mode" to callers
    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (response.choices[0].message.content or "").strip()


def _keywords(text: str) -> set[str]:
    stop = {"the", "a", "an", "is", "are", "what", "how", "does", "do", "of", "for", "to", "in", "and"}
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in stop and len(w) > 2}


# ---- mock-mode heuristics (used only when no GROQ_API_KEY is set) ----

def _mock_grade_relevance(question: str, context: str) -> bool:
    if not context.strip():
        return False
    overlap = _keywords(question) & _keywords(context)
    return len(overlap) >= 1


def _mock_rewrite_query(question: str) -> str:
    return f"{question} details specifics numbers"


def _mock_generate_answer(question: str, context: str) -> str:
    if not context.strip():
        return "Not mentioned in the document."
    q_words = _keywords(question)
    sentences = re.split(r"(?<=[.!?])\s+", context)
    best = max(sentences, key=lambda s: len(_keywords(s) & q_words), default="")
    if best and (_keywords(best) & q_words):
        return best.strip()
    return "Not mentioned in the document."


# ---- public API used by app/agent.py ----

def grade_relevance(question: str, context: str) -> bool:
    """Does the retrieved context actually contain an answer to the question?"""
    if not context.strip():
        return False
    result = _chat(
        f"Context:\n{context}\n\nQuestion: {question}\n\n"
        "Does the context contain enough information to answer the question? "
        "Reply with exactly one word: YES or NO.",
        max_tokens=3,
    )
    if result is None:
        return _mock_grade_relevance(question, context)
    return result.strip().upper().startswith("Y")


def rewrite_query(question: str, context: str) -> str:
    """Reformulate the question into a better retrieval query, given that the
    first retrieval attempt came back irrelevant."""
    result = _chat(
        f"The following retrieved context did not answer the question.\n\n"
        f"Context:\n{context}\n\nOriginal question: {question}\n\n"
        "Rewrite the question as a short, keyword-rich search query likely to "
        "retrieve the relevant passage. Reply with only the rewritten query.",
        max_tokens=40,
    )
    if result is None:
        return _mock_rewrite_query(question)
    return result.strip()


def generate_answer(question: str, context: str) -> str:
    """Answer the question using only the retrieved context."""
    if not context.strip():
        return "Not mentioned in the document."
    result = _chat(
        f"Answer the question using only the context below. If the answer "
        f"isn't in the context, reply exactly: \"Not mentioned in the document.\"\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\n\n"
        "Answer in 1-2 concise sentences, under 40 words.",
        max_tokens=100,
    )
    if result is None:
        return _mock_generate_answer(question, context)
    return result
