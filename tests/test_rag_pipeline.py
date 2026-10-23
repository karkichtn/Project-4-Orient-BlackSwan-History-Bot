"""
Unit tests for end-to-end RAG question answering pipeline using mocked components.
Tests all edge cases: supported questions, unsupported questions, empty queries, missing credentials.
"""

import pytest
from langchain_chroma import Chroma

from ask import ask_question


@pytest.fixture
def mock_chroma_store(sample_history_chunks, fake_embeddings):
    """Creates a local in-memory Chroma instance with sample documents."""
    return Chroma.from_documents(
        documents=sample_history_chunks,
        embedding=fake_embeddings,
        collection_name="test_rag_col",
    )


def test_supported_question_with_mock_llm(mock_chroma_store):
    """Test asking a question with a supported answer returns valid citations."""
    def mock_llm(query, context):
        # The mock LLM reads the context passed to it and cites the passage containing the evidence
        import re
        match = re.search(r"\[Passage (\d+)[^\]]*\]\s*([^\[]*Incorporated in 1948[^\[]*)", context)
        passage_id = int(match.group(1)) if match else 1
        return (
            f'{{"answer": "Orient BlackSwan was incorporated in 1948.", '
            f'"citations": [{{"passage_id": {passage_id}, "quote": "Incorporated in 1948"}}]}}'
        )

    res = ask_question(
        query="When was Orient BlackSwan incorporated?",
        vector_store=mock_chroma_store,
        mock_llm=mock_llm,
    )

    assert res["is_supported"] is True
    assert res["answer"] == "Orient BlackSwan was incorporated in 1948."
    assert len(res["citations"]) == 1
    assert res["citations"][0]["page_number"] == 2
    assert res["citations"][0]["page_label"] == "ii"


def test_unsupported_question_with_mock_llm(mock_chroma_store):
    """Test asking an unsupported question (e.g. Physics) returns exactly 'Not found'."""
    def mock_llm(query, context):
        return '{"answer": "Not found", "citations": []}'

    res = ask_question(
        query="What is the price of a Physics textbook?",
        vector_store=mock_chroma_store,
        mock_llm=mock_llm,
    )

    assert res["is_supported"] is False
    assert res["answer"] == "Not found"
    assert len(res["citations"]) == 0


def test_empty_query_handling(mock_chroma_store):
    """Test that an empty or whitespace-only query returns 'Not found' without calling LLM."""
    res = ask_question(query="   ", vector_store=mock_chroma_store)
    assert res["answer"] == "Not found"
    assert len(res["citations"]) == 0


def test_missing_api_key_raises_informative_error(mock_chroma_store, monkeypatch):
    """Test that missing credentials raise ValueError when calling live LLM."""
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="Missing Gemini API Key"):
        ask_question(
            query="When was Orient BlackSwan incorporated?",
            vector_store=mock_chroma_store,
            api_key="",
            mock_llm=None,
        )
