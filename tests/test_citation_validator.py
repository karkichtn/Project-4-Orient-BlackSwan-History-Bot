"""
Unit tests for citation validation, hallucination detection, and strict "Not found" enforcement.
"""

from ask import parse_and_validate_response


def test_valid_response_with_correct_citation(sample_history_chunks):
    """Test well-formed model JSON with valid passage ID and excerpt."""
    raw_json = """
    {
      "answer": "Orient BlackSwan was incorporated in 1948.",
      "citations": [
        {
          "passage_id": 1,
          "quote": "Incorporated in 1948, the consistent emphasis of our publishing programme has been on quality."
        }
      ]
    }
    """
    res = parse_and_validate_response(raw_json, sample_history_chunks)
    assert res["is_supported"] is True
    assert res["answer"] == "Orient BlackSwan was incorporated in 1948."
    assert len(res["citations"]) == 1
    assert res["citations"][0]["page_number"] == 2
    assert res["citations"][0]["page_label"] == "ii"
    assert res["citations"][0]["source"] == "History_2024-25.pdf"


def test_hallucinated_passage_id_rejection(sample_history_chunks):
    """Test that citing an out-of-bounds passage ID (e.g. 99) is rejected."""
    raw_json = """
    {
      "answer": "This is a hallucinated answer.",
      "citations": [
        {
          "passage_id": 99,
          "quote": "Some fake quote not in chunks."
        }
      ]
    }
    """
    res = parse_and_validate_response(raw_json, sample_history_chunks)
    # Because citations were completely fabricated, the answer must fall back to "Not found"
    assert res["is_supported"] is False
    assert res["answer"] == "Not found"
    assert len(res["citations"]) == 0


def test_explicit_not_found_handling(sample_history_chunks):
    """Test that model outputting Not found returns exactly 'Not found' with 0 citations."""
    variations = [
        '{"answer": "Not found", "citations": []}',
        '{"answer": "Not found.", "citations": []}',
        '{"answer": "Not found in document", "citations": []}',
        'Not found',
        'Not found.',
    ]
    for v in variations:
        res = parse_and_validate_response(v, sample_history_chunks)
        assert res["answer"] == "Not found"
        assert len(res["citations"]) == 0
        assert res["is_supported"] is False


def test_malformed_json_fallback(sample_history_chunks):
    """Test that invalid/unparseable JSON falls back safely to 'Not found'."""
    res = parse_and_validate_response("I cannot answer this question based on the text.", sample_history_chunks)
    assert res["answer"] == "Not found"
    assert len(res["citations"]) == 0
