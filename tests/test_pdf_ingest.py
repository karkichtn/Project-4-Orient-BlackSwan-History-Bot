"""
Unit tests for PDF loading, hashing, and chunking in ingest.py.
All tests run offline without consuming API tokens.
"""

import tempfile
from pathlib import Path
import pytest

from ingest import compute_file_hash, load_pdf_document, split_document_pages


def test_hash_calculation():
    """Verify compute_file_hash is deterministic and detects content changes."""
    with tempfile.NamedTemporaryFile("w+", delete=False) as f1, tempfile.NamedTemporaryFile("w+", delete=False) as f2:
        f1.write("Hello World")
        f1.flush()
        f2.write("Hello World Different")
        f2.flush()

        h1 = compute_file_hash(f1.name)
        h1_repeat = compute_file_hash(f1.name)
        h2 = compute_file_hash(f2.name)

        assert h1 == h1_repeat
        assert h1 != h2
        assert len(h1) == 64  # SHA-256 hex string length


def test_load_nonexistent_pdf_raises_error():
    """Verify loading a missing file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_pdf_document("data/non_existent_file_12345.pdf")


def test_load_empty_pdf_raises_error(tmp_path):
    """Verify loading an empty (0 bytes) file raises ValueError."""
    empty_file = tmp_path / "empty.pdf"
    empty_file.write_bytes(b"")
    with pytest.raises(ValueError, match="is empty"):
        load_pdf_document(str(empty_file))


def test_load_actual_history_pdf():
    """Verify loading the actual Orient BlackSwan catalogue PDF."""
    pdf_path = "data/History_2024-25.pdf"
    if not Path(pdf_path).exists():
        pytest.skip(f"Test PDF '{pdf_path}' not found in workspace.")

    pages = load_pdf_document(pdf_path)
    assert len(pages) == 91, f"Expected exactly 91 pages, got {len(pages)}"

    # Check page labels and 1-based page numbers
    page_1 = pages[1]
    assert page_1.metadata["page_number"] == 2
    assert page_1.metadata["page_label"] == "ii"

    page_3 = pages[3]
    assert page_3.metadata["page_number"] == 4
    assert page_3.metadata["page_label"] == "1"


def test_split_document_pages_chunk_size_and_overlap(sample_history_chunks):
    """Verify text splitting produces expected chunk size, overlap, and preserves metadata."""
    chunks = split_document_pages(sample_history_chunks, chunk_size=100, chunk_overlap=20)
    assert len(chunks) > len(sample_history_chunks)

    for chunk in chunks:
        assert len(chunk.page_content) <= 150  # roughly within chunk size boundary
        assert "page_number" in chunk.metadata
        assert "page_label" in chunk.metadata
        assert "source" in chunk.metadata
