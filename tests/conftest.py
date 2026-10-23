"""
Pytest fixtures and configuration for Orient BlackSwan History Bot test suite.
"""

import os
import sys
from pathlib import Path
import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

# Ensure project root is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


@pytest.fixture
def fake_embeddings():
    """Provides a deterministic fake embeddings instance."""
    return DeterministicFakeEmbedding(size=768)


@pytest.fixture
def sample_history_chunks():
    """Sample chunk documents mimicking the Orient BlackSwan catalogue."""
    return [
        Document(
            page_content=(
                "Orient BlackSwan is one of India’s best known and most respected publishing houses. "
                "Incorporated in 1948, the consistent emphasis of our publishing programme has been on quality."
            ),
            metadata={
                "source": "History_2024-25.pdf",
                "page": 1,
                "page_number": 2,
                "page_label": "ii",
                "chunk_index": 0,
            },
        ),
        Document(
            page_content=(
                "Prices updated on 20-01-2025. These prices are subject to change without notice. "
                "Visit our website www.orientblackswan.com for the complete catalog of history publications."
            ),
            metadata={
                "source": "History_2024-25.pdf",
                "page": 2,
                "page_number": 3,
                "page_label": "iii",
                "chunk_index": 1,
            },
        ),
        Document(
            page_content=(
                "Critical History of Women’s Health in Modern Sri Lanka, A series: new perspectives in south asian history. "
                "Darshi Thoradeniya, Senior Lecturer, Department of History, University of Colombo."
            ),
            metadata={
                "source": "History_2024-25.pdf",
                "page": 3,
                "page_number": 4,
                "page_label": "1",
                "chunk_index": 2,
            },
        ),
    ]
