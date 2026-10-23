"""
=============================================================================
Project 4 — Orient BlackSwan History Bot: Ingestion Pipeline
=============================================================================
This module handles:
1. Loading the Orient BlackSwan History PDF using PyPDFLoader.
2. Preserving accurate metadata (source filename, physical page, and book page label).
3. Splitting the document into overlapping chunks using RecursiveCharacterTextSplitter.
4. Generating embeddings using Google Gemini (gemini-embedding-001).
5. Persisting vectors into a local ChromaDB vector store.
6. Guarding against duplicate indexing and stale chunks via SHA-256 fingerprinting.
=============================================================================
"""

import argparse
import hashlib
import json
import os
import sys
import warnings
from pathlib import Path
from typing import List, Optional

# Suppress library deprecation notices for clean terminal output
warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv

# Load environment variables from .env file (if present)
load_dotenv()


def compute_file_hash(filepath: str) -> str:
    """
    Computes SHA-256 hash of a file to uniquely identify document versions.
    Prevents stale chunks from persisting when a document is updated or replaced.
    """
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192 * 1024):  # 8MB buffer
            hasher.update(chunk)
    return hasher.hexdigest()


def load_pdf_document(pdf_path: str):
    """
    Loads a PDF document using LangChain's PyPDFLoader.
    Validates file existence, readability, and non-empty extracted text.

    Returns:
        List of LangChain Document objects with page metadata preserved.
    """
    from langchain_community.document_loaders import PyPDFLoader

    path_obj = Path(pdf_path)
    if not path_obj.exists():
        raise FileNotFoundError(
            f"PDF file not found at: '{pdf_path}'\n"
            f"Please verify that the file exists in the directory."
        )

    if not path_obj.is_file():
        raise ValueError(f"Path '{pdf_path}' is not a regular file.")

    if path_obj.stat().st_size == 0:
        raise ValueError(f"PDF file '{pdf_path}' is empty (0 bytes).")

    print(f"\n[1/4] Loading PDF: {pdf_path} ({path_obj.stat().st_size / (1024 * 1024):.2f} MB)...")

    try:
        loader = PyPDFLoader(str(path_obj))
        pages = loader.load()
    except Exception as e:
        raise RuntimeError(
            f"Failed to parse PDF '{pdf_path}'. The file may be corrupted, password-protected, or unreadable.\n"
            f"Original error: {e}"
        ) from e

    if not pages:
        raise ValueError(f"No pages could be extracted from '{pdf_path}'.")

    # Check that at least some text content was extracted across the document
    total_extracted_chars = sum(len(page.page_content.strip()) for page in pages)
    if total_extracted_chars == 0:
        raise ValueError(
            f"The PDF '{pdf_path}' contains {len(pages)} pages, but NO text could be extracted.\n"
            f"The document may consist entirely of scanned images without an OCR layer."
        )

    # Standardize metadata:
    # PyPDFLoader gives 0-indexed 'page' and optional 'page_label' (printed page number).
    # We ensure each page has both physical 1-based page number and original printed label.
    for page in pages:
        physical_idx = page.metadata.get("page", 0)
        page.metadata["source"] = str(path_obj.name)
        page.metadata["source_path"] = str(path_obj)
        page.metadata["page_index"] = physical_idx  # 0-indexed
        page.metadata["page_number"] = physical_idx + 1  # 1-indexed physical page
        if "page_label" not in page.metadata:
            page.metadata["page_label"] = str(physical_idx + 1)

    print(f"      Successfully extracted {len(pages)} pages ({total_extracted_chars:,} characters total).")
    return pages


def split_document_pages(
    pages,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
):
    """
    Splits document pages into chunks using RecursiveCharacterTextSplitter.
    Preserves and enriches metadata (filename, page numbers, chunk ID).
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    print(f"\n[2/4] Splitting text (chunk_size={chunk_size}, chunk_overlap={chunk_overlap})...")

    # Filter out empty pages (such as blank cover pages) before splitting
    non_empty_pages = [p for p in pages if len(p.page_content.strip()) > 0]
    print(f"      Pages with readable text: {len(non_empty_pages)} of {len(pages)}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = text_splitter.split_documents(non_empty_pages)

    if not chunks:
        raise ValueError("Text splitting resulted in 0 chunks. Please verify document content.")

    # Assign sequential chunk ID and confirm metadata on each chunk
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = idx
        # Ensure page_number and page_label are always present
        if "page_number" not in chunk.metadata:
            chunk.metadata["page_number"] = chunk.metadata.get("page", 0) + 1

    print(f"      Generated {len(chunks)} chunks across {len(non_empty_pages)} text-bearing pages.")
    return chunks


def get_embeddings(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Initializes local Hugging Face Embeddings (sentence-transformers/all-MiniLM-L6-v2).
    Runs 100% locally without external API rate limits, tokens, or request quotas.
    """
    from langchain_huggingface import HuggingFaceEmbeddings

    resolved_model = os.getenv("EMBEDDING_MODEL") or model_name
    # Fallback to local sentence-transformers if an older Gemini embedding name is present in env
    if "gemini" in resolved_model.lower() or "text-embedding" in resolved_model.lower():
        resolved_model = "sentence-transformers/all-MiniLM-L6-v2"

    return HuggingFaceEmbeddings(model_name=resolved_model)


# Backward-compatible alias
get_gemini_embeddings = get_embeddings


def index_chunks_into_chromadb(
    chunks,
    embeddings,
    chroma_dir: str = "chroma_db",
    collection_name: str = "history_rag",
    pdf_path: str = "data/History_2024-25.pdf",
    doc_hash: str = "",
    force_reindex: bool = False,
):
    """
    Stores document vectors in persistent ChromaDB storage.
    Includes stale document detection, batch indexing, and metadata saving.
    """
    from langchain_chroma import Chroma
    import chromadb

    metadata_file = Path(chroma_dir).resolve() / "ingest_metadata.json"
    chroma_path = Path(chroma_dir).resolve()
    chroma_path.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(chroma_path))
    should_clear = force_reindex

    # Check for existing index
    if metadata_file.exists() and not should_clear:
        try:
            with open(metadata_file, "r") as f:
                saved_meta = json.load(f)
            
            existing_hash = saved_meta.get("file_hash")
            existing_chunks = saved_meta.get("total_chunks", 0)
            existing_file = saved_meta.get("pdf_path")
            existing_model = saved_meta.get("embedding_model", "")

            # If embedding model changed or hash differs, clear collection to prevent dimension mismatch
            current_model_name = getattr(embeddings, "model_name", "sentence-transformers/all-MiniLM-L6-v2")
            if existing_model != current_model_name:
                print(f"\n[NOTICE] Embedding model changed from '{existing_model}' to '{current_model_name}'.")
                print("         Resetting collection to ensure vector dimension compatibility...")
                should_clear = True
            elif existing_hash == doc_hash and not force_reindex:
                print(f"\n[INFO] Document is already indexed in '{chroma_dir}'.")
                print(f"       File: {existing_file}")
                print(f"       Total Chunks: {existing_chunks}")
                print(f"       Model: {existing_model}")
                print(f"       SHA-256: {existing_hash[:12]}...")
                print("       Skipping re-ingestion. Use '--force' to rebuild the index from scratch.")
                return Chroma(
                    client=client,
                    collection_name=collection_name,
                    embedding_function=embeddings,
                )
            elif existing_hash != doc_hash:
                print(f"\n[NOTICE] Detected a changed or replaced PDF (Hash mismatch).")
                print("         Resetting collection to prevent mixed or outdated chunks...")
                should_clear = True
        except Exception as e:
            print(f"[WARNING] Could not read existing ingestion metadata ({e}). Rebuilding...")
            should_clear = True

    if should_clear:
        try:
            client.delete_collection(name=collection_name)
        except Exception:
            pass
        if metadata_file.exists():
            try:
                metadata_file.unlink()
            except Exception:
                pass

    print(f"\n[3/4] Initializing ChromaDB vector store at '{chroma_path}'...")
    print(f"      Collection: '{collection_name}' | Total chunks to embed: {len(chunks)}")
    print(f"      Computing local HuggingFace embeddings ({getattr(embeddings, 'model_name', 'sentence-transformers/all-MiniLM-L6-v2')})...")

    # Ingest in batches of 100
    BATCH_SIZE = 100
    total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE

    vector_store = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embeddings,
    )

    try:
        for b_idx in range(total_batches):
            start = b_idx * BATCH_SIZE
            end = min(start + BATCH_SIZE, len(chunks))
            batch = chunks[start:end]
            print(f"      Embedding batch {b_idx + 1}/{total_batches} (chunks {start + 1}-{end})...")
            vector_store.add_documents(batch)
    except Exception as e:
        raise RuntimeError(
            f"Embedding/ChromaDB indexing failed on batch {b_idx + 1}: {e}"
        ) from e

    # Save ingestion metadata to track file hash and prevent stale retrieval
    meta_data = {
        "pdf_path": pdf_path,
        "filename": Path(pdf_path).name,
        "file_hash": doc_hash,
        "file_size_bytes": Path(pdf_path).stat().st_size,
        "total_chunks": len(chunks),
        "collection_name": collection_name,
        "embedding_model": getattr(embeddings, "model_name", "sentence-transformers/all-MiniLM-L6-v2"),
    }
    with open(metadata_file, "w") as f:
        json.dump(meta_data, f, indent=2)

    print(f"\n[4/4] Ingestion completed successfully!")
    print(f"      Indexed {len(chunks)} chunks into '{chroma_path}'.")
    print(f"      Saved document fingerprint ({doc_hash[:12]}...) to '{metadata_file}'.")
    return vector_store


def run_pipeline(
    pdf_path: str = "data/History_2024-25.pdf",
    chroma_dir: str = "chroma_db",
    collection_name: str = "history_rag",
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    force_reindex: bool = False,
    dry_run: bool = False,
):
    """
    Main orchestration function for the ingestion pipeline.
    """
    print("=" * 70)
    print(" Project 4 — Orient BlackSwan History Bot: Ingestion")
    print("=" * 70)

    # 1. Compute Document Hash
    doc_hash = compute_file_hash(pdf_path)
    print(f"Target Document: {pdf_path}")
    print(f"SHA-256 Hash:   {doc_hash}")

    # 2. Extract Document Text
    pages = load_pdf_document(pdf_path)

    # 3. Chunk Document
    chunks = split_document_pages(
        pages=pages,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # Print chunk sample and metadata verification
    print("\n--- Metadata Sample (First Chunk) ---")
    print(f"Source:      {chunks[0].metadata.get('source')}")
    print(f"Physical Pg: {chunks[0].metadata.get('page_number')}")
    print(f"Book Label:  {chunks[0].metadata.get('page_label')}")
    print(f"Snippet:     {chunks[0].page_content[:150].replace(chr(10), ' ')}...")
    print("-------------------------------------")

    if dry_run:
        print("\n[DRY RUN COMPLETE]")
        print("PDF extraction, page numbering, and text chunking verified successfully.")
        print("No calls to Gemini API were made.")
        return None

    # 4. Initialize Gemini Embeddings
    embeddings = get_gemini_embeddings(model_name=embedding_model)

    # 5. Index into ChromaDB
    vector_store = index_chunks_into_chromadb(
        chunks=chunks,
        embeddings=embeddings,
        chroma_dir=chroma_dir,
        collection_name=collection_name,
        pdf_path=pdf_path,
        doc_hash=doc_hash,
        force_reindex=force_reindex,
    )
    return vector_store


def main():
    parser = argparse.ArgumentParser(
        description="Ingest Orient BlackSwan History PDF into ChromaDB with Gemini Embeddings."
    )
    parser.add_argument(
        "--pdf",
        type=str,
        default=os.getenv("PDF_PATH", "data/History_2024-25.pdf"),
        help="Path to the PDF file (default: data/History_2024-25.pdf)",
    )
    parser.add_argument(
        "--chroma-dir",
        type=str,
        default=os.getenv("CHROMA_DIR", "chroma_db"),
        help="Directory to store persistent ChromaDB vectors (default: chroma_db)",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=os.getenv("COLLECTION_NAME", "history_rag"),
        help="ChromaDB collection name (default: history_rag)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=int(os.getenv("CHUNK_SIZE", "800")),
        help="Character size for each text chunk (default: 800)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=int(os.getenv("CHUNK_OVERLAP", "120")),
        help="Overlap characters between adjacent chunks (default: 120)",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        help="Embedding model (default: sentence-transformers/all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-indexing even if document was already ingested.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Extract and chunk without calling Gemini Embeddings or ChromaDB.",
    )

    args = parser.parse_args()

    try:
        run_pipeline(
            pdf_path=args.pdf,
            chroma_dir=args.chroma_dir,
            collection_name=args.collection,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            embedding_model=args.embedding_model,
            force_reindex=args.force,
            dry_run=args.dry_run,
        )
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
