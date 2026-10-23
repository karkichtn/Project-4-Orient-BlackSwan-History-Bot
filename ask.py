"""
=============================================================================
Project 4 — Orient BlackSwan History Bot: Question Answering & Citations
=============================================================================
This module handles:
1. Connecting to the persistent ChromaDB vector store.
2. Retrieving top-K relevant chunks for a user query.
3. Constructing a strictly grounded prompt with strict negative constraints.
4. Invoking Google Gemini (ChatGoogleGenerativeAI) with low temperature (0.0).
5. Enforcing the strict fallback rule: return EXACTLY "Not found" when unsupported.
6. Validating citation passage IDs and quotations against actually retrieved chunks.
=============================================================================
"""

import argparse
import json
import os
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Suppress deprecation notices
warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv

load_dotenv()


STRICT_SYSTEM_PROMPT = """You are the Orient BlackSwan History Bot, an academic research assistant that answers questions STRICTLY and EXCLUSIVELY based on the provided PDF catalogue excerpts.

CRITICAL RULES:
1. ABSOLUTE GROUNDING: Answer ONLY using facts directly stated in the excerpts below. NEVER use general knowledge, internet information, assumptions, or extrapolation.
2. NEGATIVE CONSTRAINT: If the excerpts do not contain complete, clear, and direct evidence to answer the question, you MUST return:
   {"answer": "Not found", "citations": []}
3. EXACT FALLBACK TEXT: The fallback answer must be EXACTLY "Not found" with no trailing punctuation, no extra words, and no explanation.
4. PROHIBITED SPECULATION: If a question asks for subjective opinions (e.g. "best book for beginners") or facts not explicitly stated in the excerpts, output "Not found".
5. CITATIONS: If and only if the answer is supported by the excerpts:
   - Provide the answer in clear, academic prose.
   - For every supporting fact, provide a citation with the integer "passage_id" (e.g. 1, 2) corresponding to the passage heading [Passage X], and a short direct "quote" copied verbatim from that passage.
   - Do NOT invent or fabricate passage IDs. Only cite passages that actually contain the evidence.

YOUR RESPONSE MUST BE VALID JSON IN THIS EXACT FORMAT:
{
  "answer": "<grounded factual answer OR 'Not found'>",
  "citations": [
    {
      "passage_id": 1,
      "quote": "<exact verbatim sentence from passage 1>"
    }
  ]
}
"""


def get_vector_store(
    chroma_dir: Optional[str] = None,
    collection_name: Optional[str] = None,
    embedding_model: Optional[str] = None,
    api_key: Optional[str] = None,
):
    """
    Connects to the persistent ChromaDB database using local Hugging Face embeddings.
    """
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    import chromadb

    resolved_dir = Path(chroma_dir or os.getenv("CHROMA_DIR", "chroma_db")).resolve()
    resolved_col = collection_name or os.getenv("COLLECTION_NAME", "history_rag")
    resolved_model = embedding_model or os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    if "gemini" in resolved_model.lower() or "text-embedding" in resolved_model.lower():
        resolved_model = "sentence-transformers/all-MiniLM-L6-v2"

    if not resolved_dir.exists():
        raise FileNotFoundError(
            f"ChromaDB directory '{resolved_dir}' does not exist.\n"
            f"Please run 'python ingest.py' first to index the PDF document."
        )

    embeddings = HuggingFaceEmbeddings(model_name=resolved_model)
    client = chromadb.PersistentClient(path=str(resolved_dir))

    vector_store = Chroma(
        client=client,
        collection_name=resolved_col,
        embedding_function=embeddings,
    )
    return vector_store


def retrieve_relevant_chunks(
    vector_store,
    query: str,
    top_k: int = 5,
) -> List[Any]:
    """
    Performs similarity search in ChromaDB to retrieve top-k passages.
    """
    if not query or not query.strip():
        return []
    
    # Retrieve top_k documents
    results = vector_store.similarity_search(query.strip(), k=top_k)
    return results


def format_context_passages(chunks: List[Any]) -> str:
    """
    Formats retrieved chunks into clear, numbered passages for model grounding.
    """
    if not chunks:
        return "No relevant passages were found in the document index."

    formatted = []
    for idx, chunk in enumerate(chunks, start=1):
        page_num = chunk.metadata.get("page_number", chunk.metadata.get("page", 0) + 1)
        page_lbl = chunk.metadata.get("page_label", str(page_num))
        source = chunk.metadata.get("source", "document.pdf")
        
        header = f"[Passage {idx} | Source: {source} | Physical Page: {page_num} | Book Label: {page_lbl}]"
        body = chunk.page_content.strip()
        formatted.append(f"{header}\n{body}")

    return "\n\n".join(formatted)


def parse_and_validate_response(
    raw_response_text: str,
    retrieved_chunks: List[Any],
) -> Dict[str, Any]:
    """
    Parses LLM response JSON and validates all citations against actual retrieved chunks.
    Ensures that fabricated citations or out-of-scope answers fall back to 'Not found'.
    """
    clean_text = raw_response_text.strip()
    
    # Remove markdown code block fences if present
    if clean_text.startswith("```"):
        clean_text = re.sub(r"^```(?:json)?\s*", "", clean_text)
        clean_text = re.sub(r"\s*```$", "", clean_text)
        clean_text = clean_text.strip()

    parsed = None
    try:
        parsed = json.loads(clean_text)
    except Exception:
        # If model failed to output pure JSON, inspect text content
        if "not found" in clean_text.lower():
            return {
                "answer": "Not found",
                "citations": [],
                "is_supported": False,
                "raw_response": raw_response_text,
            }
        # Attempt regex search for JSON object
        json_match = re.search(r"\{.*\}", clean_text, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(0))
            except Exception:
                pass

    if not parsed or not isinstance(parsed, dict):
        # Fallback to strict "Not found" if response cannot be parsed
        return {
            "answer": "Not found",
            "citations": [],
            "is_supported": False,
            "raw_response": raw_response_text,
        }

    raw_answer = str(parsed.get("answer", "")).strip()
    
    # Normalize "Not found" variations to exact string
    if (
        not raw_answer
        or raw_answer.lower() in ("not found", "not found.", "not found!", "not found in document", "none")
        or raw_answer.lower().startswith("not found")
    ):
        return {
            "answer": "Not found",
            "citations": [],
            "is_supported": False,
            "raw_response": raw_response_text,
        }

    # Validate citations against actually retrieved chunks
    raw_citations = parsed.get("citations", [])
    validated_citations = []

    for cit in raw_citations:
        if not isinstance(cit, dict):
            continue
        passage_id = cit.get("passage_id")
        quote = str(cit.get("quote", "")).strip()

        # Validate passage ID range (1 to len(retrieved_chunks))
        if isinstance(passage_id, int) and 1 <= passage_id <= len(retrieved_chunks):
            source_chunk = retrieved_chunks[passage_id - 1]
            meta = source_chunk.metadata
            chunk_text = source_chunk.page_content

            # Check if quote actually comes from this chunk
            # Allow minor whitespace/case normalization
            quote_clean = " ".join(quote.split())
            chunk_clean = " ".join(chunk_text.split())

            quote_found = False
            if quote_clean and quote_clean.lower() in chunk_clean.lower():
                quote_found = True
            elif len(quote_clean) > 20:
                # Substring fuzzy check
                words = quote_clean.split()
                if len(words) >= 4:
                    subphrase = " ".join(words[:4])
                    if subphrase.lower() in chunk_clean.lower():
                        quote_found = True

            # If quote wasn't verbatim, use a representative slice from the actual chunk
            snippet_to_display = quote if quote_found else chunk_text[:250].replace("\n", " ").strip() + "..."

            validated_citations.append({
                "passage_id": passage_id,
                "source": meta.get("source", "History_2024-25.pdf"),
                "page_number": meta.get("page_number", meta.get("page", 0) + 1),
                "page_label": meta.get("page_label", str(meta.get("page_number", 1))),
                "snippet": snippet_to_display,
                "quote_matched": quote_found,
            })

    # If the answer is non-trivial but claims citations that were entirely invalid/fabricated:
    # We maintain strict integrity: an answer claiming citations must have at least one validated source.
    if raw_citations and not validated_citations:
        return {
            "answer": "Not found",
            "citations": [],
            "is_supported": False,
            "raw_response": raw_response_text,
        }

    return {
        "answer": raw_answer,
        "citations": validated_citations,
        "is_supported": True,
        "raw_response": raw_response_text,
    }


def ask_question(
    query: str,
    vector_store: Optional[Any] = None,
    chroma_dir: Optional[str] = None,
    collection_name: Optional[str] = None,
    embedding_model: Optional[str] = None,
    llm_model: Optional[str] = None,
    temperature: float = 0.0,
    top_k: int = 5,
    api_key: Optional[str] = None,
    mock_llm: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Main Question Answering pipeline:
    Query -> Retrieve -> Prompt Gemini -> Parse & Validate Citations.
    """
    resolved_query = query.strip() if query else ""
    if not resolved_query:
        return {
            "query": query,
            "answer": "Not found",
            "citations": [],
            "retrieved_chunks": [],
            "is_supported": False,
        }

    resolved_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    resolved_llm_name = llm_model or os.getenv("LLM_MODEL", "gemini-2.5-flash")

    # Connect to vector store if not already provided
    if vector_store is None:
        vector_store = get_vector_store(
            chroma_dir=chroma_dir,
            collection_name=collection_name,
            embedding_model=embedding_model,
            api_key=resolved_key,
        )

    # 1. Retrieve top-k chunks
    retrieved_chunks = retrieve_relevant_chunks(vector_store, resolved_query, top_k=top_k)

    if not retrieved_chunks:
        return {
            "query": resolved_query,
            "answer": "Not found",
            "citations": [],
            "retrieved_chunks": [],
            "is_supported": False,
        }

    # 2. Format Context
    context_text = format_context_passages(retrieved_chunks)

    # 3. Invoke LLM (or mock LLM for testing)
    if mock_llm is not None:
        raw_response = mock_llm(resolved_query, context_text)
    else:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.messages import HumanMessage, SystemMessage

        if not resolved_key or resolved_key.strip() in ("", "your_gemini_api_key_here"):
            raise ValueError(
                "Missing Gemini API Key! Please configure GOOGLE_API_KEY in your .env file."
            )

        llm = ChatGoogleGenerativeAI(
            model=resolved_llm_name,
            temperature=temperature,
            google_api_key=resolved_key,
        )

        user_content = f"CONTEXT EXCERPTS:\n{context_text}\n\nUSER QUESTION: {resolved_query}\n\nREMINDER: Output JSON with 'answer' and 'citations'. If evidence is insufficient, output EXACTLY 'Not found'."
        
        messages = [
            SystemMessage(content=STRICT_SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ]
        response = llm.invoke(messages)
        raw_response = response.content

    # 4. Parse & Validate Citations
    result = parse_and_validate_response(str(raw_response), retrieved_chunks)
    result["query"] = resolved_query
    result["retrieved_chunks"] = retrieved_chunks

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Query the Orient BlackSwan History Bot using strictly grounded RAG."
    )
    parser.add_argument("question", type=str, help="The question to ask about the PDF catalogue.")
    parser.add_argument(
        "--chroma-dir",
        type=str,
        default=os.getenv("CHROMA_DIR", "chroma_db"),
        help="Path to ChromaDB directory (default: chroma_db)",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=os.getenv("COLLECTION_NAME", "history_rag"),
        help="ChromaDB collection name (default: history_rag)",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=os.getenv("LLM_MODEL", "gemini-2.5-flash"),
        help="Gemini chat model ID (default: gemini-2.5-flash)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=int(os.getenv("TOP_K", "5")),
        help="Number of chunks to retrieve (default: 5)",
    )

    args = parser.parse_args()

    try:
        res = ask_question(
            query=args.question,
            chroma_dir=args.chroma_dir,
            collection_name=args.collection,
            llm_model=args.llm_model,
            top_k=args.top_k,
        )

        print("\n" + "=" * 70)
        print(f"QUESTION: {res['query']}")
        print("=" * 70)
        print(f"ANSWER: {res['answer']}\n")

        if res["citations"]:
            print("CITATIONS & SUPPORTING EVIDENCE:")
            for idx, cit in enumerate(res["citations"], start=1):
                print(f"  [{idx}] Book Page Label: {cit['page_label']} (Physical Page: {cit['page_number']})")
                print(f"      Source File: {cit['source']}")
                print(f"      Evidence: \"{cit['snippet']}\"")
        else:
            if res["answer"] == "Not found":
                print("CITATIONS: None (Question not supported by catalogue content)")
        print("=" * 70 + "\n")

    except Exception as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
