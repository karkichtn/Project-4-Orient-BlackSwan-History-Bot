"""
=============================================================================
PDF Assistant — Minimal Classic Streamlit Interface
=============================================================================
A restrained, single-page PDF chatbot with charcoal-black and grey styling.
Workflow: Upload PDF -> Process Document -> Type Question -> Get Answer.
=============================================================================
"""

import os
import shutil
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Load local environment configuration
load_dotenv()

from ingest import (
    compute_file_hash,
    load_pdf_document,
    split_document_pages,
    get_embeddings,
    index_chunks_into_chromadb,
)
from ask import ask_question

# -----------------------------------------------------------------------------
# Page Configuration & Minimal Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="PDF Assistant",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    /* Charcoal-black & grey minimal theme */
    .stApp {
        background-color: #121214;
        color: #e4e4e7;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Center container and establish generous whitespace */
    .block-container {
        max-width: 720px;
        padding-top: 3.5rem;
        padding-bottom: 4rem;
        margin: 0 auto;
    }
    
    /* Header typography */
    .main-title {
        text-align: center;
        font-size: 2.2rem;
        font-weight: 600;
        color: #f4f4f5;
        letter-spacing: -0.02em;
        margin-bottom: 0.35rem;
    }
    
    .main-subtitle {
        text-align: center;
        font-size: 1rem;
        color: #a1a1aa;
        margin-bottom: 2.5rem;
        font-weight: 400;
    }
    
    /* Status line */
    .status-text {
        font-size: 0.875rem;
        color: #a1a1aa;
        margin-top: 0.5rem;
        margin-bottom: 1.5rem;
    }
    
    /* Clean answer container */
    .answer-container {
        background-color: #18181b;
        border: 1px solid #27272a;
        border-radius: 8px;
        padding: 1.25rem 1.5rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        line-height: 1.6;
        color: #f4f4f5;
        font-size: 1rem;
    }
    
    .answer-not-found {
        background-color: #1c1917;
        border: 1px solid #44403c;
        border-radius: 8px;
        padding: 1.25rem 1.5rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        color: #fca5a5;
        font-size: 1.05rem;
    }
    
    /* Source citation entry */
    .source-entry {
        border-left: 2px solid #52525b;
        padding-left: 1rem;
        margin-top: 0.85rem;
        margin-bottom: 0.85rem;
    }
    
    .source-page {
        font-weight: 600;
        font-size: 0.85rem;
        color: #d4d4d8;
        margin-bottom: 0.25rem;
    }
    
    .source-snippet {
        font-size: 0.875rem;
        color: #a1a1aa;
        font-style: italic;
        line-height: 1.5;
    }

    /* Streamlit widgets styling */
    div[data-testid="stFileUploader"] {
        margin-bottom: 0.5rem;
    }

    button[kind="primary"] {
        background-color: #27272a;
        color: #f4f4f5;
        border: 1px solid #3f3f46;
        border-radius: 6px;
        font-weight: 500;
    }
    button[kind="primary"]:hover {
        background-color: #3f3f46;
        border-color: #52525b;
        color: #ffffff;
    }
    
    /* Hide Streamlit default header/footer for ultra-clean UI */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PDF = str(BASE_DIR / "data" / "History_2024-25.pdf")
CHROMA_DIR = str(BASE_DIR / "chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "history_rag")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
if "gemini" in EMBEDDING_MODEL.lower() or "text-embedding" in EMBEDDING_MODEL.lower():
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")

# Read API key strictly from local environment (.env)
API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

# -----------------------------------------------------------------------------
# Session State
# -----------------------------------------------------------------------------
if "is_indexed" not in st.session_state:
    st.session_state.is_indexed = False
if "current_file_id" not in st.session_state:
    st.session_state.current_file_id = None
if "status_text" not in st.session_state:
    st.session_state.status_text = "Ready"
if "answer_result" not in st.session_state:
    st.session_state.answer_result = None

# Check if a previously built index already exists on disk
def check_disk_index():
    meta_path = Path(CHROMA_DIR) / "ingest_metadata.json"
    if meta_path.exists():
        try:
            import json
            with open(meta_path, "r") as f:
                saved = json.load(f)
            st.session_state.is_indexed = True
            st.session_state.current_file_id = saved.get("filename", "History_2024-25.pdf")
            st.session_state.status_text = f"Document indexed ({st.session_state.current_file_id})"
        except Exception:
            pass

if not st.session_state.is_indexed:
    check_disk_index()

# -----------------------------------------------------------------------------
# 1. Title & Subtitle
# -----------------------------------------------------------------------------
st.markdown("<div class='main-title'>PDF Assistant</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='main-subtitle'>Upload a document. Ask questions about its contents.</div>",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 2. PDF File Uploader
# -----------------------------------------------------------------------------
uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"],
    label_visibility="collapsed",
    help="Upload any PDF file. If left empty, the included History catalogue will be used.",
)

# Detect if user uploaded a new document, and reset state to prevent stale chunk leakage
current_upload_id = uploaded_file.name if uploaded_file else ("default" if Path(DEFAULT_PDF).exists() else None)
if current_upload_id != st.session_state.get("last_seen_file_id"):
    st.session_state.last_seen_file_id = current_upload_id
    if current_upload_id and current_upload_id != st.session_state.current_file_id:
        st.session_state.is_indexed = False
        st.session_state.status_text = "Ready"
        st.session_state.answer_result = None

# Determine active target file path
target_path = None
target_name = None

if uploaded_file is not None:
    temp_dir = Path("data/uploads")
    temp_dir.mkdir(parents=True, exist_ok=True)
    save_path = temp_dir / uploaded_file.name
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    target_path = str(save_path)
    target_name = uploaded_file.name
elif Path(DEFAULT_PDF).exists():
    target_path = DEFAULT_PDF
    target_name = "History_2024-25.pdf"

# -----------------------------------------------------------------------------
# 3. Process Document Button & 4. Status
# -----------------------------------------------------------------------------
col_proc, col_space = st.columns([1, 2])
process_btn = col_proc.button("Process Document", type="primary", use_container_width=True)

if process_btn:
    if not target_path or not Path(target_path).exists():
        st.error("Please select or upload a valid PDF file first.")
    else:
        st.session_state.status_text = "Processing..."
        with st.spinner("Processing document..."):
            try:
                doc_hash = compute_file_hash(target_path)
                pages = load_pdf_document(target_path)
                chunks = split_document_pages(pages, chunk_size=800, chunk_overlap=120)
                embeddings = get_embeddings(model_name=EMBEDDING_MODEL)
                
                index_chunks_into_chromadb(
                    chunks=chunks,
                    embeddings=embeddings,
                    chroma_dir=CHROMA_DIR,
                    collection_name=COLLECTION_NAME,
                    pdf_path=target_path,
                    doc_hash=doc_hash,
                    force_reindex=True,
                )
                
                st.session_state.is_indexed = True
                st.session_state.current_file_id = target_name
                st.session_state.status_text = f"Document indexed ({target_name})"
                st.session_state.answer_result = None
                st.rerun()
            except Exception as e:
                st.session_state.status_text = "Error indexing document"
                st.error(f"Processing failed: {e}")

# Display small processing status line
st.markdown(f"<div class='status-text'>Status: {st.session_state.status_text}</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 5. Question Input & 6. Ask Question Button
# -----------------------------------------------------------------------------
user_question = st.text_input(
    "Question",
    placeholder="Type your question here...",
    label_visibility="collapsed",
    disabled=not st.session_state.is_indexed,
)

ask_btn = st.button(
    "Ask Question",
    type="primary",
    disabled=not st.session_state.is_indexed,
)

if ask_btn:
    clean_q = user_question.strip() if user_question else ""
    if not clean_q:
        st.warning("Please type a question.")
    elif not st.session_state.is_indexed:
        st.error("Please process a document first.")
    elif not API_KEY or API_KEY.strip() in ("", "your_gemini_api_key_here"):
        st.error("Gemini API key is not configured. Please set GOOGLE_API_KEY in your local .env file.")
    else:
        with st.spinner("Searching document..."):
            try:
                res = ask_question(
                    query=clean_q,
                    chroma_dir=CHROMA_DIR,
                    collection_name=COLLECTION_NAME,
                    embedding_model=EMBEDDING_MODEL,
                    llm_model=LLM_MODEL,
                    api_key=API_KEY,
                )
                st.session_state.answer_result = res
            except Exception as e:
                st.error(f"Error querying document: {e}")

# -----------------------------------------------------------------------------
# 7. Answer Section & 8. Collapsible View Sources
# -----------------------------------------------------------------------------
if st.session_state.answer_result:
    res = st.session_state.answer_result
    answer = res.get("answer", "")
    citations = res.get("citations", [])
    is_supported = res.get("is_supported", False)

    if not is_supported or answer == "Not found":
        # Clean negative fallback answer
        st.markdown(
            "<div class='answer-not-found'>Not found</div>",
            unsafe_allow_html=True,
        )
    else:
        # Clean grounded answer
        st.markdown(
            f"<div class='answer-container'>{answer}</div>",
            unsafe_allow_html=True,
        )

        # Collapsible View Sources Section
        if citations:
            with st.expander("View Sources", expanded=False):
                for cit in citations:
                    p_label = cit.get("page_label", "")
                    p_num = cit.get("page_number", "")
                    quote = cit.get("snippet", "")
                    
                    page_display = f"Page {p_label}" if p_label else f"Page {p_num}"
                    if str(p_label) != str(p_num) and p_num:
                        page_display += f" (PDF Page {p_num})"

                    st.markdown(
                        f"""
                        <div class='source-entry'>
                            <div class='source-page'>{page_display}</div>
                            <div class='source-snippet'>"{quote}"</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
