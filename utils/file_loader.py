from PyPDF2 import PdfReader
from io import BytesIO
import streamlit as st

def load_file(uploaded_file, chunk_size=200, overlap=50):
    """
    Load and split a file into chunks with optional overlap.
    Supports PDF, TXT, CSV, DOCX.
    Returns a list of chunk dictionaries: [{"id": ..., "text": ...}, ...]
    """
    text = ""

    # ---------- Read file ----------
    if uploaded_file.name.endswith(".pdf"):
        pdf = PdfReader(BytesIO(uploaded_file.read()))
        text = "\n".join([page.extract_text() for page in pdf.pages])
    elif uploaded_file.name.endswith(".txt"):
        text = uploaded_file.read().decode("utf-8")
    elif uploaded_file.name.endswith(".csv"):
        import pandas as pd
        df = pd.read_csv(uploaded_file)
        text = df.to_csv(index=False)
    elif uploaded_file.name.endswith(".docx"):
        from docx import Document
        doc = Document(uploaded_file)
        text = "\n".join([para.text for para in doc.paragraphs])
    else:
        st.warning(f"Unsupported file type: {uploaded_file.name}")
        return []

    # ---------- Split into chunks ----------
    words = text.split()
    chunks = []

    progress_bar = st.progress(0)
    total_chunks = max(1, len(words) // chunk_size + (1 if len(words) % chunk_size else 0))

    start = 0
    chunk_num = 1
    while start < len(words):
        end = start + chunk_size
        chunk_text = " ".join(words[start:end])
        chunks.append({"id": f"{uploaded_file.name}_chunk_{chunk_num}", "text": chunk_text})
        start += chunk_size - overlap
        chunk_num += 1
        progress_bar.progress(min(chunk_num / total_chunks, 1.0))

    progress_bar.empty()  # Remove progress bar after completion
    return chunks
