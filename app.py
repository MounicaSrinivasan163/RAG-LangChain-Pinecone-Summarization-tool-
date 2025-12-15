# app.py

from dotenv import load_dotenv
load_dotenv()

import streamlit as st

# ============================================================
# 🔐 SESSION STATE INITIALIZATION (ABSOLUTE TOP)
# ============================================================
st.session_state.setdefault("indexed", False)
st.session_state.setdefault("last_summary", "")
st.session_state.setdefault("indexed_docs", [])


import streamlit as st
from utils.file_loader import load_file
from vectorstore.indexer import upsert_chunks
from vectorstore.retriever import retrieve_chunks
from crew.rag_crew import summarizer_agent
from evaluation.rouge_eval import evaluate_summary
from utils.hashing import content_hash  # for generating doc_id

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="📘 Book Search, Summarization & Evaluation",
    layout="wide"
)

st.title("📘 Book Search, Summarization & Evaluation")

# ----------------------------
# Session State
# ----------------------------
if "indexed_docs" not in st.session_state:
    st.session_state.indexed_docs = {}  # doc_name -> doc_id

if "last_summary" not in st.session_state:
    st.session_state.last_summary = ""

# ----------------------------
# Tabs
# ----------------------------
tab1, tab2, tab3 = st.tabs(
    ["📚 Indexing", "🔍 Search & Summarize", "📊 Evaluation"]
)

# ============================================================
# TAB 1 — INDEXING
# ============================================================
with tab1:
    st.header("📚 Index Book")

    uploaded_files = st.file_uploader(
        "Upload one or more documents",
        type=["pdf", "txt", "csv", "docx"],
        accept_multiple_files=True
    )

    if uploaded_files:
        total_chunks = 0
        total_indexed = 0
        total_skipped = 0

        for uploaded_file in uploaded_files:
            st.write(f"📄 **{uploaded_file.name}** received")

            # Generate a unique doc_id for this file
            doc_id = content_hash(uploaded_file.name)
            st.session_state.indexed_docs[uploaded_file.name] = doc_id

            st.write("✂️ Chunking document...")
            chunks = load_file(
                uploaded_file,
                chunk_size=500,
                overlap=100
            )
            st.success(f"✅ {len(chunks)} chunks created")
            total_chunks += len(chunks)

            st.write("📦 Indexing chunks...")
            indexed, skipped = upsert_chunks(chunks, doc_id=doc_id)

            total_indexed += indexed
            total_skipped += skipped

        st.divider()
        st.success("🎉 Document ingestion completed")
        st.write(f"📊 Total chunks processed: {total_chunks}")
        st.write(f"🆕 Newly indexed: {total_indexed}")
        st.write(f"⏭️ Skipped (duplicates): {total_skipped}")

# ============================================================
# TAB 2 — SEARCH & SUMMARIZE
# ============================================================
with tab2:
    st.header("🔍 Search & Summarize")

    if not st.session_state.indexed:
        st.info("👈 Please index documents first.")
    else:
        # -----------------------------
        # Document selector (already indexed)
        # -----------------------------
        doc_names = st.session_state.get("indexed_docs", [])

        selected_doc = st.selectbox(
            "Select document to search from",
            options=doc_names if doc_names else ["All Documents"]
        )

        query = st.text_input(
            "Enter your query",
            placeholder="e.g. Types of supervised learning"
        )

        summary_length = st.slider(
            "📝 Summary length (words)",
            50,
            500,
            200
        )

        # 🔘 EXPLICIT BUTTON (KEY FIX)
        summarize_clicked = st.button("🧠 Summarize")

        if summarize_clicked:
            if not query:
                st.warning("Please enter a query.")
            else:
                with st.spinner("🔎 Retrieving & summarizing..."):

                    # 🔹 Retrieve chunks (document-aware)
                    retrieved_chunks = retrieve_chunks(
                        query=query,
                        top_k=5,
                        doc_id=None if selected_doc == "All Documents" else selected_doc
                    )

                    # 🔹 SAFE TEXT EXTRACTION (NO None allowed)
                    texts = []
                    for c in retrieved_chunks:
                        if isinstance(c, dict):
                            txt = c.get("metadata", {}).get("text")
                        else:
                            txt = getattr(c, "page_content", None)

                        if isinstance(txt, str) and txt.strip():
                            texts.append(txt)

                    if not texts:
                        st.error("No relevant content found.")
                    else:
                        combined_text = "\n\n".join(texts)

                        prompt = f"""
                        Summarize the following content in about {summary_length} words:

                        {combined_text}
                        """

                        summary = summarizer_agent.llm.invoke(prompt).content
                        st.session_state.last_summary = summary

                        st.subheader("📄 Summary")
                        st.write(summary)


# ============================================================
# TAB 3 — EVALUATION
# ============================================================
with tab3:
    st.header("📊 Evaluation")

    if not st.session_state.last_summary:
        st.info("Generate a summary first to evaluate.")
    else:
        if st.checkbox("📐 Evaluate Summary using ROUGE"):
            scores = evaluate_summary(st.session_state.last_summary)
            st.subheader("ROUGE Scores")
            st.json(scores)

        st.subheader("🧑‍⚖️ Human Evaluation (Manual)")
        st.markdown("""
        **Instructions for Human Evaluation:**
        1. Read the generated summary.
        2. Compare it with the original section in the book.
        3. Rate on:
           - Relevance
           - Coverage
           - Coherence
           - Correctness
        4. Record feedback in your report.
        """)

        st.text_area(
            "Human Evaluation Notes",
            placeholder="Write evaluator feedback here..."
        )
