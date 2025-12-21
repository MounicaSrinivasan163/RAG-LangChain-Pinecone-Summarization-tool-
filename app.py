# app.py
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

import os
import csv
from datetime import datetime

from docs_loader import save_documents, load_documents
from utils.hashing import content_hash
from vectorstore.indexer import upsert_chunks, document_exists
from vectorstore.retriever import retrieve_chunks
from vectorstore.embeddings import embed_texts
from crew.rag_crew import summarize_chunks_task
from evaluation.rouge_eval import evaluate_summary
from reference_summaries import EVAL_QUESTIONS
from vectorstore.bm25_store import BM25Store

# ============================================================
# ✅ SESSION STATE INITIALIZATION
# ============================================================
if "indexed_docs" not in st.session_state:
    st.session_state.indexed_docs = load_documents()  # {doc_name: doc_id}

if "all_chunks" not in st.session_state:
    st.session_state.all_chunks = {}

if "last_summary" not in st.session_state:
    st.session_state.last_summary = ""

if "last_query" not in st.session_state:
    st.session_state.last_query = ""

if "bm25" not in st.session_state:
    st.session_state.bm25 = BM25Store()



# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="📘 Doc Search, Summarization & Evaluation",
    layout="wide"
)
st.title("📘 Doc Search, Summarization & Evaluation")
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
            file_bytes = uploaded_file.getvalue()
            doc_id = content_hash(file_bytes)

            # Skip if already indexed
            if document_exists(doc_id):
                st.write("📄 Document already indexed. Skipping.")
                # Ensure already indexed doc is in session_state & CSV
                st.session_state.indexed_docs.setdefault(uploaded_file.name, doc_id)
                save_documents(uploaded_file.name, doc_id)
                continue

            st.session_state.indexed_docs[uploaded_file.name] = doc_id
            save_documents(uploaded_file.name, doc_id)  # persist

            # -------- Chunking --------
            st.write("✂️ Chunking document...")
            from utils.file_loader import load_file  # lazy import
            chunks = load_file(uploaded_file, chunk_size=400, overlap=80)
            total_chunks += len(chunks)
            st.session_state.bm25.add_chunks(chunks)
            st.success(f"✅ {len(chunks)} chunks created")

            # -------- Batched Embedding --------
            st.write("🧠 Generating embeddings (batched)...")
            texts = [chunk["text"] for chunk in chunks]
            vectors = embed_texts(texts, input_type="passage")

            for chunk, vector in zip(chunks, vectors):
                chunk["vector"] = vector
                st.session_state.all_chunks[chunk["id"]] = chunk

            # -------- Pinecone Upsert --------
            st.write("📦 Indexing chunks...")
            indexed, skipped = upsert_chunks(chunks, doc_id=doc_id, doc_name=uploaded_file.name)
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

    # Load persistent document registry (doc_name → doc_id)
    doc_map = load_documents()

    doc_names = ["All Documents"] + list(doc_map.keys())
    selected_doc = st.selectbox("Select document to search from", options=doc_names)

    query = st.text_input("Enter your query", placeholder="e.g. Types of supervised learning")
    st.session_state.last_query = query

    summary_length = st.slider("📝 Summary length (words)", 10, 500, 200)

    if st.button("🧠 Summarize"):
        if not query.strip():
            st.warning("Please enter a query.")
        else:
            with st.spinner("🔎 Retrieving & summarizing..."):

                # Resolve doc_id safely
                doc_id = None
                if selected_doc != "All Documents":
                    doc_id = doc_map.get(selected_doc)

                retrieved_chunks = retrieve_chunks(query=query, doc_id=doc_id,
                                                    top_k=5, bm25_store=st.session_state.bm25 )


                if not retrieved_chunks:
                    st.error("No relevant content found.")
                else:
                    result = summarize_chunks_task({
                                 "query": query,
                                 "retrieved_chunks": retrieved_chunks,
                                 "summary_length": summary_length
                                                    })


                    st.session_state.last_summary = result["summary"]

                    st.subheader("📄 Summary")
                    st.write(st.session_state.last_summary)

# ============================================================
# TAB 3 — EVALUATION
# ============================================================

CSV_FILE = "human_evaluations.csv"

def save_human_eval(row: dict):
    file_exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

with tab3:
    st.header("📊 Evaluation")

    if not st.session_state.last_summary:
        st.info("Generate a summary first to evaluate.")
    else:
        query = st.session_state.last_query
        generated = st.session_state.last_summary
        reference = EVAL_QUESTIONS.get(query)

        # Automatic ROUGE evaluation
        if st.checkbox("📐 Evaluate Summary using ROUGE"):
            if not reference:
                st.warning("No gold reference available for this question.")
            else:
                scores = evaluate_summary(generated, reference)
                st.subheader("ROUGE Scores")
                st.json(scores)

        # Human Evaluation
        st.subheader("🧑‍⚖️ Human Evaluation (Stored as CSV)")
        if reference:
            with st.expander("📘 Gold Reference Answer"):
                st.write(reference)

        st.markdown("### Rate the summary (1 = Poor, 5 = Excellent)")

        # ======= Two columns layout =======
        col1, col2 = st.columns(2)

        with col1:
            relevance = st.slider("🎯 Relevance", 1, 5, 3)
            coverage = st.slider("📚 Coverage / Completeness", 1, 5, 3)
            correctness = st.slider("✅ Correctness", 1, 5, 3)

        with col2:
            faithfulness = st.slider("🧠 Faithfulness (No Hallucinations)", 1, 5, 3)
            coherence = st.slider("🧩 Coherence & Readability", 1, 5, 3)

        notes = st.text_area(
            "✍️ Evaluator Notes",
            placeholder="Mention missing points, hallucinations, factual errors, strengths..."
        )

        if st.button("💾 Save Human Evaluation"):
            row = {
                "timestamp": datetime.utcnow().isoformat(),
                "question": query,
                "generated_summary": generated,
                "reference_summary": reference,
                "relevance": relevance,
                "coverage": coverage,
                "correctness": correctness,
                "faithfulness": faithfulness,
                "coherence": coherence,
                "notes": notes
            }
            save_human_eval(row)
            st.success("Human evaluation saved to CSV.")

