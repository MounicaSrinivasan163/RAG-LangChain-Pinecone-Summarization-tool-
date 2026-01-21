# vectorstore/retriever.py

import os
from pinecone import Pinecone
from vectorstore.embeddings import embed_texts

# ---------------- Pinecone Init ----------------
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX"))


def retrieve_chunks(
    query: str,
    doc_id: str = None,
    top_k: int = 10,
    bm25_store=None,
    rerank_top_k: int = 5,
):
    """
    Hybrid Retrieval:
    - Vector search (Pinecone)
    - BM25 keyword search (optional)
    - BGE reranker (Pinecone)
    """

    # ---------------- Vector Search ----------------
    query_vector = embed_texts([query], input_type="query")[0]

    filter_ = {"doc_id": doc_id} if doc_id else None

    response = index.query(
        vector=query_vector,
        top_k=30,  # fetch more for reranking
        include_metadata=True,
        filter=filter_,
    )

    vector_results = []
    if response and response.matches:
        for m in response.matches:
            if m.metadata and "text" in m.metadata:
                vector_results.append(
                    {
                        "id": m.id,
                        "text": m.metadata["text"],
                    }
                )

    # ---------------- BM25 Search ----------------
    bm25_results = []
    if bm25_store:
        bm25_hits = bm25_store.search(query, top_k=20)
        for c in bm25_hits:
            bm25_results.append(
                {
                    "id": c["id"],
                    "text": c["text"],
                }
            )

    # ---------------- Merge & Deduplicate ----------------
    merged = {}
    for item in vector_results + bm25_results:
        merged[item["id"]] = item["text"]

    candidates = list(merged.values())

    # 🚨 No candidates → return empty
    if not candidates:
        return []

    # ---------------- BGE Reranker ----------------
    rerank_response = pc.inference.rerank(
        model="bge-reranker-v2-m3",
        query=query,
        documents=candidates,
        top_n=rerank_top_k,
    )

    # 🚨 Reranker failed or returned nothing → fallback
    if not rerank_response or not rerank_response.results:
        return candidates[:top_k]

    # ---------------- Final Ranked Chunks ----------------
    reranked_chunks = [
        r.document["text"]
        for r in rerank_response.results
        if r.document and "text" in r.document
    ]

    return reranked_chunks[:top_k]
