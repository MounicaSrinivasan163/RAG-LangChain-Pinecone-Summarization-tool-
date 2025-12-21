import os
from pinecone import Pinecone
from vectorstore.embeddings import embed_texts
from docs_loader import save_documents

EMBED_BATCH = 32
UPSERT_BATCH = 100

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX"))


def document_exists(doc_id: str) -> bool:
    """
    True full-document deduplication
    """
    try:
        res = index.fetch(ids=[f"{doc_id}_0"])
        return bool(res.vectors)
    except Exception:
        return False


def upsert_chunks(chunks, doc_id: str, doc_name: str):
    """
    Upserts document chunks into Pinecone
    Persists doc registry for cross-session dropdown
    """
    indexed = 0
    skipped = 0

    # ✅ Full-document deduplication
    if document_exists(doc_id):
        skipped = len(chunks)
        return indexed, skipped

    texts = [c["text"] for c in chunks]

    # ---- Embed in batches ----
    embeddings = []
    for i in range(0, len(texts), EMBED_BATCH):
        embeddings.extend(embed_texts(texts[i:i + EMBED_BATCH]))

    # ---- Prepare vectors ----
    vectors = []
    for i, (chunk, vector) in enumerate(zip(chunks, embeddings)):
        vectors.append({
            "id": f"{doc_id}_{i}",
            "values": vector,
            "metadata": {
                "doc_id": doc_id,
                "doc_name": doc_name,   # ✅ filename
                "chunk_index": i,
                "text": chunk["text"]
            }
        })

    # ---- Safe batched upsert ----
    for i in range(0, len(vectors), UPSERT_BATCH):
        batch = vectors[i:i + UPSERT_BATCH]
        index.upsert(vectors=batch)
        indexed += len(batch)

    # ---- Persist document registry ----
    save_documents(
        doc_name=doc_name,
        doc_id=doc_id
    )

    return indexed, skipped
