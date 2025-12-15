# vectorstore/indexer.py
import os
from pinecone import Pinecone
from vectorstore.embeddings import embed_text
from utils.hashing import content_hash

def get_pinecone_index():
    """
    Initialize and return the Pinecone index.
    """
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX")

    if not api_key:
        raise ValueError("PINECONE_API_KEY not set")
    if not index_name:
        raise ValueError("PINECONE_INDEX not set")

    pc = Pinecone(api_key=api_key)
    return pc.Index(index_name)

def upsert_chunks(chunks, batch_size: int = 50, doc_id: str = None):
    """
    Index chunks into Pinecone with embeddings using batched upserts.
    Deduplication is handled via SHA-256 content hash.
    
    Each chunk stores:
        - text (the chunk text)
        - doc_id (document identifier for filtering)
    
    Args:
        chunks (list): List of text chunks or dicts with {"text": ...}
        batch_size (int): Number of vectors per upsert request
        doc_id (str): Optional document identifier

    Returns:
        indexed_count, skipped_count
    """
    index = get_pinecone_index()
    indexed, skipped = 0, 0
    batch = []

    for chunk in chunks:
        text = chunk["text"] if isinstance(chunk, dict) else chunk
        chunk_id = content_hash(text)

        # Check if this chunk already exists
        existing = index.fetch(ids=[chunk_id])
        if existing and existing.get("vectors"):
            skipped += 1
            continue

        batch.append({
            "id": chunk_id,
            "values": embed_text(text),
            "metadata": {"text": text, "doc_id": doc_id or "unknown"}
        })
        indexed += 1

        if len(batch) >= batch_size:
            index.upsert(vectors=batch)
            batch.clear()

    # Upsert remaining batch
    if batch:
        index.upsert(vectors=batch)

    return indexed, skipped
