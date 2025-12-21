from dotenv import load_dotenv
load_dotenv()

import os
import hashlib
from pinecone import Pinecone
from vectorstore.embeddings import embed_text

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX"))


def _stable_id(text: str) -> str:
    """Deterministic ID to avoid duplicates & overwrites"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def upsert_chunks(chunks):
    vectors = []

    for chunk in chunks:
        text = chunk["text"] if isinstance(chunk, dict) else chunk

        vectors.append({
            "id": _stable_id(text),
            "values": embed_text(text, input_type="passage"),  # ✅ correct
            "metadata": {"text": text}
        })

    index.upsert(vectors=vectors)
    return len(vectors)


def query_index(chunk, top_k: int = 1):
    """
    Used ONLY to check whether a similar chunk already exists
    """

    text = chunk["text"] if isinstance(chunk, dict) else chunk

    vector = embed_text(text, input_type="query")  # ✅ correct

    res = index.query(
        vector=vector,
        top_k=top_k,
        include_metadata=False
    )

    return res.matches if res and hasattr(res, "matches") else []
