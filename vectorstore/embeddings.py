# vectorstore/embeddings.py

import os
from pinecone import Pinecone

MODEL_NAME = "llama-text-embed-v2"

_pc = None  # lazy singleton


def get_pinecone_client():
    global _pc
    if _pc is None:
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY is not set")
        _pc = Pinecone(api_key=api_key)
    return _pc


def embed_text(text: str, input_type: str = "passage") -> list:
    """
    Generate embeddings using Pinecone Inference API (LLaMA).

    input_type:
      - "passage" → for indexing documents
      - "query"   → for search queries
    """
    pc = get_pinecone_client()

    response = pc.inference.embed(
        model=MODEL_NAME,
        inputs=[text],
        parameters={"input_type": input_type}
    )

    return response.data[0].values
