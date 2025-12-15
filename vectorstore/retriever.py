# vectorstore/retriever.py

import os
from pinecone import Pinecone
from vectorstore.embeddings import embed_text

def get_pinecone_index():
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX")

    if not api_key:
        raise ValueError("PINECONE_API_KEY not set")
    if not index_name:
        raise ValueError("PINECONE_INDEX not set")

    pc = Pinecone(api_key=api_key)
    return pc.Index(index_name)


def retrieve_chunks(query, top_k=5, doc_id=None):
    """
    Retrieve top_k chunks relevant to the query from Pinecone.
    Can filter by doc_id if provided.

    Args:
        query (str): Search query
        top_k (int): Number of chunks to retrieve
        doc_id (str, optional): Filter chunks by this document ID

    Returns:
        List[dict]: List of retrieved chunks
    """
    index = get_pinecone_index()

    # Get embedding for the query
    query_vector = embed_text(query, input_type="query")

    # Build Pinecone filter
    pinecone_filter = {"doc_id": {"$eq": doc_id}} if doc_id else None

    # Query Pinecone
    response = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
        filter=pinecone_filter
    )

    # Return the list of metadata dicts
    return response.get("matches", [])
