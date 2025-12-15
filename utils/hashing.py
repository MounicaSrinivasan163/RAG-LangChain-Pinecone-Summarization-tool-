# utils/hashing.py

import hashlib


def content_hash(text: str) -> str:
    """
    Generate a deterministic SHA-256 hash for document content.
    Used for deduplication before indexing.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
