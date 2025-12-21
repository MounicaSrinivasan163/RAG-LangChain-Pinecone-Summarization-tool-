import hashlib

def content_hash(file_bytes: bytes) -> str:
    """
    Stable document ID based on actual file content
    """
    return hashlib.sha256(file_bytes).hexdigest()
