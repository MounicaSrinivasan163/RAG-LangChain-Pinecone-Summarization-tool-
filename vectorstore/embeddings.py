# vectorstore/embeddings.py
import os
import time
from pinecone import Pinecone
from pinecone.exceptions import PineconeApiException

MODEL_NAME = "llama-text-embed-v2"

# 🔑 SAFE VALUES FOR FREE / STARTER PLAN
BATCH_SIZE = 20        # was 50 → too aggressive
SLEEP_SECONDS = 1.2    # throttle between batches
MAX_RETRIES = 3

_pc = None


def get_pinecone_client():
    global _pc
    if _pc is None:
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY not set")
        _pc = Pinecone(api_key=api_key)
    return _pc


def embed_texts(texts: list[str], input_type: str = "passage") -> list[list[float]]:
    """
    Rate-limit safe batch embedding
    """
    pc = get_pinecone_client()
    embeddings = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]

        for attempt in range(MAX_RETRIES):
            try:
                response = pc.inference.embed(
                    model=MODEL_NAME,
                    inputs=batch,
                    parameters={"input_type": input_type}
                )

                embeddings.extend([d.values for d in response.data])
                break

            except PineconeApiException as e:
                if e.status == 429 and attempt < MAX_RETRIES - 1:
                    time.sleep(3)  # backoff
                else:
                    raise

        # 🔒 Throttle to respect TPM
        time.sleep(SLEEP_SECONDS)

    return embeddings
