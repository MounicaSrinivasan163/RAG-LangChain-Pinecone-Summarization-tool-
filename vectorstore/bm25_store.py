from rank_bm25 import BM25Okapi
import re

class BM25Store:
    def __init__(self):
        self.corpus = []
        self.chunks = []
        self.bm25 = None

    def _tokenize(self, text: str):
        return re.findall(r"\w+", text.lower())

    def add_chunks(self, chunks):
        for chunk in chunks:
            tokens = self._tokenize(chunk["text"])
            self.corpus.append(tokens)
            self.chunks.append(chunk)

        self.bm25 = BM25Okapi(self.corpus)

    def search(self, query: str, top_k: int = 5):
        if not self.bm25:
            return []

        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        ranked = sorted(
            zip(scores, self.chunks),
            key=lambda x: x[0],
            reverse=True
        )

        return [chunk for score, chunk in ranked[:top_k] if score > 0]
