import hashlib
import math
from collections import Counter

from .bm25 import tokenize


class HashingDenseIndex:
    """Deterministic dense-vector baseline.

    This is not a neural embedding model. It exists so the system has a real
    dense retrieval arm and a clean provider boundary. In production, swap it
    for BGE-M3, Qwen3-Embedding, Voyage, Cohere, or OpenAI embeddings without
    changing the retriever contract.
    """

    def __init__(self, documents: list[dict], dimensions: int = 256):
        self.documents = documents
        self.dimensions = dimensions
        self.vectors = [self._embed(doc["text"]) for doc in documents]

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        counts = Counter(tokenize(text))
        for term, count in counts.items():
            digest = hashlib.md5(term.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:2], "little") % self.dimensions
            sign = 1.0 if digest[2] % 2 == 0 else -1.0
            vector[idx] += sign * (1.0 + math.log(count))
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        query_vector = self._embed(query)
        scored = []
        for doc, vector in zip(self.documents, self.vectors):
            score = sum(q * d for q, d in zip(query_vector, vector))
            if score > 0:
                result = dict(doc)
                result["score"] = round(score, 4)
                scored.append((score, result))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [doc for _, doc in scored[:top_k]]


class SentenceTransformerEmbeddingProvider:
    """Optional neural embedding adapter.

    The import is intentionally lazy. If sentence-transformers is installed,
    this provider can load a model such as BAAI/bge-m3 or a Qwen embedding
    checkpoint exposed through a local inference stack. If it is not installed,
    the application falls back to the built-in retrieval arms.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("sentence-transformers is not installed") from exc
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()
