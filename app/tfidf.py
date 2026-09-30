import math
from collections import Counter, defaultdict

from .bm25 import tokenize


class TFIDFIndex:
    """Dependency-free TF-IDF cosine index used as the semantic retrieval arm.

    This is not a neural embedding model. It gives the MVP a second retrieval
    signal so the hybrid ranker can combine exact lexical matches from BM25
    with weighted term similarity. The production upgrade path swaps this
    class for Qwen3-Embedding, BGE-M3, Voyage, or OpenAI embeddings.
    """

    def __init__(self, documents: list[dict]):
        self.documents = documents
        self.doc_tokens = [tokenize(doc["text"]) for doc in documents]
        self.doc_freq = defaultdict(int)
        for tokens in self.doc_tokens:
            for term in set(tokens):
                self.doc_freq[term] += 1
        self.vectors = [self._vectorize(tokens) for tokens in self.doc_tokens]

    def _idf(self, term: str) -> float:
        n = len(self.documents)
        df = self.doc_freq.get(term, 0)
        return math.log((1 + n) / (1 + df)) + 1.0

    def _vectorize(self, tokens: list[str]) -> dict[str, float]:
        counts = Counter(tokens)
        vector = {term: count * self._idf(term) for term, count in counts.items()}
        norm = math.sqrt(sum(value * value for value in vector.values())) or 1.0
        return {term: value / norm for term, value in vector.items()}

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        query_vector = self._vectorize(tokenize(query))
        scored = []
        for doc, vector in zip(self.documents, self.vectors):
            score = sum(query_vector.get(term, 0.0) * value for term, value in vector.items())
            if score > 0:
                result = dict(doc)
                result["score"] = round(score, 4)
                scored.append((score, result))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [doc for _, doc in scored[:top_k]]
