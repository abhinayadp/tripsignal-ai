from .bm25 import BM25Index
from .tfidf import TFIDFIndex


class HybridRetriever:
    """Fuse BM25 and TF-IDF rankings with Reciprocal Rank Fusion."""

    def __init__(self, documents: list[dict]):
        self.documents = documents
        self.bm25 = BM25Index(documents)
        self.tfidf = TFIDFIndex(documents)

    @staticmethod
    def _rrf(rank: int, k: int = 60) -> float:
        return 1.0 / (k + rank)

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        fused: dict[str, dict] = {}
        for rank, doc in enumerate(self.bm25.search(query, top_k=20), start=1):
            item = fused.setdefault(doc["id"], dict(doc))
            item["retrieval_score"] = item.get("retrieval_score", 0.0) + self._rrf(rank)
            item["bm25_rank"] = rank
        for rank, doc in enumerate(self.tfidf.search(query, top_k=20), start=1):
            item = fused.setdefault(doc["id"], dict(doc))
            item["retrieval_score"] = item.get("retrieval_score", 0.0) + self._rrf(rank)
            item["tfidf_rank"] = rank

        results = sorted(fused.values(), key=lambda doc: doc.get("retrieval_score", 0.0), reverse=True)
        for result in results:
            result["retrieval_score"] = round(result.get("retrieval_score", 0.0), 6)
        return results[:top_k]
