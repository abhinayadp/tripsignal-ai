from .bm25 import BM25Index
from .embeddings import HashingDenseIndex
from .reranker import FeatureReranker
from .tfidf import TFIDFIndex


class HybridRetriever:
    """Fuse BM25, TF-IDF, and dense-vector rankings, then rerank.

    Retrieval arms:
    - BM25 catches exact terms, flight codes, airport codes, and policy words.
    - TF-IDF gives a weighted lexical similarity signal.
    - HashingDenseIndex provides a dense-vector contract that can be swapped
      for BGE-M3, Qwen3-Embedding, Voyage, Cohere, or OpenAI embeddings.

    Reciprocal Rank Fusion combines the ranked lists without score
    normalization. FeatureReranker then promotes candidates with stronger
    query coverage and title alignment before the answer composer sees them.
    """

    def __init__(self, documents: list[dict]):
        self.documents = documents
        self.bm25 = BM25Index(documents)
        self.tfidf = TFIDFIndex(documents)
        self.dense = HashingDenseIndex(documents)
        self.reranker = FeatureReranker()

    @staticmethod
    def _rrf(rank: int, k: int = 60) -> float:
        return 1.0 / (k + rank)

    def _fuse(self, query: str) -> list[dict]:
        fused: dict[str, dict] = {}
        arms = [
            ("bm25_rank", self.bm25.search(query, top_k=20)),
            ("tfidf_rank", self.tfidf.search(query, top_k=20)),
            ("dense_rank", self.dense.search(query, top_k=20)),
        ]
        for rank_key, results in arms:
            for rank, doc in enumerate(results, start=1):
                item = fused.setdefault(doc["id"], dict(doc))
                item["retrieval_score"] = item.get("retrieval_score", 0.0) + self._rrf(rank)
                item[rank_key] = rank
        results = sorted(fused.values(), key=lambda doc: doc.get("retrieval_score", 0.0), reverse=True)
        for result in results:
            result["retrieval_score"] = round(result.get("retrieval_score", 0.0), 6)
        return results

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        candidates = self._fuse(query)
        return self.reranker.rerank(query, candidates, top_k=top_k)
