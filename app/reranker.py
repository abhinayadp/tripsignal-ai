from .bm25 import tokenize


class FeatureReranker:
    """Reranks fused candidates with interpretable ranking features.

    Production systems would swap this for a cross-encoder such as
    bge-reranker-v2-m3, Qwen3-Reranker, Cohere Rerank, or Voyage Rerank.
    The feature layer makes ranking behavior testable in the MVP and mirrors
    the same contract: query plus candidate documents in, reranked documents
    out.
    """

    def rerank(self, query: str, documents: list[dict], top_k: int = 3) -> list[dict]:
        query_terms = set(tokenize(query))
        query_lower = query.lower()
        reranked = []
        for doc in documents:
            text_lower = doc["text"].lower()
            title_lower = doc["title"].lower()
            doc_terms = set(tokenize(doc["text"]))
            coverage = len(query_terms & doc_terms) / max(len(query_terms), 1)
            title_hits = sum(1 for term in query_terms if term in title_lower)
            phrase_bonus = 0.15 if query_lower[:24] and query_lower[:24] in text_lower else 0.0
            score = doc.get("retrieval_score", 0.0) + coverage * 0.02 + title_hits * 0.004 + phrase_bonus
            item = dict(doc)
            item["rerank_score"] = round(score, 6)
            item["rerank_features"] = {
                "term_coverage": round(coverage, 4),
                "title_hits": title_hits,
                "phrase_bonus": phrase_bonus,
            }
            reranked.append(item)
        reranked.sort(key=lambda doc: doc["rerank_score"], reverse=True)
        return reranked[:top_k]
