import math
import re
from collections import Counter, defaultdict

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class BM25Index:
    """Small dependency-free BM25 index for the local knowledge base."""

    def __init__(self, documents: list[dict]):
        self.documents = documents
        self.doc_tokens = [tokenize(doc["text"]) for doc in documents]
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.avgdl = sum(self.doc_lengths) / max(len(self.doc_lengths), 1)
        self.term_freqs = [Counter(tokens) for tokens in self.doc_tokens]
        self.doc_freq = defaultdict(int)
        for tokens in self.doc_tokens:
            for term in set(tokens):
                self.doc_freq[term] += 1

    def _idf(self, term: str) -> float:
        n = len(self.documents)
        df = self.doc_freq.get(term, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        query_terms = tokenize(query)
        scores: list[tuple[float, dict]] = []
        k1 = 1.5
        b = 0.75

        for idx, doc in enumerate(self.documents):
            tf = self.term_freqs[idx]
            dl = self.doc_lengths[idx] or 1
            score = 0.0
            for term in query_terms:
                freq = tf.get(term, 0)
                if not freq:
                    continue
                idf = self._idf(term)
                denom = freq + k1 * (1 - b + b * (dl / self.avgdl))
                score += idf * ((freq * (k1 + 1)) / denom)
            if score > 0:
                result = dict(doc)
                result["score"] = round(score, 4)
                scores.append((score, result))

        scores.sort(key=lambda item: item[0], reverse=True)
        return [doc for _, doc in scores[:top_k]]
