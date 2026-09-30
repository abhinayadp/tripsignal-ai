"""Corrective RAG-lite: evidence grading and one-shot query rewriting.

The static retrieval trap is that a top-K pipeline returns *something*
whether or not it is relevant. This module grades the top retrieved
document against the query's *concepts* (content terms, with known
synonym variants counting as the same concept). If the grade is below
threshold, the query is rewritten once with synonym / policy-vocabulary
expansion and retrieval is retried. If the retry still grades below
threshold, the caller must refuse rather than invent an answer.

Grading always uses the *original* query's concepts, so expansion terms
cannot inflate the grade: a retry only passes if it finds a document
that actually covers what the user asked about.
"""

from .bm25 import tokenize

import re

EVIDENCE_THRESHOLD = 0.35

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "do", "does", "did",
    "what", "when", "where", "which", "who", "how", "can", "could",
    "should", "would", "will", "i", "me", "my", "we", "you", "your",
    "it", "its", "this", "that", "if", "of", "to", "in", "on", "for",
    "with", "and", "or", "am", "be", "been", "get", "got", "have", "has",
    "about", "tell", "please", "me",
}

# Concept variants used only for grading: a concept counts as covered if
# any variant appears in the document. This is how "money back" can be
# covered by a document that says "refund".
CONCEPT_VARIANTS: dict[str, set[str]] = {
    "scrapped": {"cancelled", "cancellation"},
    "money": {"refund"},
    "back": {"refund"},
    "tight": {"minimum", "connection"},
    "owed": {"refund", "compensation", "entitled"},
    "room": {"hotel"},
    "pay": {"reimbursement", "reimburse", "cover"},
    "stranded": {"delay", "overnight", "hotel"},
    "stuck": {"delay", "overnight", "hotel"},
    "plane": {"flight", "rebooking"},
    "another": {"rebooking", "alternate"},
    "documents": {"passport", "visa"},
    "carry": {"baggage", "carry-on"},
}

# Expansions appended by the one-shot rewrite when the trigger phrase or
# token is present. These use the vocabulary the policy documents use.
PHRASE_EXPANSIONS: list[tuple[str, str]] = [
    ("money back", "refund original payment method"),
    ("another plane", "rebooking alternate flight"),
    ("too tight", "minimum connection time misconnection risk"),
    ("get back", "refund"),
]

TOKEN_EXPANSIONS: dict[str, str] = {
    "scrapped": "cancelled cancellation refund",
    "stranded": "overnight delay hotel meal voucher",
    "stuck": "delayed cancellation hotel benefits",
    "tight": "connection risk minimum connection time",
    "owed": "entitled refund compensation passenger rights",
    "room": "hotel voucher benefits",
    "vouchers": "goodwill benefits policy",
    "voucher": "goodwill benefits policy refund",
    "lounge": "lounge access Priority Pass benefits",
    "compensation": "passenger rights refund controllable",
    "documents": "passport visa checklist",
}


def _normalize(term: str) -> str:
    if len(term) > 3 and term.endswith("s"):
        return term[:-1]
    return term


def terms_match(a: str, b: str) -> bool:
    """Light stemming match: exact after normalization, or shared 6-char
    prefix for longer words (cancelled ~ cancellation, rebooking ~ rebook)."""
    a, b = _normalize(a), _normalize(b)
    if a == b:
        return True
    return min(len(a), len(b)) >= 6 and a[:6] == b[:6]


def matched_concepts(query: str, text: str) -> int:
    """How many of the query's concepts are covered by the text."""
    concepts = set(content_terms(query))
    text_terms = tokenize(text)
    covered = 0
    for concept in concepts:
        variants = {concept} | CONCEPT_VARIANTS.get(concept, set())
        if any(terms_match(variant, term) for variant in variants for term in text_terms):
            covered += 1
    return covered


def content_terms(text: str) -> list[str]:
    return [term for term in tokenize(text) if term not in STOPWORDS and len(term) >= 3]


def concept_coverage(query: str, doc_text: str) -> float:
    """Fraction of the query's concepts covered by the document."""
    concepts = set(content_terms(query))
    if not concepts:
        return 0.0
    return matched_concepts(query, doc_text) / len(concepts)


def _split_sentences(body: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", body)
    merged: list[str] = []
    for part in parts:
        if merged and re.search(r"\b[A-Z]\.$", merged[-1]):
            merged[-1] += " " + part
        else:
            merged.append(part)
    return merged


def grade_evidence(query: str, evidence: list[dict], threshold: float = EVIDENCE_THRESHOLD) -> dict:
    """Grade the top retrieved document against the original query."""
    if not evidence:
        return {"score": 0.0, "threshold": threshold, "passed": False, "top_doc": None}
    top = evidence[0]
    score = concept_coverage(query, top["text"])
    return {
        "score": round(score, 4),
        "threshold": threshold,
        "passed": score >= threshold,
        "top_doc": top["id"],
    }


def rewrite_query(query: str) -> str:
    """One-shot synonym / policy-vocabulary expansion. No-op if no trigger."""
    lower = query.lower()
    additions: list[str] = []
    for phrase, expansion in PHRASE_EXPANSIONS:
        if phrase in lower:
            additions.append(expansion)
    for token in tokenize(query):
        expansion = TOKEN_EXPANSIONS.get(token)
        if expansion:
            additions.append(expansion)
    if not additions:
        return query
    existing = set(tokenize(query))
    extra = [term for term in " ".join(additions).split() if term.lower() not in existing]
    if not extra:
        return query
    return query + " " + " ".join(extra)
