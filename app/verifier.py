import re


class AnswerVerifier:
    """Self-RAG-lite verification for generated answers.

    The verifier checks high-risk factual surfaces that should never be
    invented by the composer: prices, citation presence, and explicit refusal
    behavior. It is intentionally conservative and returns warnings rather
    than silently rewriting user-facing text.
    """

    PRICE_RE = re.compile(r"\$(\d{3,5})")

    def verify(self, answer: str, allowed_prices: set[int] | None = None) -> dict:
        allowed_prices = allowed_prices or set()
        found_prices = {int(value) for value in self.PRICE_RE.findall(answer)}
        unsupported_prices = sorted(found_prices - allowed_prices) if allowed_prices else []
        warnings = []
        if unsupported_prices:
            warnings.append(f"Unsupported prices found: {unsupported_prices}")
        if "Sources:" not in answer and "could not find" not in answer.lower():
            warnings.append("Answer is missing source citations")
        return {
            "supported": not warnings,
            "warnings": warnings,
            "prices_found": sorted(found_prices),
            "citation_present": "Sources:" in answer,
        }
