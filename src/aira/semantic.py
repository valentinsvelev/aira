"""Local embedding cosine similarity: a useful heuristic, NOT an entailment test.

The comparison is deliberately between an immutable *source quote* and the
proposed edited claim, not a whole PDF (which a small embedding model truncates).
"""
import re
from functools import lru_cache
from math import sqrt

from aira.context import Verdict, normalize


@lru_cache(maxsize=2)
def load_encoder(model_name: str):
    """Load lazily so ingestion/search need no embedding model or network."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            'Semantic checking requires: uv pip install -e ".[local]"'
        ) from exc
    return SentenceTransformer(model_name)


def cosine_similarity(a, b) -> float:
    """Compute cosine from two 1D embedding vectors; no numpy dependency here."""
    if len(a) != len(b) or len(a) == 0:
        raise ValueError("Embeddings must be nonempty vectors of equal length")
    dot = sum(float(x) * float(y) for x, y in zip(a, b))
    norm_a = sqrt(sum(float(x) ** 2 for x in a))
    norm_b = sqrt(sum(float(y) ** 2 for y in b))
    if norm_a == 0 or norm_b == 0:
        raise ValueError("Zero-norm embedding")
    return max(-1.0, min(1.0, dot / (norm_a * norm_b)))


class SemanticChecker:
    """Accept paraphrases above a configurable cosine threshold.

    Still demands an exact quote from the immutable evidence store and performs
    a minimal numeric check. A high similarity does NOT prove factual entailment.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 threshold: float = 0.78, encoder=None):
        if not -1.0 <= threshold <= 1.0:
            raise ValueError("Similarity threshold must be between -1 and 1")
        self.model_name = model_name
        self.threshold = threshold
        self._encoder = encoder

    def compare(self, original: str, edited: str) -> float:
        if not original.strip() or not edited.strip():
            raise ValueError("Both original and edited texts must be nonempty")
        # MiniLM truncates long inputs. Force source/claim sized excerpts.
        if len(original.split()) > 100 or len(edited.split()) > 100:
            raise ValueError("Compare passages of at most 100 words each; use a shorter source quote")
        encoder = self._encoder or load_encoder(self.model_name)
        vectors = encoder.encode([original, edited])
        return cosine_similarity(vectors[0], vectors[1])

    def check(self, claim: str, quote: str, evidence: str) -> Verdict:
        c, q, e = normalize(claim), normalize(quote), normalize(evidence)
        if not c or not q or not e:
            return Verdict(False, "Claim, quote and evidence must all be nonempty",
                           method="semantic_cosine")
        if len(c) < 20:
            return Verdict(False, "Claim must contain at least 20 normalized characters",
                           method="semantic_cosine")
        if q not in e:
            return Verdict(False, "Supplied source quote not found in immutable evidence",
                           method="semantic_cosine")
        # A simple guard against some high-similarity but numerically wrong rewrites.
        numbers_in_claim = set(re.findall(r"\d+(?:[.,]\d+)?", c))
        numbers_in_source = set(re.findall(r"\d+(?:[.,]\d+)?", q))
        if not numbers_in_claim.issubset(numbers_in_source):
            return Verdict(False, "Edited claim introduces number(s) absent from the source quote",
                           method="semantic_cosine")
        try:
            score = self.compare(quote, claim)
        except ValueError as exc:
            return Verdict(False, str(exc), method="semantic_cosine")
        accepted = score >= self.threshold
        reason = (f"Semantic cosine={score:.4f}, threshold={self.threshold:.4f}; "
                  + ("heuristic similarity check passed (NOT proof of entailment)" if accepted
                     else "below threshold; review or select a better source quote"))
        return Verdict(accepted, reason, similarity=round(score, 6), method="semantic_cosine")
