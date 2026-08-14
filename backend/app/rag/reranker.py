"""Reranking — the precision stage, and the source of the abstention signal.

Fusion (RRF) produces a good *ordering* but its scores are rank-derived and
therefore uncalibrated: the top hit always scores ~1/(k+1) whether the corpus
answers the question or not. You cannot threshold on that, and a system that
cannot threshold cannot abstain.

So the reranker's job is twofold:
  1. reorder the fused candidates by true query-passage relevance, and
  2. emit a score in [0, 1] that means something absolute, so
     `MIN_RETRIEVAL_SCORE` can gate generation.

`CrossEncoderReranker` uses ms-marco-MiniLM when available.
`LexicalSemanticReranker` is the always-available fallback: an interpretable
blend of dense cosine, query-term coverage and clause-title match. Its
components are exposed in the API response so a reviewer can see exactly why a
passage was kept or the question refused.
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod

from .bm25 import _DOMAIN_STOPWORDS
from .chunking import Chunk
from .embeddings import tokenize

logger = logging.getLogger(__name__)


class BaseReranker(ABC):
    name = "base"

    @abstractmethod
    def score(self, query: str, candidates: list[tuple[Chunk, dict]]) -> list[float]:
        """Return one relevance score in [0, 1] per candidate."""


# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------

class LexicalSemanticReranker(BaseReranker):
    """Transparent, dependency-free relevance scorer.

    score = 0.45 * dense_cosine
          + 0.40 * query_term_coverage
          + 0.15 * clause_title_match

    Every term is bounded in [0, 1], so the total is too. Coverage is weighted
    heavily on purpose: in a specification corpus, a passage that does not
    mention the query's identifiers is almost never the right passage, however
    close it sits in embedding space.
    """

    name = "lexical-semantic"

    def score(self, query: str, candidates: list[tuple[Chunk, dict]]) -> list[float]:
        q_terms = {t for t in tokenize(query) if t not in _DOMAIN_STOPWORDS and len(t) > 1}
        if not q_terms:
            return [0.0] * len(candidates)

        # Rare, specific terms should count for more than common ones.
        weights = {t: (2.0 if (any(c.isdigit() for c in t) or len(t) <= 5) else 1.0) for t in q_terms}
        total_weight = sum(weights.values()) or 1.0

        scores: list[float] = []
        for chunk, signals in candidates:
            chunk_terms = set(tokenize(chunk.text))
            covered = sum(w for t, w in weights.items() if t in chunk_terms)
            coverage = covered / total_weight

            dense = signals.get("dense_score")
            # Cosine on normalised vectors is in [-1, 1]; map to [0, 1].
            dense_norm = ((dense + 1.0) / 2.0) if dense is not None else coverage

            title_terms = set(tokenize(f"{chunk.clause_title} {chunk.breadcrumb}"))
            title_match = (
                sum(w for t, w in weights.items() if t in title_terms) / total_weight
            )

            value = 0.45 * dense_norm + 0.40 * coverage + 0.15 * title_match
            signals["explain"] = {
                "dense_norm": round(dense_norm, 4),
                "term_coverage": round(coverage, 4),
                "title_match": round(title_match, 4),
            }
            scores.append(max(0.0, min(1.0, value)))

        return scores


# ---------------------------------------------------------------------------
# Preferred
# ---------------------------------------------------------------------------

class CrossEncoderReranker(BaseReranker):
    """ms-marco cross-encoder; logits squashed to [0, 1] with a sigmoid."""

    name = "ms-marco-MiniLM-L-6-v2"

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name, max_length=512)
        self.name = model_name

    def score(self, query: str, candidates: list[tuple[Chunk, dict]]) -> list[float]:
        if not candidates:
            return []
        pairs = [(query, chunk.text) for chunk, _ in candidates]
        logits = self._model.predict(pairs, show_progress_bar=False)
        out = []
        for logit, (_, signals) in zip(logits, candidates):
            prob = 1.0 / (1.0 + math.exp(-float(logit)))
            signals["explain"] = {"cross_encoder_logit": round(float(logit), 4)}
            out.append(prob)
        return out


def build_reranker(prefer_neural: bool = True) -> BaseReranker:
    if prefer_neural:
        try:
            reranker = CrossEncoderReranker()
            logger.info("Using cross-encoder reranker: %s", reranker.name)
            return reranker
        except Exception as exc:
            logger.warning(
                "Cross-encoder unavailable (%s); using lexical-semantic reranker.",
                exc.__class__.__name__,
            )
    return LexicalSemanticReranker()
