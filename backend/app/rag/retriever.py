"""Hybrid retrieval: dense + BM25 -> Reciprocal Rank Fusion -> rerank -> gate.

The gate is the important part. Most RAG hallucinations are not generation
failures at all — they are retrieval failures that the generator dutifully
papers over. If the corpus does not contain the answer, the only correct
behaviour is to say so, and the cheapest place to decide that is *before* the
LLM is invoked.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from .chunking import Chunk
from .reranker import BaseReranker, build_reranker
from .vector_store import HybridIndex

logger = logging.getLogger(__name__)

# Users cite specs directly ("what does 23.501 say about...") — honour that as
# a hard filter rather than hoping the retriever ranks the right document first.
_DOC_FILTER_RE = re.compile(r"\b(?:TS|TR)\s?(\d{2})[.\s]?(\d{3})\b", re.IGNORECASE)
_CLAUSE_FILTER_RE = re.compile(r"(?:clause|section|§)\s*(\d{1,2}(?:\.\d{1,3})+)", re.IGNORECASE)


@dataclass
class RetrievedChunk:
    """A candidate passage plus every score that produced its position."""

    chunk: Chunk
    score: float                      # final reranker score, [0, 1]
    dense_rank: int | None = None
    sparse_rank: int | None = None
    dense_score: float | None = None
    sparse_score: float | None = None
    fusion_score: float = 0.0
    explain: dict = field(default_factory=dict)
    source_index: int = 0             # 1-based [S1], [S2]... shown to the LLM

    def to_dict(self, include_text: bool = True) -> dict:
        payload = {
            "source_index": self.source_index,
            "citation_key": f"S{self.source_index}",
            "chunk_id": self.chunk.chunk_id,
            "doc_id": self.chunk.doc_id,
            "doc_title": self.chunk.doc_title,
            "version": self.chunk.version,
            "release": self.chunk.release,
            "clause_id": self.chunk.clause_id,
            "clause_title": self.chunk.clause_title,
            "breadcrumb": self.chunk.breadcrumb,
            "citation_label": self.chunk.citation_label,
            "page": self.chunk.page,
            "is_normative": self.chunk.is_normative,
            "has_table": self.chunk.has_table,
            "scores": {
                "final": round(self.score, 4),
                "fusion": round(self.fusion_score, 4),
                "dense": round(self.dense_score, 4) if self.dense_score is not None else None,
                "sparse": round(self.sparse_score, 4) if self.sparse_score is not None else None,
                "dense_rank": self.dense_rank,
                "sparse_rank": self.sparse_rank,
                "explain": self.explain,
            },
        }
        if include_text:
            payload["text"] = self.chunk.body
        return payload


@dataclass
class RetrievalResult:
    query: str
    effective_query: str
    chunks: list[RetrievedChunk]
    top_score: float
    passed_gate: bool
    gate_threshold: float
    filters_applied: dict = field(default_factory=dict)
    candidate_count: int = 0
    timings_ms: dict = field(default_factory=dict)


class HybridRetriever:
    def __init__(
        self,
        index: HybridIndex,
        reranker: BaseReranker | None = None,
        *,
        top_k: int = 24,
        rerank_top_n: int = 6,
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
        min_score: float = 0.28,
    ):
        self.index = index
        self.reranker = reranker or build_reranker()
        self.top_k = top_k
        self.rerank_top_n = rerank_top_n
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.min_score = min_score

    # -- query understanding -------------------------------------------------

    @staticmethod
    def _parse_filters(query: str) -> dict:
        filters: dict = {}
        docs = {
            f"{m.group(1)}.{m.group(2)}" for m in _DOC_FILTER_RE.finditer(query)
        }
        if docs:
            filters["doc_numbers"] = sorted(docs)
        clauses = {m.group(1) for m in _CLAUSE_FILTER_RE.finditer(query)}
        if clauses:
            filters["clauses"] = sorted(clauses)
        return filters

    def _passes_filters(self, chunk: Chunk, filters: dict) -> bool:
        if "doc_numbers" in filters:
            number = chunk.doc_id.split()[-1] if chunk.doc_id else ""
            if number not in filters["doc_numbers"]:
                return False
        if "clauses" in filters:
            # Prefix match so "clause 5.15" also admits 5.15.2.1.
            if not any(
                chunk.clause_id == c or chunk.clause_id.startswith(c + ".")
                for c in filters["clauses"]
            ):
                return False
        return True

    # -- fusion --------------------------------------------------------------

    def _fuse(self, dense: list, sparse: list) -> dict[int, dict]:
        """Weighted Reciprocal Rank Fusion.

        RRF over raw score interpolation because cosine similarity and BM25
        scores live on incomparable scales; normalising them per-query is
        unstable when one retriever returns nothing. Ranks are always
        comparable.
        """
        fused: dict[int, dict] = {}

        for rank, (idx, score) in enumerate(dense, start=1):
            entry = fused.setdefault(idx, {"fusion": 0.0})
            entry["fusion"] += self.dense_weight / (self.rrf_k + rank)
            entry["dense_rank"] = rank
            entry["dense_score"] = score

        for rank, (idx, score) in enumerate(sparse, start=1):
            entry = fused.setdefault(idx, {"fusion": 0.0})
            entry["fusion"] += self.sparse_weight / (self.rrf_k + rank)
            entry["sparse_rank"] = rank
            entry["sparse_score"] = score

        return fused

    # -- main ----------------------------------------------------------------

    def retrieve(self, query: str, *, top_n: int | None = None) -> RetrievalResult:
        import time

        top_n = top_n or self.rerank_top_n
        timings: dict[str, float] = {}
        filters = self._parse_filters(query)

        if not self.index.is_ready:
            return RetrievalResult(
                query=query,
                effective_query=query,
                chunks=[],
                top_score=0.0,
                passed_gate=False,
                gate_threshold=self.min_score,
                filters_applied=filters,
            )

        t0 = time.perf_counter()
        dense = self.index.dense_search(query, self.top_k)
        timings["dense_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        t0 = time.perf_counter()
        sparse = self.index.sparse_search(query, self.top_k)
        timings["sparse_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        fused = self._fuse(dense, sparse)

        # Apply hard filters, but never let them empty the candidate pool — a
        # user typo in a spec number should degrade to unfiltered search, not
        # to a false "not in the corpus".
        if filters:
            filtered = {
                idx: data
                for idx, data in fused.items()
                if self._passes_filters(self.index.get(idx), filters)
            }
            if filtered:
                fused = filtered
            else:
                filters["ignored"] = True

        ordered = sorted(fused.items(), key=lambda kv: kv[1]["fusion"], reverse=True)
        # Rerank a window wider than we will keep: the cross-encoder frequently
        # promotes a candidate that fusion placed 10th.
        window = ordered[: max(top_n * 4, 20)]

        candidates: list[tuple[Chunk, dict]] = [
            (self.index.get(idx), dict(data)) for idx, data in window
        ]

        t0 = time.perf_counter()
        scores = self.reranker.score(query, candidates)
        timings["rerank_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        scored = [
            RetrievedChunk(
                chunk=chunk,
                score=score,
                dense_rank=signals.get("dense_rank"),
                sparse_rank=signals.get("sparse_rank"),
                dense_score=signals.get("dense_score"),
                sparse_score=signals.get("sparse_score"),
                fusion_score=signals.get("fusion", 0.0),
                explain=signals.get("explain", {}),
            )
            for (chunk, signals), score in zip(candidates, scores)
        ]
        scored.sort(key=lambda r: r.score, reverse=True)

        kept = scored[:top_n]
        for position, item in enumerate(kept, start=1):
            item.source_index = position

        top_score = kept[0].score if kept else 0.0

        return RetrievalResult(
            query=query,
            effective_query=query,
            chunks=kept,
            top_score=top_score,
            passed_gate=top_score >= self.min_score,
            gate_threshold=self.min_score,
            filters_applied=filters,
            candidate_count=len(fused),
            timings_ms=timings,
        )
