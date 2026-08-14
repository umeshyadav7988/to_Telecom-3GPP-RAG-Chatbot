"""BM25 Okapi, implemented directly (no `rank_bm25` dependency).

Lexical retrieval is not optional for 3GPP. Queries are full of exact
identifiers — `5QI 82`, `N3IWF`, `T3512`, `TS 33.501`, `RRC_INACTIVE` — where
a dense embedder's nearest neighbours are semantically adjacent but factually
wrong (returning the 5QI 83 row instead of 5QI 82 is a hallucination waiting
to happen). BM25 nails exact-term matching; the dense retriever covers
paraphrase. Fused, they cover each other's blind spots.
"""

from __future__ import annotations

import math
from collections import Counter

from .embeddings import tokenize

# Extremely common in 3GPP text; carry no discriminative signal.
_DOMAIN_STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "for", "and", "or", "is", "are", "be",
    "as", "by", "with", "that", "this", "it", "if", "on", "at", "from", "shall",
    "may", "can", "which", "when", "then", "there", "these", "such", "was",
}


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_len: list[int] = []
        self.avg_doc_len: float = 0.0
        self.term_freqs: list[dict[str, int]] = []
        self.doc_freq: dict[str, int] = {}
        self.idf: dict[str, float] = {}
        self.n_docs: int = 0

    # -- build --------------------------------------------------------------

    def fit(self, documents: list[str]) -> None:
        self.term_freqs = []
        self.doc_len = []
        self.doc_freq = {}

        for doc in documents:
            tokens = [t for t in tokenize(doc) if t not in _DOMAIN_STOPWORDS]
            counts = Counter(tokens)
            self.term_freqs.append(dict(counts))
            self.doc_len.append(len(tokens))
            for term in counts:
                self.doc_freq[term] = self.doc_freq.get(term, 0) + 1

        self.n_docs = len(documents)
        self.avg_doc_len = (sum(self.doc_len) / self.n_docs) if self.n_docs else 0.0

        # Robertson/Sparck-Jones IDF with the +0.5 smoothing, floored at a small
        # positive value so ubiquitous terms contribute ~0 rather than negative.
        self.idf = {}
        for term, df in self.doc_freq.items():
            value = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)
            self.idf[term] = max(value, 1e-6)

    # -- query --------------------------------------------------------------

    def search(self, query: str, top_k: int = 20) -> list[tuple[int, float]]:
        if not self.n_docs:
            return []

        q_terms = [t for t in tokenize(query) if t not in _DOMAIN_STOPWORDS]
        if not q_terms:
            return []

        scores = [0.0] * self.n_docs
        for term in q_terms:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for doc_idx, freqs in enumerate(self.term_freqs):
                tf = freqs.get(term)
                if not tf:
                    continue
                length_norm = 1 - self.b + self.b * (self.doc_len[doc_idx] / (self.avg_doc_len or 1))
                scores[doc_idx] += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * length_norm)

        ranked = sorted(
            ((i, s) for i, s in enumerate(scores) if s > 0),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:top_k]

    # -- persistence --------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "k1": self.k1,
            "b": self.b,
            "doc_len": self.doc_len,
            "avg_doc_len": self.avg_doc_len,
            "term_freqs": self.term_freqs,
            "doc_freq": self.doc_freq,
            "idf": self.idf,
            "n_docs": self.n_docs,
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "BM25Index":
        index = cls(k1=state.get("k1", 1.5), b=state.get("b", 0.75))
        index.doc_len = state["doc_len"]
        index.avg_doc_len = state["avg_doc_len"]
        index.term_freqs = state["term_freqs"]
        index.doc_freq = state["doc_freq"]
        index.idf = state["idf"]
        index.n_docs = state["n_docs"]
        return index
