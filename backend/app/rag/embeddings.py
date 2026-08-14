"""Pluggable embedding backends.

Two implementations, selected automatically:

1. `SentenceTransformerEmbedder` — BAAI/bge-small-en-v1.5. True semantic
   embeddings; used whenever `sentence-transformers` is installed.
2. `HashingEmbedder` — dependency-free deterministic fallback (hashed word
   n-grams + character n-grams, TF-IDF weighted, L2 normalised).

The fallback exists so the project runs end-to-end after a 30-second
`pip install`, with no 2 GB PyTorch download and no network access. It is
genuinely weaker at paraphrase matching, which is stated plainly in the README
rather than hidden — but because retrieval is *hybrid* and gated by a score
threshold, the degradation shows up as more abstentions, not as more
hallucinations. That is the correct failure direction for this system.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from abc import ABC, abstractmethod

import numpy as np

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokenizer that preserves telecom identifiers.

    `5G-AKA`, `N3IWF`, `TS 23.501`, `5QI` must survive tokenization intact —
    splitting them destroys exactly the terms that make a spec query precise.
    """
    return _TOKEN_RE.findall(text.lower())


class BaseEmbedder(ABC):
    name: str = "base"
    dimension: int = 0

    @abstractmethod
    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        """Return an (n, dim) float32 array of L2-normalised row vectors."""

    def encode_one(self, text: str, is_query: bool = False) -> np.ndarray:
        return self.encode([text], is_query=is_query)[0]


# ---------------------------------------------------------------------------
# Fallback: hashed n-gram TF-IDF
# ---------------------------------------------------------------------------

class HashingEmbedder(BaseEmbedder):
    """Deterministic, offline, dependency-free vectoriser.

    Combines word unigrams/bigrams (topical signal) with character 4-grams
    (morphological robustness: "registration" ~ "registered"). IDF weights are
    fitted on the corpus at index time and persisted with the index.
    """

    name = "hashing-tfidf"

    def __init__(self, dimension: int = 4096):
        self.dimension = dimension
        self._idf: dict[int, float] = {}
        self._default_idf: float = 1.0
        self._fitted = False

    # -- feature extraction -------------------------------------------------

    def _features(self, text: str) -> list[int]:
        tokens = tokenize(text)
        feats: list[str] = list(tokens)
        feats.extend(f"{a}_{b}" for a, b in zip(tokens, tokens[1:]))

        compact = re.sub(r"\s+", " ", text.lower())
        feats.extend(compact[i : i + 4] for i in range(0, max(0, len(compact) - 3), 2))

        return [self._hash(f) for f in feats]

    def _hash(self, feature: str) -> int:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "little") % self.dimension

    # -- IDF fitting --------------------------------------------------------

    def fit(self, texts: list[str]) -> None:
        doc_freq: dict[int, int] = {}
        for text in texts:
            for idx in set(self._features(text)):
                doc_freq[idx] = doc_freq.get(idx, 0) + 1

        n = max(len(texts), 1)
        self._idf = {
            idx: math.log((n + 1) / (df + 1)) + 1.0 for idx, df in doc_freq.items()
        }
        self._default_idf = math.log((n + 1) / 1) + 1.0
        self._fitted = True
        logger.info("HashingEmbedder fitted on %d documents (%d features)", n, len(self._idf))

    # -- encoding -----------------------------------------------------------

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        out = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            counts: dict[int, int] = {}
            for idx in self._features(text):
                counts[idx] = counts.get(idx, 0) + 1
            for idx, tf in counts.items():
                weight = (1.0 + math.log(tf)) * self._idf.get(idx, self._default_idf)
                out[row, idx] = weight
            norm = float(np.linalg.norm(out[row]))
            if norm > 0:
                out[row] /= norm
        return out

    # -- persistence --------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "dimension": self.dimension,
            "default_idf": self._default_idf,
            "idf_keys": list(self._idf.keys()),
            "idf_values": list(self._idf.values()),
        }

    def load_state_dict(self, state: dict) -> None:
        self.dimension = int(state["dimension"])
        self._default_idf = float(state["default_idf"])
        self._idf = dict(zip(state["idf_keys"], state["idf_values"]))
        self._fitted = True


# ---------------------------------------------------------------------------
# Preferred: sentence-transformers
# ---------------------------------------------------------------------------

class SentenceTransformerEmbedder(BaseEmbedder):
    """BGE-family bi-encoder. Requires the asymmetric query prefix to work well."""

    name = "bge-small-en-v1.5"
    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self.name = model_name
        self.dimension = int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        payload = [self.QUERY_PREFIX + t for t in texts] if is_query else texts
        vectors = self._model.encode(
            payload,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.astype(np.float32)

    def fit(self, texts: list[str]) -> None:  # noqa: D401 - interface parity
        """No-op: pretrained model needs no corpus fitting."""


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_embedder(prefer_neural: bool = True) -> BaseEmbedder:
    if prefer_neural:
        try:
            embedder = SentenceTransformerEmbedder()
            logger.info("Using neural embedder: %s (dim=%d)", embedder.name, embedder.dimension)
            return embedder
        except Exception as exc:  # ImportError, download failure, no network...
            logger.warning(
                "Neural embedder unavailable (%s). Falling back to hashed TF-IDF. "
                "Install requirements-ml.txt for better semantic recall.",
                exc.__class__.__name__,
            )
    return HashingEmbedder()
