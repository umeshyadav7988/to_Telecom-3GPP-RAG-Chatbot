"""Persisted hybrid index: chunks + dense vectors + BM25.

Storage layout under `data/index/`:

    chunks.json.gz      all Chunk records (source of truth for citations)
    vectors.npy         (n_chunks, dim) float32, L2-normalised
    bm25.json.gz        BM25 term statistics
    meta.json           embedder name/dim, counts, build timestamp

Dense search is exact cosine via a single matmul. At the scale of a spec
corpus (tens of thousands of chunks) that is sub-millisecond and avoids an ANN
index's recall loss — a recall miss here becomes an abstention or, worse, an
answer grounded in the second-best clause. FAISS is used automatically if
installed and the corpus is large enough to warrant it.
"""

from __future__ import annotations

import gzip
import json
import logging
import time
from dataclasses import fields as dataclass_fields
from pathlib import Path

import numpy as np

from .bm25 import BM25Index
from .chunking import Chunk
from .embeddings import BaseEmbedder, HashingEmbedder, build_embedder

logger = logging.getLogger(__name__)

FAISS_THRESHOLD = 50_000  # below this, brute force is faster than index build


def _write_gzip_json(path: Path, payload) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh)


def _read_gzip_json(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


class HybridIndex:
    """Owns the corpus, its vectors and its lexical statistics."""

    def __init__(self, index_dir: Path, embedder: BaseEmbedder | None = None):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder or build_embedder()
        self.chunks: list[Chunk] = []
        self.vectors: np.ndarray | None = None
        self.bm25 = BM25Index()
        self.meta: dict = {}
        self._faiss = None

    # -- properties ---------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        return bool(self.chunks) and self.vectors is not None

    @property
    def size(self) -> int:
        return len(self.chunks)

    def document_summary(self) -> list[dict]:
        """Per-specification statistics, for the frontend's corpus panel."""
        by_doc: dict[str, dict] = {}
        for chunk in self.chunks:
            entry = by_doc.setdefault(
                chunk.doc_id,
                {
                    "doc_id": chunk.doc_id,
                    "title": chunk.doc_title,
                    "version": chunk.version,
                    "release": chunk.release,
                    "chunks": 0,
                    "clauses": set(),
                    "normative_chunks": 0,
                    "source_file": Path(chunk.source_path).name,
                },
            )
            entry["chunks"] += 1
            if chunk.clause_id:
                entry["clauses"].add(chunk.clause_id)
            if chunk.is_normative:
                entry["normative_chunks"] += 1

        summary = []
        for entry in by_doc.values():
            entry["clauses"] = len(entry["clauses"])
            summary.append(entry)
        return sorted(summary, key=lambda e: e["doc_id"])

    # -- build --------------------------------------------------------------

    def build(self, chunks: list[Chunk], progress=None) -> None:
        if not chunks:
            raise ValueError("Cannot build an index from zero chunks")

        self.chunks = chunks
        texts = [c.text for c in chunks]

        if progress:
            progress("Fitting lexical statistics")
        if hasattr(self.embedder, "fit"):
            self.embedder.fit(texts)
        self.bm25.fit(texts)

        if progress:
            progress(f"Embedding {len(texts)} chunks with {self.embedder.name}")
        batch = 256
        blocks = []
        for i in range(0, len(texts), batch):
            blocks.append(self.embedder.encode(texts[i : i + batch], is_query=False))
            if progress:
                progress(f"Embedded {min(i + batch, len(texts))}/{len(texts)}")
        self.vectors = np.vstack(blocks).astype(np.float32)

        self.meta = {
            "embedder": self.embedder.name,
            "dimension": int(self.vectors.shape[1]),
            "chunk_count": len(chunks),
            "document_count": len({c.doc_id for c in chunks}),
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        self._maybe_build_faiss()

    def _maybe_build_faiss(self) -> None:
        if self.vectors is None or len(self.chunks) < FAISS_THRESHOLD:
            self._faiss = None
            return
        try:
            import faiss

            index = faiss.IndexFlatIP(self.vectors.shape[1])
            index.add(self.vectors)
            self._faiss = index
            logger.info("FAISS index active (%d vectors)", index.ntotal)
        except ImportError:
            self._faiss = None

    # -- persistence --------------------------------------------------------

    def save(self) -> None:
        if self.vectors is None:
            raise RuntimeError("Nothing to save - build the index first")

        _write_gzip_json(
            self.index_dir / "chunks.json.gz", [c.to_dict() for c in self.chunks]
        )
        np.save(self.index_dir / "vectors.npy", self.vectors)
        _write_gzip_json(self.index_dir / "bm25.json.gz", self.bm25.state_dict())

        meta = dict(self.meta)
        if isinstance(self.embedder, HashingEmbedder):
            meta["embedder_state"] = self.embedder.state_dict()
        (self.index_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        logger.info("Index saved to %s (%d chunks)", self.index_dir, len(self.chunks))

    def load(self) -> bool:
        chunks_path = self.index_dir / "chunks.json.gz"
        vectors_path = self.index_dir / "vectors.npy"
        meta_path = self.index_dir / "meta.json"
        if not (chunks_path.exists() and vectors_path.exists() and meta_path.exists()):
            return False

        meta = json.loads(meta_path.read_text(encoding="utf-8"))

        # An index built with a different embedder is unusable: the query would
        # be projected into a different space than the documents. Rebuild
        # rather than silently return garbage neighbours.
        if meta.get("embedder") != self.embedder.name:
            logger.warning(
                "Index was built with embedder %r but %r is active - rebuild required.",
                meta.get("embedder"),
                self.embedder.name,
            )
            return False

        if isinstance(self.embedder, HashingEmbedder) and "embedder_state" in meta:
            self.embedder.load_state_dict(meta["embedder_state"])

        valid = {f.name for f in dataclass_fields(Chunk)}
        self.chunks = [
            Chunk(**{k: v for k, v in raw.items() if k in valid})
            for raw in _read_gzip_json(chunks_path)
        ]
        self.vectors = np.load(vectors_path).astype(np.float32)
        self.bm25 = BM25Index.from_state_dict(_read_gzip_json(self.index_dir / "bm25.json.gz"))
        self.meta = meta
        self._maybe_build_faiss()
        logger.info("Index loaded: %d chunks, embedder=%s", len(self.chunks), self.embedder.name)
        return True

    # -- search -------------------------------------------------------------

    def dense_search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        if self.vectors is None or not self.chunks:
            return []
        q = self.embedder.encode_one(query, is_query=True).astype(np.float32)

        if self._faiss is not None:
            scores, ids = self._faiss.search(q.reshape(1, -1), min(top_k, len(self.chunks)))
            return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]

        sims = self.vectors @ q  # both sides are L2-normalised => cosine
        k = min(top_k, sims.shape[0])
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [(int(i), float(sims[i])) for i in top]

    def sparse_search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        return self.bm25.search(query, top_k=top_k)

    def get(self, idx: int) -> Chunk:
        return self.chunks[idx]
