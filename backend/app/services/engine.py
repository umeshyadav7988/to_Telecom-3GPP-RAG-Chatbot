"""Process-wide singleton wiring the RAG components together.

Models and indices are expensive to construct and immutable once built, so
they are created once at import and shared. Rebuilds take an exclusive lock so
a `POST /api/documents/reindex` cannot swap the index out from under an
in-flight query.
"""

from __future__ import annotations

import logging
import threading

from config import settings

from ..rag.llm import build_client
from ..rag.pipeline import RAGPipeline
from ..rag.reranker import build_reranker
from ..rag.retriever import HybridRetriever
from ..rag.vector_store import HybridIndex
from .ingestion import build_index
from .store import Store

logger = logging.getLogger(__name__)


class Engine:
    def __init__(self):
        self._lock = threading.RLock()
        self.index = HybridIndex(settings.index_dir)
        self.reranker = build_reranker()
        self.store = Store(settings.db_path)
        self.client = build_client(
            settings.llm_provider, settings.api_key, settings.answer_model
        )
        self.last_build: dict = {}

        loaded = self.index.load()
        if not loaded and settings.auto_index_on_boot:
            # Serverless cold start: nothing was persisted, so rebuild from the
            # bundled corpus. Cheap enough (<1s for the bundled specs) to do on
            # every cold start, and it keeps the deployment stateless.
            try:
                logger.info("No index found; building from %s", settings.corpus_dir)
                self.last_build = build_index(
                    settings.corpus_dir,
                    self.index,
                    target_chars=settings.chunk_target_chars,
                    overlap_chars=settings.chunk_overlap_chars,
                    min_chars=settings.chunk_min_chars,
                )
                loaded = True
            except Exception:
                logger.exception("Boot-time index build failed")
        if not loaded:
            logger.warning(
                "No usable index at %s. Run `python scripts/ingest.py` or "
                "POST /api/documents/reindex.",
                settings.index_dir,
            )

        self.retriever = HybridRetriever(
            self.index,
            self.reranker,
            top_k=settings.retrieval_top_k,
            rerank_top_n=settings.rerank_top_n,
            rrf_k=settings.rrf_k,
            dense_weight=settings.dense_weight,
            sparse_weight=settings.sparse_weight,
            min_score=settings.min_retrieval_score,
        )
        self.pipeline = RAGPipeline(
            self.retriever,
            client=self.client,
            answer_model=settings.answer_model,
            verifier_model=settings.verifier_model,
            rewrite_model=settings.rewrite_model,
            enable_verifier=settings.enable_verifier,
            enable_numeric_guard=settings.enable_numeric_guard,
            enable_premise_guard=settings.enable_premise_guard,
            min_support_ratio=settings.min_support_ratio,
            max_answer_tokens=settings.max_answer_tokens,
            max_verifier_tokens=settings.max_verifier_tokens,
        )

    # -- lifecycle ----------------------------------------------------------

    def reindex(self, progress=None) -> dict:
        with self._lock:
            report = build_index(
                settings.corpus_dir,
                self.index,
                target_chars=settings.chunk_target_chars,
                overlap_chars=settings.chunk_overlap_chars,
                min_chars=settings.chunk_min_chars,
                progress=progress,
            )
            self.last_build = report
            return report

    # -- introspection ------------------------------------------------------

    def status(self) -> dict:
        return {
            "index_ready": self.index.is_ready,
            "chunk_count": self.index.size,
            "documents": self.index.document_summary(),
            "embedder": self.index.embedder.name,
            "reranker": self.reranker.name,
            "llm_enabled": self.client is not None,
            "mode": "generative" if self.client is not None else "extractive",
            "provider": self.client.provider if self.client else "none",
            # Capability flags the frontend adapts to, rather than assuming a
            # deployment shape and hanging when the assumption is wrong.
            "capabilities": {
                "streaming": settings.supports_streaming,
                "corpus_writable": not settings.read_only_filesystem,
                "persistent_history": not settings.is_serverless,
            },
            "models": {
                "answer": settings.answer_model,
                "verifier": settings.verifier_model,
                "rewrite": settings.rewrite_model,
            },
            "guardrails": {
                "min_retrieval_score": settings.min_retrieval_score,
                "min_support_ratio": settings.min_support_ratio,
                "verifier_enabled": settings.enable_verifier,
                "numeric_guard_enabled": settings.enable_numeric_guard,
                "premise_guard_enabled": settings.enable_premise_guard,
                "rerank_top_n": settings.rerank_top_n,
                "retrieval_top_k": settings.retrieval_top_k,
            },
            "index_meta": {k: v for k, v in self.index.meta.items() if k != "embedder_state"},
            "last_build": self.last_build,
        }


_engine: Engine | None = None
_engine_lock = threading.Lock()


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = Engine()
    return _engine
