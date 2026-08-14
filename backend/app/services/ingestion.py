"""Corpus ingestion: files on disk -> clause-aware chunks -> hybrid index."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from ..rag.chunking import Chunk, chunk_document
from ..rag.loaders import discover_documents, load_document
from ..rag.vector_store import HybridIndex

logger = logging.getLogger(__name__)


def build_index(
    corpus_dir: Path,
    index: HybridIndex,
    *,
    target_chars: int = 1400,
    overlap_chars: int = 200,
    min_chars: int = 120,
    progress=None,
) -> dict:
    """Ingest every supported file under `corpus_dir` and (re)build the index."""

    def report(message: str) -> None:
        logger.info(message)
        if progress:
            progress(message)

    started = time.perf_counter()
    paths = discover_documents(corpus_dir)
    if not paths:
        raise FileNotFoundError(
            f"No ingestible documents found in {corpus_dir}. "
            "Supported types: .pdf, .docx, .txt, .md"
        )

    all_chunks: list[Chunk] = []
    per_document: list[dict] = []
    failures: list[dict] = []

    for path in paths:
        try:
            report(f"Loading {path.name}")
            doc = load_document(path)
            chunks = chunk_document(
                doc,
                target_chars=target_chars,
                overlap_chars=overlap_chars,
                min_chars=min_chars,
            )
            if not chunks:
                failures.append({"file": path.name, "error": "No text extracted"})
                continue

            all_chunks.extend(chunks)
            per_document.append(
                {
                    "file": path.name,
                    "doc_id": doc.doc_id,
                    "title": doc.title,
                    "version": doc.version,
                    "release": doc.release,
                    "chunks": len(chunks),
                    "characters": len(doc.text),
                    "clauses": len({c.clause_id for c in chunks if c.clause_id}),
                }
            )
            report(f"  {doc.doc_id}: {len(chunks)} chunks")
        except Exception as exc:
            logger.exception("Failed to ingest %s", path.name)
            failures.append({"file": path.name, "error": str(exc)})

    if not all_chunks:
        raise RuntimeError("Ingestion produced zero chunks; nothing to index.")

    # Identical clause text across releases produces duplicate retrievals that
    # waste context and make an answer look better-corroborated than it is.
    seen: set[str] = set()
    deduped: list[Chunk] = []
    for chunk in all_chunks:
        if chunk.chunk_id in seen:
            continue
        seen.add(chunk.chunk_id)
        deduped.append(chunk)
    duplicates = len(all_chunks) - len(deduped)

    report(f"Indexing {len(deduped)} chunks from {len(per_document)} documents")
    index.build(deduped, progress=report)
    index.save()

    elapsed = round(time.perf_counter() - started, 2)
    report(f"Index built in {elapsed}s")

    return {
        "documents": per_document,
        "failures": failures,
        "chunk_count": len(deduped),
        "duplicates_removed": duplicates,
        "document_count": len(per_document),
        "embedder": index.embedder.name,
        "dimension": index.meta.get("dimension"),
        "elapsed_seconds": elapsed,
    }
