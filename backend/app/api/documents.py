"""Corpus management: inspect, upload, delete and reindex specifications."""

from __future__ import annotations

import logging
import re

from flask import Blueprint, jsonify, request
from werkzeug.utils import secure_filename

from config import settings

from ..rag.loaders import SUPPORTED_SUFFIXES, discover_documents
from ..services.engine import get_engine

logger = logging.getLogger(__name__)
bp = Blueprint("documents", __name__, url_prefix="/api/documents")

MAX_UPLOAD_BYTES = 80 * 1024 * 1024  # a full 3GPP spec PDF is a few MB


@bp.get("")
def list_documents():
    engine = get_engine()
    on_disk = [
        {"filename": p.name, "size_bytes": p.stat().st_size, "indexed": False}
        for p in discover_documents(settings.corpus_dir)
    ]
    indexed = engine.index.document_summary()
    indexed_files = {d["source_file"] for d in indexed}
    for entry in on_disk:
        entry["indexed"] = entry["filename"] in indexed_files

    return jsonify(
        {
            "corpus_dir": str(settings.corpus_dir),
            "files": on_disk,
            "indexed_documents": indexed,
            "chunk_count": engine.index.size,
            "index_ready": engine.index.is_ready,
        }
    )


def _read_only_response():
    return (
        jsonify(
            {
                "error": "The corpus is read-only in this deployment.",
                "detail": (
                    "Serverless filesystems are read-only and reset on every cold "
                    "start. Add specifications to backend/data/corpus/ and redeploy, "
                    "or run the backend on a host with a persistent disk."
                ),
            }
        ),
        409,
    )


@bp.post("/upload")
def upload_document():
    if settings.read_only_filesystem:
        return _read_only_response()
    if "file" not in request.files:
        return jsonify({"error": "Attach a file under the `file` field."}), 400

    uploaded = request.files["file"]
    if not uploaded.filename:
        return jsonify({"error": "Empty filename."}), 400

    filename = secure_filename(uploaded.filename)
    # secure_filename can strip a name to nothing (e.g. all-CJK filenames).
    if not filename or filename.startswith("."):
        return jsonify({"error": "Invalid filename."}), 400

    suffix = ("." + filename.rsplit(".", 1)[-1]).lower() if "." in filename else ""
    if suffix not in SUPPORTED_SUFFIXES:
        return jsonify(
            {"error": f"Unsupported type {suffix!r}. Allowed: {sorted(SUPPORTED_SUFFIXES)}"}
        ), 415

    destination = settings.corpus_dir / filename
    # Resolve and confirm containment — defence in depth behind secure_filename.
    if settings.corpus_dir.resolve() not in destination.resolve().parents:
        return jsonify({"error": "Invalid destination path."}), 400

    uploaded.save(destination)
    size = destination.stat().st_size
    if size > MAX_UPLOAD_BYTES:
        destination.unlink(missing_ok=True)
        return jsonify({"error": f"File exceeds {MAX_UPLOAD_BYTES // (1024*1024)} MB."}), 413

    return jsonify(
        {
            "filename": filename,
            "size_bytes": size,
            "message": "Uploaded. Trigger POST /api/documents/reindex to make it searchable.",
        }
    ), 201


@bp.delete("/<path:filename>")
def delete_document(filename: str):
    if settings.read_only_filesystem:
        return _read_only_response()
    safe = secure_filename(filename)
    if not safe:
        return jsonify({"error": "Invalid filename."}), 400
    target = settings.corpus_dir / safe
    if not target.exists():
        return jsonify({"error": "Not found."}), 404
    target.unlink()
    return jsonify({"deleted": safe, "message": "Reindex to update the search index."})


@bp.post("/reindex")
def reindex():
    engine = get_engine()
    try:
        report = engine.reindex()
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:
        logger.exception("Reindex failed")
        return jsonify({"error": "Reindex failed.", "detail": str(exc)}), 500
    return jsonify(report)


@bp.get("/chunk/<chunk_id>")
def get_chunk(chunk_id: str):
    """Fetch one chunk's full text — backs the citation drill-down panel."""
    if not re.fullmatch(r"[0-9a-f]{6,64}", chunk_id):
        return jsonify({"error": "Invalid chunk id."}), 400

    engine = get_engine()
    for chunk in engine.index.chunks:
        if chunk.chunk_id == chunk_id:
            return jsonify(chunk.to_dict())
    return jsonify({"error": "Chunk not found."}), 404
