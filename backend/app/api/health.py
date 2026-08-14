"""Liveness, readiness and full system status."""

from __future__ import annotations

from flask import Blueprint, jsonify

from ..services.engine import get_engine

bp = Blueprint("health", __name__, url_prefix="/api")


@bp.get("/health")
def health():
    """Cheap liveness probe — never touches the index or the LLM."""
    return jsonify({"status": "ok", "service": "telecom-rag-backend"})


@bp.get("/status")
def status():
    """Full readiness report: index, models, and every guardrail setting."""
    engine = get_engine()
    payload = engine.status()
    payload["ready"] = payload["index_ready"]
    if not payload["index_ready"]:
        payload["hint"] = (
            "No index found. Run `python scripts/ingest.py` from the backend "
            "directory, or POST /api/documents/reindex."
        )
    return jsonify(payload)
