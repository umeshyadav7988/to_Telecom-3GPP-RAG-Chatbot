"""Evaluation endpoints — run the golden set from the UI.

Exposed over HTTP (not just as a CLI script) because "how do you know it does
not hallucinate?" is the question this project exists to answer, and the answer
should be one click away rather than buried in a terminal transcript.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from flask import Blueprint, Response, jsonify, request, stream_with_context

from config import BASE_DIR

from ..services.engine import get_engine

# The eval harness lives beside the app rather than inside it, so it stays
# usable as a standalone CLI. Make it importable regardless of the working
# directory the server was launched from (gunicorn often differs from `flask run`).
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

logger = logging.getLogger(__name__)
bp = Blueprint("evaluation", __name__, url_prefix="/api/evaluation")

GOLDEN_SET_PATH = BASE_DIR / "eval" / "golden_set.json"


def _load_golden_set() -> list[dict]:
    if not GOLDEN_SET_PATH.exists():
        return []
    data = json.loads(Path(GOLDEN_SET_PATH).read_text(encoding="utf-8"))
    return data.get("cases", [])


@bp.get("/golden-set")
def golden_set():
    cases = _load_golden_set()
    return jsonify(
        {
            "path": str(GOLDEN_SET_PATH),
            "count": len(cases),
            "categories": sorted({c.get("category", "uncategorised") for c in cases}),
            "cases": cases,
        }
    )


@bp.post("/run")
def run_evaluation():
    """Stream evaluation progress as SSE, ending with aggregate metrics."""
    body = request.get_json(silent=True) or {}
    limit = body.get("limit")
    categories = set(body.get("categories") or [])
    retrieval_only = bool(body.get("retrieval_only", False))

    cases = _load_golden_set()
    if categories:
        cases = [c for c in cases if c.get("category") in categories]
    if limit:
        cases = cases[: int(limit)]

    if not cases:
        return jsonify({"error": "No evaluation cases matched."}), 404

    engine = get_engine()

    def stream():
        from eval.metrics import aggregate, score_case  # local import: eval/ is not a package dep

        def sse(event: str, data: dict) -> str:
            return f"event: {event}\ndata: {json.dumps(data)}\n\n"

        yield sse("start", {"total": len(cases), "retrieval_only": retrieval_only})
        results = []
        for i, case in enumerate(cases, start=1):
            try:
                scored = score_case(engine, case, retrieval_only=retrieval_only)
            except Exception as exc:
                logger.exception("Evaluation case failed: %s", case.get("id"))
                scored = {
                    "id": case.get("id"),
                    "question": case.get("question"),
                    "error": str(exc),
                    "passed": False,
                }
            results.append(scored)
            yield sse("case", {"index": i, "total": len(cases), "result": scored})

        yield sse("summary", aggregate(results))
        yield sse("done", {})

    return Response(
        stream_with_context(stream()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
    )
