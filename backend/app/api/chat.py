"""Chat endpoints: blocking JSON, SSE streaming, and retrieval-only search."""

from __future__ import annotations

import json
import logging

from flask import Blueprint, Response, jsonify, request, stream_with_context

from ..services.engine import get_engine

logger = logging.getLogger(__name__)
bp = Blueprint("chat", __name__, url_prefix="/api")

MAX_QUESTION_CHARS = 2000


def _read_question() -> tuple[str | None, str | None, tuple | None]:
    """Return (question, conversation_id, error_response)."""
    body = request.get_json(silent=True) or {}
    question = (body.get("message") or body.get("question") or "").strip()
    conversation_id = body.get("conversation_id")

    if not question:
        return None, None, (jsonify({"error": "A non-empty `message` is required."}), 400)
    if len(question) > MAX_QUESTION_CHARS:
        return None, None, (
            jsonify({"error": f"Question exceeds {MAX_QUESTION_CHARS} characters."}),
            413,
        )
    return question, conversation_id, None


@bp.post("/chat")
def chat():
    """Blocking variant. Same payload the stream terminates with."""
    question, conversation_id, error = _read_question()
    if error:
        return error

    engine = get_engine()
    conversation_id = engine.store.ensure_conversation(conversation_id, question)
    history = engine.store.get_history(conversation_id)

    user_turn_id = engine.store.add_turn(conversation_id, "user", question)

    try:
        result = engine.pipeline.run(question, history)
    except Exception as exc:
        logger.exception("Pipeline failure")
        return jsonify({"error": "The pipeline failed while answering.", "detail": str(exc)}), 500

    assistant_turn_id = engine.store.add_turn(
        conversation_id, "assistant", result.get("answer", ""), payload=result
    )

    result["conversation_id"] = conversation_id
    result["turn_id"] = assistant_turn_id
    result["user_turn_id"] = user_turn_id
    return jsonify(result)


@bp.post("/chat/stream")
def chat_stream():
    """Server-Sent Events stream of pipeline stages, ending with `result`.

    Stages are streamed because the visible pipeline is the product: watching
    retrieval scores and verification verdicts arrive is what lets a user
    calibrate how much to trust the answer.
    """
    question, conversation_id, error = _read_question()
    if error:
        return error

    engine = get_engine()
    conversation_id = engine.store.ensure_conversation(conversation_id, question)
    history = engine.store.get_history(conversation_id)
    engine.store.add_turn(conversation_id, "user", question)

    def event_stream():
        def sse(event: str, data: dict) -> str:
            return f"event: {event}\ndata: {json.dumps(data)}\n\n"

        yield sse("open", {"conversation_id": conversation_id})
        try:
            for event in engine.pipeline.run_stream(question, history):
                name, data = event["event"], event["data"]
                if name == "result":
                    turn_id = engine.store.add_turn(
                        conversation_id, "assistant", data.get("answer", ""), payload=data
                    )
                    data["conversation_id"] = conversation_id
                    data["turn_id"] = turn_id
                yield sse(name, data)
        except Exception as exc:
            logger.exception("Streaming pipeline failure")
            yield sse("error", {"message": "The pipeline failed while answering.", "detail": str(exc)})
        finally:
            yield sse("done", {"conversation_id": conversation_id})

    return Response(
        stream_with_context(event_stream()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",   # stop nginx from buffering the stream
            "Connection": "keep-alive",
        },
    )


@bp.post("/search")
def search():
    """Retrieval only — no generation. Useful for tuning and for the eval harness."""
    body = request.get_json(silent=True) or {}
    query = (body.get("query") or "").strip()
    if not query:
        return jsonify({"error": "A non-empty `query` is required."}), 400

    top_n = min(int(body.get("top_n") or 8), 25)
    engine = get_engine()
    result = engine.retriever.retrieve(query, top_n=top_n)

    return jsonify(
        {
            "query": query,
            "passed_gate": result.passed_gate,
            "top_score": round(result.top_score, 4),
            "gate_threshold": round(result.gate_threshold, 4),
            "candidate_count": result.candidate_count,
            "filters_applied": result.filters_applied,
            "timings_ms": result.timings_ms,
            "results": [item.to_dict() for item in result.chunks],
        }
    )
