"""Conversation history and answer feedback."""

from __future__ import annotations

from flask import Blueprint, jsonify, request

from ..services.engine import get_engine

bp = Blueprint("conversations", __name__, url_prefix="/api")


@bp.get("/conversations")
def list_conversations():
    return jsonify({"conversations": get_engine().store.list_conversations()})


@bp.post("/conversations")
def create_conversation():
    body = request.get_json(silent=True) or {}
    conversation_id = get_engine().store.create_conversation(
        (body.get("title") or "New conversation")[:120]
    )
    return jsonify({"conversation_id": conversation_id}), 201


@bp.get("/conversations/<conversation_id>")
def get_conversation(conversation_id: str):
    turns = get_engine().store.get_turns(conversation_id)
    if not turns:
        return jsonify({"conversation_id": conversation_id, "turns": []})
    return jsonify({"conversation_id": conversation_id, "turns": turns})


@bp.delete("/conversations/<conversation_id>")
def delete_conversation(conversation_id: str):
    get_engine().store.delete_conversation(conversation_id)
    return jsonify({"deleted": conversation_id})


@bp.post("/feedback")
def submit_feedback():
    body = request.get_json(silent=True) or {}
    turn_id = (body.get("turn_id") or "").strip()
    rating = (body.get("rating") or "").strip().lower()

    if not turn_id:
        return jsonify({"error": "`turn_id` is required."}), 400
    if rating not in {"up", "down"}:
        return jsonify({"error": "`rating` must be 'up' or 'down'."}), 400

    feedback_id = get_engine().store.add_feedback(
        turn_id,
        rating,
        comment=(body.get("comment") or "")[:2000],
        confidence=body.get("confidence"),
        status=body.get("status"),
    )
    return jsonify({"feedback_id": feedback_id}), 201


@bp.get("/feedback/summary")
def feedback_summary():
    return jsonify(get_engine().store.feedback_summary())
