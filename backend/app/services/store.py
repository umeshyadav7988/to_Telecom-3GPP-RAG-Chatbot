"""SQLite persistence for conversations, turns and user feedback.

Feedback is stored alongside the full pipeline trace (retrieval scores,
verification verdicts, confidence). A thumbs-down on a high-confidence answer
is the single most valuable signal this system can collect: it points directly
at a calibration failure, and the stored trace says which stage produced it.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from pathlib import Path

_LOCAL = threading.local()

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL DEFAULT 'New conversation',
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS turns (
    id               TEXT PRIMARY KEY,
    conversation_id  TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role             TEXT NOT NULL,
    content          TEXT NOT NULL,
    payload          TEXT,
    created_at       TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_turns_conversation ON turns(conversation_id, created_at);

CREATE TABLE IF NOT EXISTS feedback (
    id          TEXT PRIMARY KEY,
    turn_id     TEXT NOT NULL,
    rating      TEXT NOT NULL,
    comment     TEXT,
    confidence  REAL,
    status      TEXT,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


class Store:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        # One connection per thread: Flask's dev server and gunicorn workers
        # both hand requests to different threads, and SQLite objects are not
        # safe to share across them.
        conn = getattr(_LOCAL, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")
            _LOCAL.conn = conn
        return conn

    # -- conversations ------------------------------------------------------

    def create_conversation(self, title: str = "New conversation") -> str:
        conversation_id = uuid.uuid4().hex
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO conversations (id, title) VALUES (?, ?)",
                (conversation_id, title[:120]),
            )
        return conversation_id

    def ensure_conversation(self, conversation_id: str | None, title: str = "") -> str:
        if conversation_id:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT id FROM conversations WHERE id = ?", (conversation_id,)
                ).fetchone()
            if row:
                return conversation_id
        return self.create_conversation(title or "New conversation")

    def list_conversations(self, limit: int = 50) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT c.id, c.title, c.created_at, c.updated_at,
                       (SELECT COUNT(*) FROM turns t WHERE t.conversation_id = c.id) AS turn_count
                FROM conversations c
                ORDER BY c.updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_conversation(self, conversation_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM turns WHERE conversation_id = ?", (conversation_id,))
            conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))

    def rename_conversation(self, conversation_id: str, title: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE conversations SET title = ?, updated_at = datetime('now') WHERE id = ?",
                (title[:120], conversation_id),
            )

    # -- turns --------------------------------------------------------------

    def add_turn(
        self,
        conversation_id: str,
        role: str,
        content: str,
        payload: dict | None = None,
    ) -> str:
        turn_id = uuid.uuid4().hex
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO turns (id, conversation_id, role, content, payload) "
                "VALUES (?, ?, ?, ?, ?)",
                (turn_id, conversation_id, role, content, json.dumps(payload) if payload else None),
            )
            conn.execute(
                "UPDATE conversations SET updated_at = datetime('now') WHERE id = ?",
                (conversation_id,),
            )
            # First user message becomes the conversation title.
            if role == "user":
                row = conn.execute(
                    "SELECT title FROM conversations WHERE id = ?", (conversation_id,)
                ).fetchone()
                if row and row["title"] == "New conversation":
                    conn.execute(
                        "UPDATE conversations SET title = ? WHERE id = ?",
                        (content[:80], conversation_id),
                    )
        return turn_id

    def get_turns(self, conversation_id: str, limit: int = 100) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, role, content, payload, created_at FROM turns "
                "WHERE conversation_id = ? ORDER BY created_at ASC, rowid ASC LIMIT ?",
                (conversation_id, limit),
            ).fetchall()

        turns = []
        for row in rows:
            turn = dict(row)
            if turn.get("payload"):
                try:
                    turn["payload"] = json.loads(turn["payload"])
                except json.JSONDecodeError:
                    turn["payload"] = None
            turns.append(turn)
        return turns

    def get_history(self, conversation_id: str, max_turns: int = 12) -> list[dict]:
        """Lightweight role/content pairs for query contextualisation."""
        turns = self.get_turns(conversation_id, limit=max_turns * 2)
        return [{"role": t["role"], "content": t["content"]} for t in turns[-max_turns:]]

    # -- feedback -----------------------------------------------------------

    def add_feedback(
        self,
        turn_id: str,
        rating: str,
        comment: str = "",
        confidence: float | None = None,
        status: str | None = None,
    ) -> str:
        feedback_id = uuid.uuid4().hex
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO feedback (id, turn_id, rating, comment, confidence, status) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (feedback_id, turn_id, rating, comment[:2000], confidence, status),
            )
        return feedback_id

    def feedback_summary(self) -> dict:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT rating, COUNT(*) AS n, AVG(confidence) AS avg_confidence "
                "FROM feedback GROUP BY rating"
            ).fetchall()
        return {
            r["rating"]: {
                "count": r["n"],
                "avg_confidence": round(r["avg_confidence"], 3) if r["avg_confidence"] else None,
            }
            for r in rows
        }
