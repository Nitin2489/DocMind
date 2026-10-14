"""SQLite persistence for documents and chunks (with embeddings as BLOBs)."""
import sqlite3
import threading
import time
from dataclasses import dataclass

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    idx INTEGER NOT NULL,
    page INTEGER NOT NULL,
    text TEXT NOT NULL,
    embedding BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
"""


@dataclass
class ChunkRow:
    id: int
    doc_id: int
    doc_name: str
    idx: int
    page: int
    text: str
    embedding: np.ndarray


class Store:
    def __init__(self, path: str):
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.executescript(SCHEMA)
        self.version = 0  # bumped on every write so the search index knows to rebuild
    def close(self) -> None:
        """Close the SQLite connection and release its file handle."""
        with self._lock:
            self._db.close()

    def add_document(self, name: str, chunks: list[tuple[int, str]], embeddings: np.ndarray) -> int:
        with self._lock, self._db:
            cur = self._db.execute(
                "INSERT INTO documents (name, created_at) VALUES (?, ?)", (name, time.time())
            )
            doc_id = cur.lastrowid
            self._db.executemany(
                "INSERT INTO chunks (doc_id, idx, page, text, embedding) VALUES (?, ?, ?, ?, ?)",
                [
                    (doc_id, i, page, text, embeddings[i].astype(np.float32).tobytes())
                    for i, (page, text) in enumerate(chunks)
                ],
            )
            self.version += 1
        return doc_id

    def list_documents(self) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                """SELECT d.id, d.name, d.created_at, COUNT(c.id) AS chunks
                   FROM documents d LEFT JOIN chunks c ON c.doc_id = d.id
                   GROUP BY d.id ORDER BY d.id DESC"""
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_document(self, doc_id: int) -> bool:
        with self._lock, self._db:
            cur = self._db.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            if cur.rowcount:
                self.version += 1
            return cur.rowcount > 0

    def all_chunks(self) -> list[ChunkRow]:
        with self._lock:
            rows = self._db.execute(
                """SELECT c.id, c.doc_id, d.name AS doc_name, c.idx, c.page, c.text, c.embedding
                   FROM chunks c JOIN documents d ON d.id = c.doc_id ORDER BY c.id"""
            ).fetchall()
        return [
            ChunkRow(
                r["id"], r["doc_id"], r["doc_name"], r["idx"], r["page"], r["text"],
                np.frombuffer(r["embedding"], dtype=np.float32),
            )
            for r in rows
        ]
