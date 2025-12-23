# crystal_brain/store.py
# EXTENDED with audio_snippets + triggers support.
from __future__ import annotations
import sqlite3
from pathlib import Path
from typing import List, Tuple, Dict, Any
import json
import numpy as np
from dataclasses import dataclass
try:
    from config import CONFIG  # type: ignore
except Exception:
    from ...config import CONFIG  # type: ignore

@dataclass
class MemoryRecord:
    id: int
    text: str
    embedding: np.ndarray
    created_at: float
    tags: Dict[str, Any]

@dataclass
class AudioSnippetRecord:
    id: int
    time: float
    path: str
    embedding: np.ndarray
    asr_text: str

@dataclass
class TriggerRecord:
    id: int
    snippet_id: int
    phrase: str
    threshold: float
    embedding: np.ndarray

class MemoryStore:
    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or CONFIG.db.db_path
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute("""
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    embedding BLOB NOT NULL,
    created_at REAL NOT NULL,
    tags TEXT NOT NULL
)""")
            cur.execute("""
CREATE TABLE IF NOT EXISTS energetics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    time REAL NOT NULL,
    H_bits REAL NOT NULL,
    S_field REAL NOT NULL,
    L REAL NOT NULL,
    coherence REAL NOT NULL,
    phi REAL NOT NULL
)""")
            cur.execute("""
CREATE TABLE IF NOT EXISTS voice_profile (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    time REAL NOT NULL,
    clarity REAL NOT NULL,
    stability REAL NOT NULL
)""")
            # NEW: audio snippets from child (including noise)
            cur.execute("""
CREATE TABLE IF NOT EXISTS audio_snippets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    time REAL NOT NULL,
    path TEXT NOT NULL,
    embedding BLOB NOT NULL,
    asr_text TEXT NOT NULL
)""")
            # NEW: guardian-labeled triggers
            cur.execute("""
CREATE TABLE IF NOT EXISTS triggers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snippet_id INTEGER NOT NULL,
    phrase TEXT NOT NULL,
    threshold REAL NOT NULL,
    embedding BLOB NOT NULL,
    FOREIGN KEY(snippet_id) REFERENCES audio_snippets(id)
)""")
            conn.commit()

    def store_memory(self, text: str, embedding: np.ndarray,
                     created_at: float, tags: Dict[str, Any]) -> int:
        emb_bytes = embedding.astype("float32").tobytes()
        tags_json = json.dumps(tags, ensure_ascii=False)
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute("""
INSERT INTO memories (text, embedding, created_at, tags)
VALUES (?, ?, ?, ?)
""", (text, emb_bytes, created_at, tags_json))
            mem_id = cur.lastrowid
            conn.commit()
            return int(mem_id)

    def load_recent_embeddings(self, limit: int = 128) -> np.ndarray:
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute("""
SELECT embedding FROM memories ORDER BY created_at DESC LIMIT ?
""", (limit,))
            rows = cur.fetchall()
            if not rows:
                return np.zeros((0, 64), dtype="float32")
            embs = []
            for (blob,) in rows:
                emb = np.frombuffer(blob, dtype="float32")
                embs.append(emb)
            return np.stack(embs, axis=0)

    def store_energetics(self, time: float, H_bits: float,
                         S_field: float, L: float,
                         coherence: float, phi: float) -> int:
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute("""
INSERT INTO energetics (time, H_bits, S_field, L, coherence, phi)
VALUES (?, ?, ?, ?, ?, ?)
""", (time, H_bits, S_field, L, coherence, phi))
            eid = cur.lastrowid
            conn.commit()
            return int(eid)

    def get_recent_energetics(self, limit: int = 256) -> list[tuple]:
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute("""
SELECT time, H_bits, S_field, L, coherence, phi
FROM energetics ORDER BY time DESC LIMIT ?
""", (limit,))
            return cur.fetchall()

    # voice_profile
    def store_voice_profile(self, time: float,
                            clarity: float,
                            stability: float) -> int:
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute("""
INSERT INTO voice_profile (time, clarity, stability)
VALUES (?, ?, ?)
""", (time, float(clarity), float(stability)))
            vid = cur.lastrowid
            conn.commit()
            return int(vid)

    def get_voice_profile_series(self, limit: int = 512) -> list[tuple]:
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute("""
SELECT time, clarity, stability
FROM voice_profile
ORDER BY time DESC
LIMIT ?
""", (limit,))
            return cur.fetchall()

    # NEW: audio_snippets
    def store_audio_snippet(self,
                            time: float,
                            path: str,
                            embedding: np.ndarray,
                            asr_text: str) -> int:
        emb_bytes = embedding.astype("float32").tobytes()
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute("""
INSERT INTO audio_snippets (time, path, embedding, asr_text)
VALUES (?, ?, ?, ?)
""", (time, path, emb_bytes, asr_text))
            sid = cur.lastrowid
            conn.commit()
            return int(sid)

    def get_audio_snippet(self, snippet_id: int) -> AudioSnippetRecord:
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute("""
SELECT id, time, path, embedding, asr_text
FROM audio_snippets WHERE id = ?
""", (snippet_id,))
            row = cur.fetchone()
            if row is None:
                raise KeyError(f"audio_snippets id {snippet_id} not found")
            emb = np.frombuffer(row[3], dtype="float32")
            return AudioSnippetRecord(
                id=row[0],
                time=row[1],
                path=row[2],
                embedding=emb,
                asr_text=row[4],
            )

    def list_unlabeled_snippets(self, limit: int = 64) -> list[AudioSnippetRecord]:
        """
        Return recent snippets that do not yet have triggers pointing to them.
        Useful for the guardian UI to label noises.
        """
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute("""
SELECT s.id, s.time, s.path, s.embedding, s.asr_text
FROM audio_snippets s
LEFT JOIN triggers t ON t.snippet_id = s.id
WHERE t.id IS NULL
ORDER BY s.time DESC
LIMIT ?
""", (limit,))
            rows = cur.fetchall()
            recs: list[AudioSnippetRecord] = []
            for row in rows:
                emb = np.frombuffer(row[3], dtype="float32")
                recs.append(AudioSnippetRecord(
                    id=row[0],
                    time=row[1],
                    path=row[2],
                    embedding=emb,
                    asr_text=row[4],
                ))
            return recs

    # NEW: triggers
    def store_trigger(self,
                      snippet_id: int,
                      phrase: str,
                      threshold: float,
                      embedding: np.ndarray) -> int:
        emb_bytes = embedding.astype("float32").tobytes()
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute("""
INSERT INTO triggers (snippet_id, phrase, threshold, embedding)
VALUES (?, ?, ?, ?)
""", (snippet_id, phrase, threshold, emb_bytes))
            tid = cur.lastrowid
            conn.commit()
            return int(tid)

    def get_triggers(self) -> list[TriggerRecord]:
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute("""
SELECT id, snippet_id, phrase, threshold, embedding
FROM triggers
""")
            rows = cur.fetchall()
            triggers: list[TriggerRecord] = []
            for row in rows:
                emb = np.frombuffer(row[4], dtype="float32")
                triggers.append(TriggerRecord(
                    id=row[0],
                    snippet_id=row[1],
                    phrase=row[2],
                    threshold=row[3],
                    embedding=emb,
                ))
            return triggers
