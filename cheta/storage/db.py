"""SQLite store: people and their encrypted biometric embeddings (one file per app)."""

from __future__ import annotations

import functools
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
from numpy.typing import NDArray

from cheta.memory.types import Memory
from cheta.person import Person
from cheta.storage.crypto import decrypt_embedding, encrypt_embedding, load_or_create_key

SCHEMA = """
CREATE TABLE IF NOT EXISTS person (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    consent_at REAL NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS face_embedding (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id TEXT NOT NULL REFERENCES person(id) ON DELETE CASCADE,
    blob BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS face_embedding_person ON face_embedding(person_id);
CREATE TABLE IF NOT EXISTS voice_embedding (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id TEXT NOT NULL REFERENCES person(id) ON DELETE CASCADE,
    blob BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS voice_embedding_person ON voice_embedding(person_id);
CREATE TABLE IF NOT EXISTS memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id TEXT NOT NULL REFERENCES person(id) ON DELETE CASCADE,
    text TEXT NOT NULL,
    embedding BLOB NOT NULL,
    created_at REAL NOT NULL,
    expires_at REAL
);
CREATE INDEX IF NOT EXISTS memory_person ON memory(person_id);
"""


F = TypeVar("F", bound=Callable[..., Any])


def _locked(fn: F) -> F:
    """Serialise access to the shared SQLite connection across threads."""

    @functools.wraps(fn)
    def wrapper(self: Store, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            return fn(self, *args, **kwargs)

    return wrapper  # type: ignore[return-value]


class Store:
    """Encrypted persistent store. Biometric embeddings are only ever written encrypted."""

    def __init__(self, data_dir: str | Path, app: str, key: bytes | None = None) -> None:
        self.data_dir = Path(data_dir).expanduser()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.data_dir / f"{app}.db"
        self._key = key if key is not None else load_or_create_key(self.data_dir, app)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._load_vec()
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.execute("PRAGMA secure_delete = ON")  # overwrite deleted content
        self._db.executescript(SCHEMA)
        self.path.chmod(0o600)

    def _load_vec(self) -> None:
        """Load the sqlite-vec extension (provides vec_distance_cosine) for memory search."""
        try:
            import sqlite_vec

            self._db.enable_load_extension(True)
            sqlite_vec.load(self._db)
            self._db.enable_load_extension(False)
            self.has_vec = True
        except Exception:  # extension unavailable: biometrics still work, memory does not
            self.has_vec = False

    @_locked
    def add_person(self, name: str, consent: bool) -> Person:
        """Create a person. Refuses without explicit consent."""
        if consent is not True:
            raise PermissionError("consent=True is required to store a person")
        person = Person(id=uuid.uuid4().hex, name=name)
        now = time.time()
        with self._db:
            self._db.execute(
                "INSERT INTO person (id, name, consent_at, created_at) VALUES (?, ?, ?, ?)",
                (person.id, name, now, now),
            )
        return person

    @_locked
    def add_face_embeddings(self, person_id: str, embeddings: list[NDArray[np.float32]]) -> None:
        """Encrypt and store face embeddings for a person."""
        with self._db:
            self._db.executemany(
                "INSERT INTO face_embedding (person_id, blob) VALUES (?, ?)",
                [(person_id, encrypt_embedding(self._key, e)) for e in embeddings],
            )

    @_locked
    def face_embeddings(self, person_id: str) -> list[NDArray[np.float32]]:
        """Return a person's decrypted face embeddings."""
        rows = self._db.execute(
            "SELECT blob FROM face_embedding WHERE person_id = ? ORDER BY id", (person_id,)
        ).fetchall()
        return [decrypt_embedding(self._key, r[0]) for r in rows]

    @_locked
    def all_face_embeddings(self) -> dict[str, list[NDArray[np.float32]]]:
        """Return decrypted face embeddings for everyone, keyed by person id."""
        out: dict[str, list[NDArray[np.float32]]] = {}
        rows = self._db.execute("SELECT person_id, blob FROM face_embedding ORDER BY id")
        for person_id, blob in rows:
            out.setdefault(person_id, []).append(decrypt_embedding(self._key, blob))
        return out

    @_locked
    def add_voice_embeddings(self, person_id: str, embeddings: list[NDArray[np.float32]]) -> None:
        """Encrypt and store voice embeddings for a person."""
        with self._db:
            self._db.executemany(
                "INSERT INTO voice_embedding (person_id, blob) VALUES (?, ?)",
                [(person_id, encrypt_embedding(self._key, e)) for e in embeddings],
            )

    @_locked
    def voice_embeddings(self, person_id: str) -> list[NDArray[np.float32]]:
        """Return a person's decrypted voice embeddings."""
        rows = self._db.execute(
            "SELECT blob FROM voice_embedding WHERE person_id = ? ORDER BY id", (person_id,)
        ).fetchall()
        return [decrypt_embedding(self._key, r[0]) for r in rows]

    @_locked
    def all_voice_embeddings(self) -> dict[str, list[NDArray[np.float32]]]:
        """Return decrypted voice embeddings for everyone, keyed by person id."""
        out: dict[str, list[NDArray[np.float32]]] = {}
        rows = self._db.execute("SELECT person_id, blob FROM voice_embedding ORDER BY id")
        for person_id, blob in rows:
            out.setdefault(person_id, []).append(decrypt_embedding(self._key, blob))
        return out

    @_locked
    def list_people(self) -> list[Person]:
        """Return all enrolled people."""
        rows = self._db.execute("SELECT id, name FROM person ORDER BY created_at").fetchall()
        return [Person(id=r[0], name=r[1]) for r in rows]

    @_locked
    def get_person(self, person_id: str) -> Person | None:
        """Look a person up by id."""
        row = self._db.execute("SELECT id, name FROM person WHERE id = ?", (person_id,)).fetchone()
        return Person(id=row[0], name=row[1]) if row else None

    @_locked
    def find_by_name(self, name: str) -> list[Person]:
        """Return people with the given name (case-insensitive)."""
        rows = self._db.execute(
            "SELECT id, name FROM person WHERE name = ? COLLATE NOCASE ORDER BY created_at",
            (name,),
        ).fetchall()
        return [Person(id=r[0], name=r[1]) for r in rows]

    @_locked
    def delete_person(self, person_id: str) -> None:
        """Hard-delete a person and all their biometrics."""
        with self._db:
            self._db.execute("DELETE FROM person WHERE id = ?", (person_id,))  # cascades
        self._db.execute("VACUUM")  # drop freed pages so deleted data is not left in the file

    @_locked
    def add_memory(
        self,
        person_id: str,
        text: str,
        embedding: NDArray[np.float32],
        expires_at: float | None = None,
    ) -> int:
        """Store a memory with its embedding. Returns the memory id."""
        self._require_vec()
        cur = self._db.execute(
            "INSERT INTO memory (person_id, text, embedding, created_at, expires_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (
                person_id,
                text,
                np.asarray(embedding, dtype=np.float32).tobytes(),
                time.time(),
                expires_at,
            ),
        )
        self._db.commit()
        assert cur.lastrowid is not None
        return cur.lastrowid

    @_locked
    def search_memories(
        self, person_id: str, query: NDArray[np.float32], k: int, now: float | None = None
    ) -> list[Memory]:
        """Return the ``k`` unexpired memories of a person most similar to ``query``."""
        self._require_vec()
        rows = self._db.execute(
            "SELECT id, person_id, text, created_at, expires_at,"
            " 1.0 - vec_distance_cosine(embedding, ?) AS score"
            " FROM memory WHERE person_id = ? AND (expires_at IS NULL OR expires_at > ?)"
            " ORDER BY score DESC, id DESC LIMIT ?",
            (
                np.asarray(query, dtype=np.float32).tobytes(),
                person_id,
                time.time() if now is None else now,
                k,
            ),
        ).fetchall()
        return [Memory(*r) for r in rows]

    @_locked
    def list_memories(self, person_id: str) -> list[Memory]:
        """Return all of a person's memories (including expired ones), oldest first."""
        rows = self._db.execute(
            "SELECT id, person_id, text, created_at, expires_at FROM memory"
            " WHERE person_id = ? ORDER BY id",
            (person_id,),
        ).fetchall()
        return [Memory(*r) for r in rows]

    @_locked
    def purge_expired(self, now: float | None = None) -> int:
        """Delete expired memories. Returns how many were removed."""
        with self._db:
            cur = self._db.execute(
                "DELETE FROM memory WHERE expires_at IS NOT NULL AND expires_at <= ?",
                (time.time() if now is None else now,),
            )
        return cur.rowcount

    def _require_vec(self) -> None:
        if not self.has_vec:
            raise RuntimeError(
                "sqlite-vec could not be loaded; memory needs a Python build with "
                "SQLite extension support"
            )

    @_locked
    def close(self) -> None:
        """Close the database connection."""
        self._db.close()
