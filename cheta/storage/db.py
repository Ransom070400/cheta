"""SQLite store: people and their encrypted biometric embeddings (one file per app)."""

from __future__ import annotations

import sqlite3
import time
import uuid
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

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
"""


class Store:
    """Encrypted persistent store. Biometric embeddings are only ever written encrypted."""

    def __init__(self, data_dir: str | Path, app: str, key: bytes | None = None) -> None:
        self.data_dir = Path(data_dir).expanduser()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.data_dir / f"{app}.db"
        self._key = key if key is not None else load_or_create_key(self.data_dir, app)
        self._db = sqlite3.connect(self.path)
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.execute("PRAGMA secure_delete = ON")  # overwrite deleted content
        self._db.executescript(SCHEMA)
        self.path.chmod(0o600)

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

    def add_face_embeddings(self, person_id: str, embeddings: list[NDArray[np.float32]]) -> None:
        """Encrypt and store face embeddings for a person."""
        with self._db:
            self._db.executemany(
                "INSERT INTO face_embedding (person_id, blob) VALUES (?, ?)",
                [(person_id, encrypt_embedding(self._key, e)) for e in embeddings],
            )

    def face_embeddings(self, person_id: str) -> list[NDArray[np.float32]]:
        """Return a person's decrypted face embeddings."""
        rows = self._db.execute(
            "SELECT blob FROM face_embedding WHERE person_id = ? ORDER BY id", (person_id,)
        ).fetchall()
        return [decrypt_embedding(self._key, r[0]) for r in rows]

    def all_face_embeddings(self) -> dict[str, list[NDArray[np.float32]]]:
        """Return decrypted face embeddings for everyone, keyed by person id."""
        out: dict[str, list[NDArray[np.float32]]] = {}
        rows = self._db.execute("SELECT person_id, blob FROM face_embedding ORDER BY id")
        for person_id, blob in rows:
            out.setdefault(person_id, []).append(decrypt_embedding(self._key, blob))
        return out

    def list_people(self) -> list[Person]:
        """Return all enrolled people."""
        rows = self._db.execute("SELECT id, name FROM person ORDER BY created_at").fetchall()
        return [Person(id=r[0], name=r[1]) for r in rows]

    def get_person(self, person_id: str) -> Person | None:
        """Look a person up by id."""
        row = self._db.execute("SELECT id, name FROM person WHERE id = ?", (person_id,)).fetchone()
        return Person(id=row[0], name=row[1]) if row else None

    def find_by_name(self, name: str) -> list[Person]:
        """Return people with the given name (case-insensitive)."""
        rows = self._db.execute(
            "SELECT id, name FROM person WHERE name = ? COLLATE NOCASE ORDER BY created_at",
            (name,),
        ).fetchall()
        return [Person(id=r[0], name=r[1]) for r in rows]

    def delete_person(self, person_id: str) -> None:
        """Hard-delete a person and all their biometrics."""
        with self._db:
            self._db.execute("DELETE FROM person WHERE id = ?", (person_id,))
        self._db.execute("VACUUM")

    def close(self) -> None:
        """Close the database connection."""
        self._db.close()
