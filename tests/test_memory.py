import time

import pytest

from cheta import Identity, Person
from cheta.storage import Store
from tests.conftest import FakeTextEmbedder, rand_emb

FACTS = {
    "Dave loves playing the guitar on weekends": "what instrument does he play",
    "Dave is allergic to peanuts": "any food allergies",
    "Dave's dog is called Biscuit": "what is his dog called",
    "Dave works as a civil engineer": "what is his job",
    "Dave wants to visit Japan next spring": "where does he want to travel",
    "Dave drinks oat milk coffee every morning": "what does he drink in the morning",
    "Dave's birthday is in March": "when is his birthday",
    "Dave studied at the University of Leeds": "where did he study",
    "Dave runs marathons for charity": "what does he do for fitness",
    "Dave's sister lives in Canada": "where does his sister live",
}


@pytest.fixture
def idt(store: Store) -> Identity:
    return Identity("test", store=store, text_embedder=FakeTextEmbedder())


@pytest.fixture
def dave(store: Store) -> Person:
    return store.add_person("Dave", consent=True)


def test_recall_ranks_right_fact_first(idt: Identity, dave: Person) -> None:
    for fact in FACTS:
        idt.remember(dave, fact)
    # Fake bag-of-words embeddings only share literal tokens, so use keyword queries here;
    # the real-model semantic version lives in test_memory_real.py.
    queries = {
        "guitar": "guitar",
        "peanuts": "peanuts",
        "dog Biscuit": "Biscuit",
        "civil engineer": "engineer",
        "Japan": "Japan",
        "oat milk coffee": "coffee",
        "birthday March": "birthday",
        "University Leeds": "Leeds",
        "marathons": "marathons",
        "sister Canada": "Canada",
    }
    for query, word in queries.items():
        top = idt.recall(dave, query, k=3)
        assert word in top[0].text, (query, top[0].text)


def test_recall_respects_k_and_scores_descend(idt: Identity, dave: Person) -> None:
    for fact in FACTS:
        idt.remember(dave, fact)
    res = idt.recall(dave, "dog Biscuit", k=3)
    assert len(res) == 3
    assert [m.score for m in res] == sorted((m.score for m in res), reverse=True)


def test_memories_are_per_person(idt: Identity, dave: Person, store: Store) -> None:
    ann = store.add_person("Ann", consent=True)
    idt.remember(dave, "Dave likes tea")
    idt.remember(ann, "Ann likes coffee")
    assert [m.text for m in idt.recall(dave, "likes", k=5)] == ["Dave likes tea"]
    assert [m.text for m in idt.recall(ann, "likes", k=5)] == ["Ann likes coffee"]


def test_expired_memories_not_recalled_and_purged(
    idt: Identity, dave: Person, store: Store
) -> None:
    idt.remember(dave, "Dave has a cold", expires_in_days=1)
    idt.remember(dave, "Dave likes tea")
    cold_id = store.list_memories(dave.id)[0].id
    assert len(idt.recall(dave, "Dave")) == 2
    store._db.execute("UPDATE memory SET expires_at = ? WHERE id = ?", (time.time() - 1, cold_id))
    assert [m.text for m in idt.recall(dave, "Dave")] == ["Dave likes tea"]
    assert len(store.list_memories(dave.id)) == 1  # purged


def test_validation(idt: Identity, dave: Person) -> None:
    with pytest.raises(ValueError):
        idt.remember(dave, "  ")
    with pytest.raises(ValueError):
        idt.remember(dave, "x", expires_in_days=0)
    with pytest.raises(ValueError):
        idt.recall(dave, "x", k=0)
    with pytest.raises(KeyError):
        idt.remember(Person("nope", "Nobody"), "x")


def test_forget_leaves_zero_rows(idt: Identity, dave: Person, store: Store) -> None:
    from tests.conftest import rand_emb

    store.add_face_embeddings(dave.id, [rand_emb(i) for i in range(5)])
    for fact in FACTS:
        idt.remember(dave, fact)
    idt.forget(dave)
    for table in ("person", "face_embedding", "memory"):
        assert store._db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0, table


def test_forget_only_removes_that_person(idt: Identity, dave: Person, store: Store) -> None:
    ann = store.add_person("Ann", consent=True)
    idt.remember(dave, "Dave likes tea")
    idt.remember(ann, "Ann likes coffee")
    idt.forget(dave)
    assert [p.name for p in store.list_people()] == ["Ann"]
    assert len(idt.recall(ann, "likes")) == 1


def test_forgotten_text_not_left_in_file(tmp_path, key) -> None:  # type: ignore[no-untyped-def]
    s = Store(tmp_path, "t", key=key)
    idt = Identity("t", store=s, text_embedder=FakeTextEmbedder())
    p = s.add_person("Dave", consent=True)
    idt.remember(p, "zebra-unique-secret-fact")
    idt.forget(p)
    s.close()
    assert b"zebra-unique-secret-fact" not in s.path.read_bytes()


def test_export(idt: Identity, dave: Person, store: Store) -> None:
    from tests.conftest import rand_emb

    store.add_face_embeddings(dave.id, [rand_emb(1), rand_emb(2)])
    idt.remember(dave, "Dave likes tea", expires_in_days=7)
    data = idt.export(dave)
    assert data["name"] == "Dave" and data["face_samples"] == 2
    assert data["memories"][0]["text"] == "Dave likes tea"
    assert data["memories"][0]["expires_at"] is not None
    import json

    json.dumps(data)  # serialisable
    assert "embedding" not in json.dumps(data)


def test_memory_text_is_encrypted_at_rest(idt: Identity, dave: Person, store: Store) -> None:
    idt.remember(dave, "Dave's dog is called Biscuit")
    raw = store._db.execute("SELECT text FROM memory").fetchone()[0]
    assert isinstance(raw, bytes) and b"Biscuit" not in raw
    assert [m.text for m in idt.recall(dave, "dog")] == ["Dave's dog is called Biscuit"]
    assert [m.text for m in store.list_memories(dave.id)] == ["Dave's dog is called Biscuit"]


def test_memory_text_not_in_database_file(idt: Identity, dave: Person, store: Store) -> None:
    idt.remember(dave, "Dave's dog is called Biscuit")
    store._db.commit()
    assert b"Biscuit" not in store.path.read_bytes()


def test_legacy_plaintext_memory_still_readable(idt: Identity, dave: Person, store: Store) -> None:
    store._db.execute(
        "INSERT INTO memory (person_id, text, embedding, created_at) VALUES (?, ?, ?, ?)",
        (dave.id, "old plaintext", rand_emb(1).tobytes(), time.time()),
    )
    assert [m.text for m in store.list_memories(dave.id)] == ["old plaintext"]
