"""Semantic recall with the real BGE model. Deselected by default (needs the model download).

Run with: uv run pytest -m network tests/test_memory_real.py
"""

import pytest

from cheta import Identity
from cheta.storage import Store
from tests.test_memory import FACTS

pytestmark = pytest.mark.network


def test_semantic_recall_ranks_right_fact_first(store: Store) -> None:
    idt = Identity("test", store=store)  # real BgeEmbedder
    dave = store.add_person("Dave", consent=True)
    for fact in FACTS:
        idt.remember(dave, fact)
    wrong = [
        (q, idt.recall(dave, q, k=1)[0].text)
        for fact, q in FACTS.items()
        if idt.recall(dave, q, k=1)[0].text != fact
    ]
    assert not wrong, wrong
