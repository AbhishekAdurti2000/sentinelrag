from pathlib import Path

import pytest

from sentinelrag.memory.store import LongTermMemory


@pytest.fixture()
def memory(tmp_path: Path) -> LongTermMemory:
    mem = LongTermMemory(repo="acme/repo", db_path=tmp_path / "memory.sqlite3")
    yield mem
    mem.close()


def test_remember_and_recall(memory: LongTermMemory):
    memory.remember("priority", "Project X is top priority this quarter")
    assert memory.recall("priority") == "Project X is top priority this quarter"


def test_recall_missing_key_returns_none(memory: LongTermMemory):
    assert memory.recall("nonexistent") is None


def test_remember_same_key_updates_in_place_no_growth(memory: LongTermMemory):
    memory.remember("priority", "first value")
    memory.remember("priority", "second value")
    facts = memory.recall_all()
    assert len(facts) == 1
    assert facts[0].value == "second value"


def test_memory_scoped_per_repo(tmp_path: Path):
    db = tmp_path / "shared.sqlite3"
    mem_a = LongTermMemory(repo="acme/repo-a", db_path=db)
    mem_b = LongTermMemory(repo="acme/repo-b", db_path=db)
    mem_a.remember("priority", "repo A priority")
    assert mem_b.recall("priority") is None
    mem_a.close()
    mem_b.close()


def test_forget_removes_fact(memory: LongTermMemory):
    memory.remember("temp", "value")
    memory.forget("temp")
    assert memory.recall("temp") is None
