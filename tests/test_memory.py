"""Tests for the memory system."""


import pytest

from duxx_ai.memory.manager import (
    EpisodicMemory,
    MemoryManager,
    ProceduralMemory,
    SemanticMemory,
    SharedMemory,
    WorkingMemory,
)


class TestWorkingMemory:
    def test_store_and_recall(self):
        wm = WorkingMemory()
        wm.store("key1", "value1")
        assert wm.recall("key1") == "value1"

    def test_recall_missing(self):
        wm = WorkingMemory()
        assert wm.recall("nonexistent") is None

    def test_eviction(self):
        wm = WorkingMemory(max_items=3)
        wm.store("a", "1")
        wm.store("b", "2")
        wm.store("c", "3")
        wm.store("d", "4")  # Should evict oldest
        assert len(wm.items) == 3

    def test_clear(self):
        wm = WorkingMemory()
        wm.store("a", "1")
        wm.clear()
        assert len(wm.items) == 0


class TestEpisodicMemory:
    def test_record_and_recall(self):
        em = EpisodicMemory()
        em.record("User asked about revenue", agent_id="analyst", importance=0.8)
        em.record("Generated quarterly report", agent_id="analyst", importance=0.6)

        results = em.recall("revenue report", top_k=2)
        assert len(results) > 0

    def test_importance_affects_recall(self):
        em = EpisodicMemory()
        em.record("Low importance item", importance=0.1)
        em.record("High importance revenue data", importance=0.9)

        results = em.recall("revenue", top_k=1)
        assert "revenue" in results[0].content.lower()


class TestSemanticMemory:
    def test_store_and_recall(self):
        sm = SemanticMemory()
        sm.store("fiscal_year", "April to March")
        assert sm.recall("fiscal_year") == "April to March"

    def test_search(self):
        sm = SemanticMemory()
        sm.store("revenue_q1", "Revenue Q1: $1M")
        sm.store("revenue_q2", "Revenue Q2: $1.2M")
        sm.store("employees", "Total employees: 50")

        results = sm.search("revenue")
        assert len(results) == 2


class TestProceduralMemory:
    def test_record_and_get(self):
        pm = ProceduralMemory()
        steps = [
            {"action": "fetch_data"},
            {"action": "analyze"},
            {"action": "report"},
        ]
        pm.record_procedure("monthly_report", steps)

        result = pm.get_procedure("monthly_report")
        assert result is not None
        assert len(result) == 3

    def test_list_procedures(self):
        pm = ProceduralMemory()
        pm.record_procedure("proc1", [])
        pm.record_procedure("proc2", [])
        assert len(pm.list_procedures()) == 2


class TestSharedMemory:
    def test_write_and_read(self):
        sm = SharedMemory()
        sm.write("status", "active", agent_id="agent1")
        assert sm.read("status") == "active"

    def test_lock_prevents_write(self):
        sm = SharedMemory()
        sm.write("status", "active", agent_id="agent1")
        sm.lock("status", "agent1")

        with pytest.raises(ValueError):
            sm.write("status", "inactive", agent_id="agent2")

    def test_unlock_allows_write(self):
        sm = SharedMemory()
        sm.write("status", "active", agent_id="agent1")
        sm.lock("status", "agent1")
        sm.unlock("status", "agent1")
        sm.write("status", "inactive", agent_id="agent2")
        assert sm.read("status") == "inactive"


class TestMemoryManager:
    def test_auto_store(self):
        mm = MemoryManager()
        mm.auto_store("Processing sales data", agent_id="analyst", context="task")
        mm.auto_store("Q4 revenue was $2.5M", agent_id="analyst", context="fact")

        results = mm.recall_all("revenue")
        assert len(results) > 0
