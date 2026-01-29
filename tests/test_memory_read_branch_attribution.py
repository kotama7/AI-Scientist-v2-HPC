"""Tests for memory read branch_id attribution.

Verifies that mem_recall_search and mem_archival_search use the correct
branch_id (the caller's branch) instead of always defaulting to root_branch_id.
This ensures memory operation logs are attributed to the correct tree node.
"""

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from ai_scientist.memory.memgpt_store import MemoryManager


class TestMemRecallSearchBranchAttribution(unittest.TestCase):
    """mem_recall_search should use the provided branch_id, not root."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "test.sqlite"
        self.log_path = Path(self.tmp.name) / "memory_calls.jsonl"
        self.mm = MemoryManager(
            self.db_path,
            {
                "memory_log_enabled": True,
                "memory_log_dir": self.tmp.name,
            },
        )
        # Create root branch
        self.root_id = self.mm.mem_node_fork(None, "root_branch")
        # Create child branch
        self.child_id = self.mm.mem_node_fork("root_branch", "child_branch")

    def tearDown(self):
        self.tmp.cleanup()

    def test_default_uses_root(self):
        """Without branch_id param, should use root (backward compat)."""
        self.mm.mem_recall_search("test query", k=5)
        # Read log to verify branch_id
        events = self._read_log_events("mem_recall_search")
        self.assertTrue(len(events) >= 1)
        self.assertEqual(events[-1]["branch_id"], "root_branch")

    def test_explicit_branch_id_used(self):
        """With branch_id param, should use that branch, not root."""
        self.mm.mem_recall_search("test query", k=5, branch_id="child_branch")
        events = self._read_log_events("mem_recall_search")
        self.assertTrue(len(events) >= 1)
        self.assertEqual(events[-1]["branch_id"], "child_branch")

    def test_apply_llm_memory_updates_passes_branch_id(self):
        """apply_llm_memory_updates should pass its branch_id to mem_recall_search."""
        # Write a recall event so search has something to find
        self.mm.mem_recall_append({
            "branch_id": "child_branch",
            "kind": "observation",
            "text": "test data",
            "summary": "test data",
        })
        self.mm.apply_llm_memory_updates(
            branch_id="child_branch",
            updates={"mem_recall_search": {"query": "test", "k": 5}},
            node_id="node_1",
            phase="phase2",
        )
        events = self._read_log_events("mem_recall_search")
        self.assertTrue(len(events) >= 1)
        # Must be child_branch, NOT root_branch
        self.assertEqual(events[-1]["branch_id"], "child_branch")

    def _read_log_events(self, op_filter: str) -> list:
        """Read memory call log and filter by op."""
        events = []
        if self.mm.memory_log_path and self.mm.memory_log_path.exists():
            with open(self.mm.memory_log_path) as f:
                for line in f:
                    d = json.loads(line.strip())
                    if d.get("op") == op_filter:
                        events.append(d)
        return events


class TestMemArchivalSearchBranchAttribution(unittest.TestCase):
    """mem_archival_search should use the provided branch_id, not root."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "test.sqlite"
        self.mm = MemoryManager(
            self.db_path,
            {
                "memory_log_enabled": True,
                "memory_log_dir": self.tmp.name,
            },
        )
        self.root_id = self.mm.mem_node_fork(None, "root_branch")
        self.child_id = self.mm.mem_node_fork("root_branch", "child_branch")

    def tearDown(self):
        self.tmp.cleanup()

    def test_default_uses_root(self):
        """Without branch_id param, should use root."""
        self.mm.mem_archival_search("test query", k=5)
        events = self._read_log_events("mem_archival_search")
        self.assertTrue(len(events) >= 1)
        self.assertEqual(events[-1]["branch_id"], "root_branch")

    def test_explicit_branch_id_used(self):
        """With branch_id param, should use that branch."""
        self.mm.mem_archival_search("test query", k=5, branch_id="child_branch")
        events = self._read_log_events("mem_archival_search")
        self.assertTrue(len(events) >= 1)
        self.assertEqual(events[-1]["branch_id"], "child_branch")

    def _read_log_events(self, op_filter: str) -> list:
        events = []
        if self.mm.memory_log_path and self.mm.memory_log_path.exists():
            with open(self.mm.memory_log_path) as f:
                for line in f:
                    d = json.loads(line.strip())
                    if d.get("op") == op_filter:
                        events.append(d)
        return events


if __name__ == "__main__":
    unittest.main()
