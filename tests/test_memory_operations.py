"""Tests for memory operations including LLM-driven memory updates."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from ai_scientist.memory import MemoryManager


class TestMemoryOperations(unittest.TestCase):
    """Test cases for MemoryManager operations."""

    def setUp(self):
        """Set up a temporary memory manager for each test."""
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = Path(self.tmpdir) / "memory.sqlite"
        self.cfg = SimpleNamespace(
            core_max_chars=4000,
            recall_max_events=20,
            retrieval_k=8,
            use_fts="off",
            memory_log_enabled=False,
            auto_consolidate=False,
        )
        self.mem = MemoryManager(self.db_path, self.cfg)

    def tearDown(self):
        """Clean up temporary directory."""
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_create_branch(self):
        """Test branch creation."""
        root = self.mem.create_branch(None, node_uid="root")
        self.assertIsInstance(root, str)
        self.assertTrue(len(root) > 0)

    def test_create_child_branch(self):
        """Test child branch creation."""
        root = self.mem.create_branch(None, node_uid="root")
        child = self.mem.create_branch(root, node_uid="child")
        self.assertIsInstance(child, str)
        self.assertNotEqual(root, child)

    def test_set_and_get_core(self):
        """Test setting and getting core memory."""
        branch = self.mem.create_branch(None, node_uid="test")
        self.mem.set_core(branch, "test_key", "test_value")

        # Render prompt should include the core value
        prompt = self.mem.render_for_prompt(branch, task_hint="test", budget_chars=4000)
        self.assertIn("test_key", prompt)
        self.assertIn("test_value", prompt)

    def test_write_event(self):
        """Test writing recall events."""
        branch = self.mem.create_branch(None, node_uid="test")
        self.mem.write_event(branch, "test_event", "event content", tags=["TAG1"])

        prompt = self.mem.render_for_prompt(branch, task_hint="test", budget_chars=4000)
        self.assertIn("event content", prompt)

    def test_write_archival(self):
        """Test writing archival memory."""
        branch = self.mem.create_branch(None, node_uid="test")
        self.mem.write_archival(branch, "archival content", tags=["ARCH_TAG"])

        prompt = self.mem.render_for_prompt(branch, task_hint="archival", budget_chars=4000)
        self.assertIn("archival content", prompt)

    def test_apply_llm_memory_updates_write(self):
        """Test applying LLM memory updates for write operations."""
        branch = self.mem.create_branch(None, node_uid="test")

        # Simulate LLM memory update with write operations
        updates = {
            "core": {"algorithm": "gradient_descent", "learning_rate": "0.01"},
            "archival": [
                {"text": "Test archival entry", "tags": ["TEST", "ALGO"]}
            ],
        }

        # Note: argument order is (branch_id, updates, node_id)
        results = self.mem.apply_llm_memory_updates(branch, updates, "test_node")

        # Verify core was updated
        prompt = self.mem.render_for_prompt(branch, task_hint="test", budget_chars=4000)
        self.assertIn("algorithm", prompt)
        self.assertIn("gradient_descent", prompt)
        self.assertIn("learning_rate", prompt)
        self.assertIn("0.01", prompt)

    def test_apply_llm_memory_updates_read(self):
        """Test applying LLM memory updates for read operations."""
        branch = self.mem.create_branch(None, node_uid="test")

        # First write some data
        self.mem.set_core(branch, "existing_key", "existing_value")
        self.mem.write_archival(branch, "searchable content about algorithms", tags=["ALGO"])

        # Simulate LLM memory update with read operations
        updates = {
            "core_get": ["existing_key", "nonexistent_key"],
            "archival_search": {"query": "algorithm", "k": 3},
        }

        # Note: argument order is (branch_id, updates, node_id)
        results = self.mem.apply_llm_memory_updates(branch, updates, "test_node")

        # Verify read results
        self.assertIn("core_get", results)
        self.assertIn("existing_key", results["core_get"])
        self.assertEqual(results["core_get"]["existing_key"], "existing_value")
        self.assertIsNone(results["core_get"]["nonexistent_key"])

        self.assertIn("archival_search", results)
        self.assertIsInstance(results["archival_search"], list)

    def test_branch_inheritance(self):
        """Test that child branches inherit parent memory."""
        root = self.mem.create_branch(None, node_uid="root")
        self.mem.set_core(root, "parent_key", "parent_value")
        self.mem.write_archival(root, "parent archival", tags=["PARENT"])

        child = self.mem.create_branch(root, node_uid="child")
        prompt = self.mem.render_for_prompt(child, task_hint="test", budget_chars=4000)

        self.assertIn("parent_key", prompt)
        self.assertIn("parent_value", prompt)
        self.assertIn("parent archival", prompt)

    def test_branch_isolation(self):
        """Test that sibling branches are isolated."""
        root = self.mem.create_branch(None, node_uid="root")
        child1 = self.mem.create_branch(root, node_uid="child1")
        child2 = self.mem.create_branch(root, node_uid="child2")

        # Write to child1
        self.mem.set_core(child1, "child1_key", "child1_value")
        self.mem.write_event(child1, "note", "child1 event", tags=["C1"])

        # Check child2 doesn't see child1's data
        prompt2 = self.mem.render_for_prompt(child2, task_hint="test", budget_chars=4000)
        self.assertNotIn("child1_key", prompt2)
        self.assertNotIn("child1_value", prompt2)
        self.assertNotIn("child1 event", prompt2)

    def test_empty_updates(self):
        """Test that empty updates are handled gracefully."""
        branch = self.mem.create_branch(None, node_uid="test")

        # Empty dict - argument order is (branch_id, updates, node_id)
        results = self.mem.apply_llm_memory_updates(branch, {}, "test_node")
        self.assertIsInstance(results, dict)

        # None values
        results = self.mem.apply_llm_memory_updates(
            branch,
            {"core": None, "archival": None},
            "test_node"
        )
        self.assertIsInstance(results, dict)


class TestMemoryUpdateExtraction(unittest.TestCase):
    """Test cases for extracting memory updates from LLM responses."""

    def test_extract_memory_updates_from_response(self):
        """Test extracting memory_update blocks from LLM response."""
        from ai_scientist.treesearch.utils.response import extract_memory_updates

        response = '''Some text before
<memory_update>
{
  "core": {"key1": "value1"},
  "archival": [{"text": "test", "tags": ["TAG"]}]
}
</memory_update>
Some text after
{"command": "pip install numpy", "done": false}'''

        updates = extract_memory_updates(response)

        self.assertIsNotNone(updates)
        self.assertIn("core", updates)
        self.assertEqual(updates["core"]["key1"], "value1")
        self.assertIn("archival", updates)
        self.assertEqual(len(updates["archival"]), 1)

    def test_extract_memory_updates_no_block(self):
        """Test extraction when no memory_update block exists."""
        from ai_scientist.treesearch.utils.response import extract_memory_updates

        response = '{"command": "pip install numpy", "done": false}'
        updates = extract_memory_updates(response)

        self.assertIsNone(updates)

    def test_extract_memory_updates_invalid_json(self):
        """Test extraction with invalid JSON in memory_update block."""
        from ai_scientist.treesearch.utils.response import extract_memory_updates

        response = '''<memory_update>
{invalid json}
</memory_update>'''

        updates = extract_memory_updates(response)
        self.assertIsNone(updates)

    def test_extract_memory_updates_read_operations(self):
        """Test extraction of read operations."""
        from ai_scientist.treesearch.utils.response import extract_memory_updates

        response = '''<memory_update>
{
  "core_get": ["key1", "key2"],
  "archival_search": {"query": "algorithm", "k": 5}
}
</memory_update>'''

        updates = extract_memory_updates(response)

        self.assertIsNotNone(updates)
        self.assertIn("core_get", updates)
        self.assertEqual(updates["core_get"], ["key1", "key2"])
        self.assertIn("archival_search", updates)
        self.assertEqual(updates["archival_search"]["query"], "algorithm")
        self.assertEqual(updates["archival_search"]["k"], 5)


class TestMemoryReadResultsHelper(unittest.TestCase):
    """Test cases for _has_memory_read_results helper."""

    def test_has_read_results_with_core_get(self):
        """Test detection of core_get read results."""
        from ai_scientist.treesearch.parallel_agent import _has_memory_read_results

        results = {
            "core_get": {"key1": "value1"}
        }
        self.assertTrue(_has_memory_read_results(results))

    def test_has_read_results_with_archival_search(self):
        """Test detection of archival_search read results."""
        from ai_scientist.treesearch.parallel_agent import _has_memory_read_results

        results = {
            "archival_search": [{"content": "test", "tags": ["TAG"]}]
        }
        self.assertTrue(_has_memory_read_results(results))

    def test_has_read_results_empty(self):
        """Test detection with empty results."""
        from ai_scientist.treesearch.parallel_agent import _has_memory_read_results

        self.assertFalse(_has_memory_read_results({}))
        self.assertFalse(_has_memory_read_results(None))

    def test_has_read_results_write_only(self):
        """Test detection with write-only results."""
        from ai_scientist.treesearch.parallel_agent import _has_memory_read_results

        results = {
            "core_written": {"key1": "value1"},
            "archival_written": [{"text": "test"}]
        }
        self.assertFalse(_has_memory_read_results(results))


if __name__ == "__main__":
    unittest.main()
