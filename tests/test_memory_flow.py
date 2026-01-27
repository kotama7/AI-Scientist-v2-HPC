"""Tests for memory flow as documented in docs/memory/memory-flow.md and memory-flow-phases.md.

This file tests the documented memory flows including:
- Memory injection (_inject_memory)
- Response tag processing (remove_memory_update_tags, sanitize_memory_update_tags)
- Error handling (MissingMemoryUpdateError)
- Two-phase pattern
"""

import unittest

from ai_scientist.treesearch.utils.response import (
    extract_memory_updates,
    remove_memory_update_tags,
    sanitize_memory_update_tags,
    check_malformed_memory_update,
)
from ai_scientist.treesearch.utils.phase_plan import (
    MissingMemoryUpdateError,
    PhasePlanError,
)


class TestMemoryUpdateTagProcessing(unittest.TestCase):
    """Test memory_update tag processing functions documented in memory-flow.md."""

    def test_remove_memory_update_tags_normal(self):
        """Test removing normal <memory_update>...</memory_update> blocks."""
        text = '''<memory_update>
{"core": {"key": "value"}}
</memory_update>
{"phase_artifacts": {"download": {}}}'''

        result = remove_memory_update_tags(text)
        self.assertNotIn("<memory_update>", result)
        self.assertNotIn("</memory_update>", result)
        self.assertIn("phase_artifacts", result)

    def test_remove_memory_update_tags_escaped_slash(self):
        """Test removing tags with escaped slash (LLM quirk)."""
        text = '<memory_update>\n{"core": {"key": "value"}}\n<\\/memory_update>\n{"phase_artifacts": {}}'

        result = remove_memory_update_tags(text)
        self.assertNotIn("<memory_update>", result)
        self.assertNotIn("<\\/memory_update>", result)
        self.assertIn("phase_artifacts", result)

    def test_remove_memory_update_tags_empty(self):
        """Test with empty or None input."""
        self.assertEqual(remove_memory_update_tags(""), "")
        self.assertEqual(remove_memory_update_tags(None), None)

    def test_sanitize_memory_update_tags_with_attributes(self):
        """Test sanitizing malformed opening tags with injected attributes."""
        # LLM sometimes outputs tags with injected attributes
        text = '<memory_update  大发彩票官网_json_duplication="true">{"core": {}}</memory_update>'
        result = sanitize_memory_update_tags(text)
        self.assertIn("<memory_update>", result)
        self.assertNotIn('大发彩票官网', result)

    def test_sanitize_memory_update_tags_normal(self):
        """Test that normal tags are unchanged."""
        text = '<memory_update>{"core": {}}</memory_update>'
        result = sanitize_memory_update_tags(text)
        self.assertEqual(result, text)

    def test_sanitize_memory_update_tags_empty(self):
        """Test with empty input."""
        self.assertEqual(sanitize_memory_update_tags(""), "")
        self.assertEqual(sanitize_memory_update_tags(None), None)

    def test_check_malformed_memory_update_self_closing(self):
        """Test detection of malformed self-closing pattern."""
        # Malformed: <memory_update"core":{...}}/>
        malformed = '<memory_update"core":{"key":"value"}}/>'
        self.assertTrue(check_malformed_memory_update(malformed))

    def test_check_malformed_memory_update_normal(self):
        """Test that normal format is not flagged."""
        normal = '<memory_update>{"core": {}}</memory_update>'
        self.assertFalse(check_malformed_memory_update(normal))

    def test_check_malformed_memory_update_escaped(self):
        """Test that escaped slash is not flagged as malformed."""
        escaped = '<memory_update>{"core": {}}<\\/memory_update>'
        self.assertFalse(check_malformed_memory_update(escaped))

    def test_check_malformed_memory_update_empty(self):
        """Test with empty input."""
        self.assertFalse(check_malformed_memory_update(""))
        self.assertFalse(check_malformed_memory_update(None))


class TestMissingMemoryUpdateError(unittest.TestCase):
    """Test MissingMemoryUpdateError as documented in memory-flow-phases.md:467-468."""

    def test_error_is_subclass_of_phase_plan_error(self):
        """MissingMemoryUpdateError should be a subclass of PhasePlanError."""
        self.assertTrue(issubclass(MissingMemoryUpdateError, PhasePlanError))

    def test_error_can_be_raised(self):
        """Test that error can be raised and caught."""
        with self.assertRaises(MissingMemoryUpdateError):
            raise MissingMemoryUpdateError("Memory update block required")

    def test_error_message_preserved(self):
        """Test that error message is preserved."""
        msg = "Missing <memory_update> block in response"
        try:
            raise MissingMemoryUpdateError(msg)
        except MissingMemoryUpdateError as e:
            self.assertIn("memory_update", str(e).lower())


class TestExtractMemoryUpdatesIntegration(unittest.TestCase):
    """Integration tests for extract_memory_updates with various LLM output formats."""

    def test_extract_with_phase_artifacts(self):
        """Test extraction from response with both memory_update and phase_artifacts."""
        # As documented in memory-flow-phases.md:452-463
        response = '''<memory_update>
{
  "core": {"optimization_level": "O3"},
  "archival": [{"text": "Compiled with -O3 flag", "tags": ["COMPILE"]}]
}
</memory_update>
{"phase_artifacts": {"compile": {"commands": ["make"]}}, "constraints": {}}'''

        updates = extract_memory_updates(response)
        self.assertIsNotNone(updates)
        self.assertEqual(updates["core"]["optimization_level"], "O3")
        self.assertEqual(len(updates["archival"]), 1)

    def test_extract_with_read_operations(self):
        """Test extraction with read operations (triggers re-query loop)."""
        # As documented in memory-flow.md:198-222
        response = '''<memory_update>
{
  "archival_search": {"query": "PHASE1_ERROR", "k": 3},
  "core_get": ["python_deps_path"]
}
</memory_update>
{"command": "pip install numpy", "done": false}'''

        updates = extract_memory_updates(response)
        self.assertIsNotNone(updates)
        self.assertIn("archival_search", updates)
        self.assertEqual(updates["archival_search"]["query"], "PHASE1_ERROR")
        self.assertIn("core_get", updates)

    def test_extract_with_sanitization_needed(self):
        """Test extraction handles malformed tags via sanitization."""
        response = '<memory_update attr="bad">{"core": {"key": "value"}}</memory_update>'
        updates = extract_memory_updates(response)
        self.assertIsNotNone(updates)
        self.assertEqual(updates["core"]["key"], "value")

    def test_extract_multiline_json(self):
        """Test extraction with multiline JSON content."""
        response = '''<memory_update>
{
  "core": {
    "experiment_config": "Himeno benchmark",
    "target_speedup": "2x"
  },
  "archival": [
    {
      "text": "Initial baseline: 10s execution time",
      "tags": ["BASELINE", "PERFORMANCE"]
    }
  ]
}
</memory_update>
{"phase_artifacts": {}}'''

        updates = extract_memory_updates(response)
        self.assertIsNotNone(updates)
        self.assertEqual(updates["core"]["experiment_config"], "Himeno benchmark")
        self.assertEqual(len(updates["archival"]), 1)


class TestMemoryContextStructure(unittest.TestCase):
    """Test memory context structure as documented in memory-flow.md:85-129."""

    def test_render_for_prompt_includes_core(self):
        """Test that rendered prompt includes Core Memory section."""
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from ai_scientist.memory import MemoryManager

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "memory.sqlite"
            cfg = SimpleNamespace(
                core_max_chars=4000,
                recall_max_events=20,
                retrieval_k=8,
                use_fts="off",
                memory_log_enabled=False,
                auto_consolidate=False,
            )
            mem = MemoryManager(db_path, cfg)
            branch = mem.create_branch(None, node_uid="test")
            mem.set_core(branch, "idea_md_summary", "Research on optimization")

            prompt = mem.render_for_prompt(branch, task_hint="test", budget_chars=4000)

            # Core memory should be visible
            self.assertIn("idea_md_summary", prompt)
            self.assertIn("Research on optimization", prompt)

    def test_render_for_prompt_includes_recall(self):
        """Test that rendered prompt includes Recall Memory section."""
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from ai_scientist.memory import MemoryManager

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "memory.sqlite"
            cfg = SimpleNamespace(
                core_max_chars=4000,
                recall_max_events=20,
                retrieval_k=8,
                use_fts="off",
                memory_log_enabled=False,
                auto_consolidate=False,
            )
            mem = MemoryManager(db_path, cfg)
            branch = mem.create_branch(None, node_uid="test")
            mem.write_event(branch, "node_created", "Node 5 started", tags=["NODE"])

            prompt = mem.render_for_prompt(branch, task_hint="test", budget_chars=4000)

            # Recall event should be visible
            self.assertIn("Node 5 started", prompt)

    def test_render_for_prompt_includes_archival_search_results(self):
        """Test that rendered prompt includes Archival Memory search results."""
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from ai_scientist.memory import MemoryManager

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "memory.sqlite"
            cfg = SimpleNamespace(
                core_max_chars=4000,
                recall_max_events=20,
                retrieval_k=8,
                use_fts="off",
                memory_log_enabled=False,
                auto_consolidate=False,
            )
            mem = MemoryManager(db_path, cfg)
            branch = mem.create_branch(None, node_uid="test")
            mem.write_archival(branch, "8 threads optimal for workload", tags=["PERFORMANCE"])

            # Search with task hint related to content
            prompt = mem.render_for_prompt(branch, task_hint="thread optimization", budget_chars=4000)

            # Archival content should be in search results
            self.assertIn("optimal", prompt)


class TestTwoPhasePattern(unittest.TestCase):
    """Test the Two-Phase Pattern as documented in memory-flow.md:296-359.

    Note: Full integration testing of _run_memory_update_phase requires
    mocking LLM calls. These tests verify the helper functions and patterns.
    """

    def test_has_memory_read_results_detects_core_get(self):
        """Test detection of core_get results (triggers Phase 2)."""
        from ai_scientist.treesearch.parallel_agent import _has_memory_read_results

        results = {"core_get": {"key1": "value1"}}
        self.assertTrue(_has_memory_read_results(results))

    def test_has_memory_read_results_detects_archival_search(self):
        """Test detection of archival_search results (triggers Phase 2)."""
        from ai_scientist.treesearch.parallel_agent import _has_memory_read_results

        results = {"archival_search": [{"content": "test"}]}
        self.assertTrue(_has_memory_read_results(results))

    def test_has_memory_read_results_detects_recall_search(self):
        """Test detection of recall_search results (triggers Phase 2)."""
        from ai_scientist.treesearch.parallel_agent import _has_memory_read_results

        results = {"recall_search": [{"kind": "compile_failed"}]}
        self.assertTrue(_has_memory_read_results(results))

    def test_has_memory_read_results_ignores_writes(self):
        """Test that write-only results don't trigger Phase 2."""
        from ai_scientist.treesearch.parallel_agent import _has_memory_read_results

        results = {
            "core_written": {"key1": "value1"},
            "archival_written": [{"text": "test"}]
        }
        self.assertFalse(_has_memory_read_results(results))


class TestPhase1IterativeMemoryFlow(unittest.TestCase):
    """Test Phase 1 iterative memory flow as documented in memory-flow.md:131-197.

    Note: Full integration testing requires mocking LLM and container execution.
    These tests verify the response parsing and memory update extraction.
    """

    def test_phase1_response_format_with_memory(self):
        """Test parsing Phase 1 response with memory_update block."""
        response = '''<memory_update>
{"core": {"numpy_version": "1.24.0"}, "archival": [{"text": "Installed numpy", "tags": ["PHASE1_INSTALL"]}]}
</memory_update>
{"command": "pip install scipy", "done": false, "notes": "Installing scipy next"}'''

        updates = extract_memory_updates(response)
        self.assertIsNotNone(updates)
        self.assertEqual(updates["core"]["numpy_version"], "1.24.0")

        # Clean response should still contain JSON
        cleaned = remove_memory_update_tags(response)
        self.assertIn('"command"', cleaned)
        self.assertIn('"done"', cleaned)

    def test_phase1_response_format_done_true(self):
        """Test parsing Phase 1 final response (done=true)."""
        response = '''<memory_update>
{"core": {"phase1_status": "completed"}}
</memory_update>
{"command": "", "done": true, "notes": "All dependencies installed"}'''

        updates = extract_memory_updates(response)
        self.assertEqual(updates["core"]["phase1_status"], "completed")


class TestBranchInheritanceDocumented(unittest.TestCase):
    """Test branch inheritance as documented in memory-flow-phases.md:473-486."""

    def test_child_inherits_parent_core(self):
        """Child branch should inherit parent's Core Memory."""
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from ai_scientist.memory import MemoryManager

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "memory.sqlite"
            cfg = SimpleNamespace(
                core_max_chars=4000,
                recall_max_events=20,
                retrieval_k=8,
                use_fts="off",
                memory_log_enabled=False,
                auto_consolidate=False,
            )
            mem = MemoryManager(db_path, cfg)

            # root_branch (Phase 0 memory)
            root = mem.create_branch(None, node_uid="root")
            mem.set_core(root, "phase0_plan", "Use OpenMP with 8 threads")

            # node_1_branch (fork from root)
            node1 = mem.create_branch(root, node_uid="node1")
            prompt = mem.render_for_prompt(node1, task_hint="test", budget_chars=4000)

            # Child should see parent's core memory
            self.assertIn("phase0_plan", prompt)
            self.assertIn("OpenMP", prompt)

    def test_siblings_isolated(self):
        """Sibling branches should be isolated from each other."""
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from ai_scientist.memory import MemoryManager

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "memory.sqlite"
            cfg = SimpleNamespace(
                core_max_chars=4000,
                recall_max_events=20,
                retrieval_k=8,
                use_fts="off",
                memory_log_enabled=False,
                auto_consolidate=False,
            )
            mem = MemoryManager(db_path, cfg)

            root = mem.create_branch(None, node_uid="root")

            # Two sibling branches
            node1 = mem.create_branch(root, node_uid="node1")
            node2 = mem.create_branch(root, node_uid="node2")

            # Write to node1
            mem.set_core(node1, "node1_secret", "confidential")
            mem.write_archival(node1, "Node 1 insight", tags=["NODE1"])

            # node2 should NOT see node1's writes
            prompt2 = mem.render_for_prompt(node2, task_hint="test", budget_chars=4000)
            self.assertNotIn("node1_secret", prompt2)
            self.assertNotIn("confidential", prompt2)

    def test_grandchild_inherits_chain(self):
        """Grandchild inherits from both parent and grandparent."""
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from ai_scientist.memory import MemoryManager

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "memory.sqlite"
            cfg = SimpleNamespace(
                core_max_chars=4000,
                recall_max_events=20,
                retrieval_k=8,
                use_fts="off",
                memory_log_enabled=False,
                auto_consolidate=False,
            )
            mem = MemoryManager(db_path, cfg)

            # Chain: root -> node1 -> node3
            root = mem.create_branch(None, node_uid="root")
            mem.set_core(root, "root_config", "baseline")

            node1 = mem.create_branch(root, node_uid="node1")
            mem.set_core(node1, "node1_improvement", "optimization A")

            node3 = mem.create_branch(node1, node_uid="node3")
            prompt = mem.render_for_prompt(node3, task_hint="test", budget_chars=4000)

            # Grandchild sees both root and node1 memories
            self.assertIn("root_config", prompt)
            self.assertIn("node1_improvement", prompt)


if __name__ == "__main__":
    unittest.main()
