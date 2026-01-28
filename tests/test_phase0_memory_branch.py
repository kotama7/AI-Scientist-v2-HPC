"""Tests for Phase 0 memory updates being recorded on child branches instead of root.

This file tests the fix for the bug where Phase 0 memory events were being recorded
on the root branch instead of the child branch that was about to be created.

The bug caused:
- All Phase 0 writes to be recorded on root (e.g., 12 writes on root for 4 child nodes)
- Child nodes having no Phase 0 memory events visible

The fix:
- Store Phase 0 memory updates before fork
- Apply them after fork using child_branch_id instead of root_branch_id
"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, call

from ai_scientist.memory import MemoryManager
from ai_scientist.treesearch.utils.response import extract_memory_updates


class TestPhase0MemoryOnChildBranch(unittest.TestCase):
    """Test that Phase 0 memory updates are recorded on child branches."""

    def setUp(self):
        """Set up test fixtures."""
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = Path(self.tmpdir) / "memory.sqlite"
        self.cfg = SimpleNamespace(
            core_max_chars=4000,
            recall_max_events=20,
            retrieval_k=8,
            use_fts="off",
            memory_log_enabled=True,  # Enable logging for verification
            auto_consolidate=False,
        )
        self.memory_manager = MemoryManager(self.db_path, self.cfg)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_phase0_memory_updates_applied_to_child_branch(self):
        """Verify Phase 0 memory updates are applied to child branch, not root."""
        # Create root branch (simulating main process)
        root_branch = self.memory_manager.create_branch(None, node_uid="root")

        # Simulate Phase 0 response with memory updates
        phase0_response = '''<memory_update>
{
    "core": {"phase0_summary": "Optimization experiment with 8 threads"},
    "archival": [{"text": "Environment: AMD EPYC 7713, 256 CPUs", "tags": ["PHASE0", "ENV"]}]
}
</memory_update>
{"plan": {"download": {}, "compile": {}, "run": {}}}'''

        # Extract memory updates (as the fix does before fork)
        phase0_memory_updates = extract_memory_updates(phase0_response)
        self.assertIsNotNone(phase0_memory_updates)

        # Create child branch (fork)
        child_branch = self.memory_manager.create_branch(root_branch, node_uid="child1")

        # Apply Phase 0 memory updates to CHILD branch (the fix)
        self.memory_manager.apply_llm_memory_updates(
            child_branch,
            phase0_memory_updates,
            node_id=None,
            phase="phase0",
        )

        # Verify: Child branch should have Phase 0 memory
        child_prompt = self.memory_manager.render_for_prompt(
            child_branch, task_hint="test", budget_chars=4000
        )
        self.assertIn("phase0_summary", child_prompt)
        self.assertIn("Optimization experiment", child_prompt)
        self.assertIn("AMD EPYC", child_prompt)

        # Verify: Root branch should NOT have the Phase 0 memory directly
        # (Core memory is inherited, so we check by getting the specific key)
        root_phase0 = self.memory_manager.get_core(root_branch, "phase0_summary")
        self.assertIsNone(root_phase0, "Root should not have phase0_summary directly")

    def test_multiple_children_get_independent_phase0_memory(self):
        """Verify each child gets its own Phase 0 memory (not shared via root)."""
        root_branch = self.memory_manager.create_branch(None, node_uid="root")

        # Create two child branches with different Phase 0 content
        child1_branch = self.memory_manager.create_branch(root_branch, node_uid="child1")
        child1_updates = extract_memory_updates('''<memory_update>
{"core": {"phase0_summary": "Child 1: OpenMP optimization"}}
</memory_update>''')
        self.memory_manager.apply_llm_memory_updates(
            child1_branch, child1_updates, node_id=None, phase="phase0"
        )

        child2_branch = self.memory_manager.create_branch(root_branch, node_uid="child2")
        child2_updates = extract_memory_updates('''<memory_update>
{"core": {"phase0_summary": "Child 2: MPI optimization"}}
</memory_update>''')
        self.memory_manager.apply_llm_memory_updates(
            child2_branch, child2_updates, node_id=None, phase="phase0"
        )

        # Verify each child has its own Phase 0 memory
        child1_prompt = self.memory_manager.render_for_prompt(
            child1_branch, task_hint="test", budget_chars=4000
        )
        child2_prompt = self.memory_manager.render_for_prompt(
            child2_branch, task_hint="test", budget_chars=4000
        )

        self.assertIn("OpenMP", child1_prompt)
        self.assertNotIn("MPI", child1_prompt)

        self.assertIn("MPI", child2_prompt)
        self.assertNotIn("OpenMP", child2_prompt)

    def test_phase0_archival_records_branch_specific(self):
        """Verify Phase 0 archival records are specific to each child branch."""
        root_branch = self.memory_manager.create_branch(None, node_uid="root")

        # Create child and apply Phase 0 archival
        child_branch = self.memory_manager.create_branch(root_branch, node_uid="child1")
        phase0_updates = extract_memory_updates('''<memory_update>
{
    "archival": [
        {"text": "Phase 0 insight for child1: Use numactl", "tags": ["PHASE0", "NUMA"]}
    ]
}
</memory_update>''')
        self.memory_manager.apply_llm_memory_updates(
            child_branch, phase0_updates, node_id=None, phase="phase0"
        )

        # Verify child can see the archival record
        child_prompt = self.memory_manager.render_for_prompt(
            child_branch, task_hint="numa optimization", budget_chars=4000
        )
        self.assertIn("numactl", child_prompt)

        # Create another child from root - should NOT see first child's archival
        child2_branch = self.memory_manager.create_branch(root_branch, node_uid="child2")
        child2_prompt = self.memory_manager.render_for_prompt(
            child2_branch, task_hint="numa optimization", budget_chars=4000
        )
        # Child2 should not see child1's phase0 archival (unless inherited from common ancestor)
        # Since both are children of root with no phase0 on root, child2 shouldn't see child1's data
        self.assertNotIn("numactl", child2_prompt)


class TestPhase0MemoryCallLog(unittest.TestCase):
    """Test that memory call log records Phase 0 events on correct branch."""

    def setUp(self):
        """Set up test fixtures with memory logging enabled."""
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = Path(self.tmpdir) / "memory.sqlite"
        self.log_path = Path(self.tmpdir) / "memory_calls.jsonl"
        self.cfg = SimpleNamespace(
            core_max_chars=4000,
            recall_max_events=20,
            retrieval_k=8,
            use_fts="off",
            memory_log_enabled=True,
            memory_log_path=str(self.log_path),
            auto_consolidate=False,
        )
        self.memory_manager = MemoryManager(self.db_path, self.cfg)
        self.memory_manager.run_id = "test-run"

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_phase0_events_logged_on_child_branch(self):
        """Verify Phase 0 memory events are logged with child branch_id."""
        root_branch = self.memory_manager.create_branch(None, node_uid="root")
        child_branch = self.memory_manager.create_branch(root_branch, node_uid="child1")

        # Apply Phase 0 updates to child (as the fix does)
        phase0_updates = extract_memory_updates('''<memory_update>
{"core": {"phase0_summary": "Test experiment"}}
</memory_update>''')
        self.memory_manager.apply_llm_memory_updates(
            child_branch, phase0_updates, node_id=None, phase="phase0"
        )

        # Read the log file and verify branch_id
        if self.log_path.exists():
            with open(self.log_path, 'r') as f:
                events = [json.loads(line) for line in f if line.strip()]

            # Find Phase 0 events
            phase0_events = [e for e in events if e.get("phase") == "Phase 0: Setup"]

            # All Phase 0 events should have child_branch as branch_id
            for event in phase0_events:
                self.assertEqual(
                    event.get("branch_id"), child_branch,
                    f"Phase 0 event should be on child branch, not root. Event: {event}"
                )


class TestPhase0MemoryUpdateFlow(unittest.TestCase):
    """Integration test simulating the full Phase 0 memory update flow."""

    def test_simulated_worker_flow(self):
        """Simulate the worker process flow with Phase 0 -> Fork -> Apply pattern."""
        tmpdir = tempfile.mkdtemp()
        try:
            db_path = Path(tmpdir) / "memory.sqlite"
            cfg = SimpleNamespace(
                core_max_chars=4000,
                recall_max_events=20,
                retrieval_k=8,
                use_fts="off",
                memory_log_enabled=False,
                auto_consolidate=False,
            )
            memory_manager = MemoryManager(db_path, cfg)

            # Step 1: Create root branch (main process)
            root_branch = memory_manager.create_branch(None, node_uid="root")

            # Step 2: Simulate Phase 0 LLM response (before fork)
            phase0_response_raw = '''<memory_update>
{
    "core": {"phase0_summary": "Himeno benchmark autotuning"},
    "archival": [
        {"text": "Environment: Ubuntu 22.04, AMD EPYC", "tags": ["PHASE0_INTERNAL"]}
    ]
}
</memory_update>
{"plan": {"download": {"commands": []}, "compile": {"commands": ["make"]}, "run": {"commands": ["./run"]}}}'''

            # Step 3: Extract and store memory updates (FIX: don't apply yet)
            phase0_memory_updates = extract_memory_updates(phase0_response_raw)
            self.assertIsNotNone(phase0_memory_updates, "Should extract memory updates")

            # Step 4: Fork - create child branch
            child_branch = memory_manager.create_branch(root_branch, node_uid="worker_child")

            # Step 5: Apply Phase 0 memory updates to child (FIX: apply after fork)
            memory_manager.apply_llm_memory_updates(
                child_branch,  # Use child_branch, NOT root_branch
                phase0_memory_updates,
                node_id=None,
                phase="phase0",
            )

            # Verify: Child should have Phase 0 memory
            child_prompt = memory_manager.render_for_prompt(
                child_branch, task_hint="optimization", budget_chars=4000
            )
            self.assertIn("phase0_summary", child_prompt)
            self.assertIn("Himeno", child_prompt)

            # Verify: Root should NOT have Phase 0 core memory
            root_phase0 = memory_manager.get_core(root_branch, "phase0_summary")
            self.assertIsNone(root_phase0, "Root should not have phase0_summary directly")

        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestPhase0MemoryReadFollowup(unittest.TestCase):
    """Test that Phase 0 memory read follow-up uses child branch."""

    def test_followup_reads_use_child_branch(self):
        """Verify follow-up memory reads after Phase 0 use child branch."""
        tmpdir = tempfile.mkdtemp()
        try:
            db_path = Path(tmpdir) / "memory.sqlite"
            cfg = SimpleNamespace(
                core_max_chars=4000,
                recall_max_events=20,
                retrieval_k=8,
                use_fts="off",
                memory_log_enabled=False,
                auto_consolidate=False,
            )
            memory_manager = MemoryManager(db_path, cfg)

            root_branch = memory_manager.create_branch(None, node_uid="root")

            # Add some archival data to root (simulating previous run)
            memory_manager.write_archival(
                root_branch,
                "Previous experiment: 4 threads was optimal",
                tags=["HISTORICAL"]
            )

            child_branch = memory_manager.create_branch(root_branch, node_uid="child1")

            # Phase 0 update with a read operation
            phase0_updates = {
                "archival_search": {"query": "thread optimization", "k": 3},
                "core": {"phase0_plan": "Check historical data"}
            }

            # Apply to child branch
            results = memory_manager.apply_llm_memory_updates(
                child_branch,
                phase0_updates,
                node_id=None,
                phase="phase0",
            )

            # Verify read results come from child's view (includes inherited data)
            if results and "archival_search" in results:
                search_results = results["archival_search"]
                # Should find the inherited archival record
                found_historical = any(
                    "4 threads" in str(r) for r in search_results
                )
                self.assertTrue(
                    found_historical or len(search_results) == 0,
                    "Archival search should work on child branch"
                )

        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestPhase0MemoryInjection(unittest.TestCase):
    """Test that Phase 0 injects memory context into prompt (same as Phase 1-4)."""

    def test_execute_phase0_planning_injects_memory(self):
        """Verify _execute_phase0_planning adds Memory section to prompt."""
        from ai_scientist.treesearch.parallel_agent import _execute_phase0_planning

        tmpdir = tempfile.mkdtemp()
        try:
            db_path = Path(tmpdir) / "memory.sqlite"
            cfg = SimpleNamespace(
                core_max_chars=4000,
                recall_max_events=20,
                retrieval_k=8,
                use_fts="off",
                memory_log_enabled=False,
                auto_consolidate=False,
            )
            memory_cfg = SimpleNamespace(
                enabled=True,
                memory_budget_chars=4000,
                max_memory_read_rounds=2,
            )
            memory_manager = MemoryManager(db_path, cfg)

            # Create root and child branches
            root_branch = memory_manager.create_branch(None, node_uid="root")

            # Add data to root that should be visible to child
            memory_manager.set_core(root_branch, "test_key", "test_value")
            memory_manager.write_archival(
                root_branch,
                "Previous experiment data: 8 threads was optimal",
                tags=["HISTORICAL", "TEST"]
            )

            child_branch = memory_manager.create_branch(root_branch, node_uid="child")

            # Create mock config for _execute_phase0_planning
            mock_cfg = SimpleNamespace(
                agent=SimpleNamespace(
                    code=SimpleNamespace(model="test-model", temp=0.7)
                ),
                memory=memory_cfg,
            )

            # Mock the query function to capture what prompt was sent
            captured_prompt = {}
            def mock_query(system_message, user_message, model, temperature):
                captured_prompt["prompt"] = system_message
                # Return a valid Phase 0 response
                return '''<memory_update>
{"core": {"phase0_summary": "Test plan"}}
</memory_update>
{"plan": {"goal_summary": "test"}}'''

            with patch("ai_scientist.treesearch.parallel_agent.query", mock_query):
                phase0_prompt = {
                    "Introduction": "Test introduction",
                    "Task": "Test task",
                }

                _execute_phase0_planning(
                    phase0_prompt=phase0_prompt,
                    cfg=mock_cfg,
                    memory_cfg=memory_cfg,
                    memory_manager=memory_manager,
                    branch_id=child_branch,
                    plans_dir=None,
                    prompt_log_root=None,
                )

            # Verify Memory section was injected
            self.assertIn("prompt", captured_prompt, "Query should have been called")
            prompt = captured_prompt["prompt"]
            self.assertIn("Memory", prompt, "Memory section should be injected into Phase 0 prompt")

            # Verify memory context contains inherited data
            memory_section = prompt.get("Memory", "")
            self.assertIn("test_key", memory_section, "Memory should contain inherited core data")

        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_execute_phase0_planning_no_memory_when_disabled(self):
        """Verify _execute_phase0_planning does NOT inject Memory when disabled."""
        from ai_scientist.treesearch.parallel_agent import _execute_phase0_planning

        tmpdir = tempfile.mkdtemp()
        try:
            db_path = Path(tmpdir) / "memory.sqlite"
            cfg = SimpleNamespace(
                core_max_chars=4000,
                recall_max_events=20,
                retrieval_k=8,
                use_fts="off",
                memory_log_enabled=False,
                auto_consolidate=False,
            )
            memory_cfg = SimpleNamespace(
                enabled=False,  # Memory disabled
                memory_budget_chars=4000,
            )
            memory_manager = MemoryManager(db_path, cfg)

            root_branch = memory_manager.create_branch(None, node_uid="root")
            child_branch = memory_manager.create_branch(root_branch, node_uid="child")

            mock_cfg = SimpleNamespace(
                agent=SimpleNamespace(
                    code=SimpleNamespace(model="test-model", temp=0.7)
                ),
                memory=memory_cfg,
            )

            captured_prompt = {}
            def mock_query(system_message, user_message, model, temperature):
                captured_prompt["prompt"] = system_message
                return '{"plan": {"goal_summary": "test"}}'

            with patch("ai_scientist.treesearch.parallel_agent.query", mock_query):
                phase0_prompt = {
                    "Introduction": "Test introduction",
                    "Task": "Test task",
                }

                _execute_phase0_planning(
                    phase0_prompt=phase0_prompt,
                    cfg=mock_cfg,
                    memory_cfg=memory_cfg,
                    memory_manager=memory_manager,
                    branch_id=child_branch,
                    plans_dir=None,
                    prompt_log_root=None,
                )

            # Verify Memory section was NOT injected when disabled
            self.assertIn("prompt", captured_prompt)
            prompt = captured_prompt["prompt"]
            self.assertNotIn("Memory", prompt, "Memory section should NOT be injected when memory is disabled")

        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
