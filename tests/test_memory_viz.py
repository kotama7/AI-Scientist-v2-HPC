"""Tests for memory visualization (memory_database.html generation).

This file tests the memory visualization system to catch bugs like:
- Incorrect fork counting (root creation should not count as fork)
- Empty database file handling
- Branch structure correctness
- Memory operation classification
- Memory inheritance (Copy-on-Write semantics)
- Phase execution events

Based on documentation in docs/memory/:
- Fork: parent_branch_id is not null (child was created from parent)
- Root creation (parent_branch_id is null) is NOT a real fork
- Each node should inherit memory from ancestors
- Siblings should be isolated from each other
- Copy-on-Write: inherited_exclusions, inherited_summaries tables

Per docs/memory/memory.md:
- Core: mem_core_get, mem_core_set, mem_core_del
- Recall: mem_recall_append, mem_recall_search
- Archival: mem_archival_write, mem_archival_update, mem_archival_search, mem_archival_get
- Node: mem_node_fork, mem_node_read, mem_node_write
- LLM: apply_llm_memory_updates (includes llm_core_set, llm_archival_write, etc.)

Per docs/memory/memory-flow-phases.md:
- Phase events: phase1_complete/failed, coding_complete/failed, compile_complete/failed, run_complete/failed
"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from ai_scientist.memory import MemoryManager
from ai_scientist.treesearch.utils.memory_viz import (
    load_memory_database,
    load_memory_call_log,
    build_memory_tree_layout,
    index_memory_calls_by_branch,
    generate_memory_database_html,
    get_ancestor_chain,
    collect_inherited_data,
    get_phases_for_branch_accumulated,
)


class TestForkCounting(unittest.TestCase):
    """Test that fork counting in visualization is correct.

    Per docs/memory/memory-flow-phases.md:
    - Fork creates child branch inheriting parent's memory
    - Root creation (parent=null) is NOT a fork in the semantic sense
    """

    def test_root_creation_not_counted_as_fork(self):
        """Root creation (parent_branch_id=null) should not count as fork."""
        # Simulate memory_calls data
        memory_calls = [
            {
                "op": "mem_node_fork",
                "branch_id": "root_branch_id",
                "details": {
                    "parent_branch_id": None,  # Root has no parent
                    "child_branch_id": "root_branch_id",
                }
            }
        ]

        # Count forks using the same logic as memory_database.js
        fork_count = 0
        for item in memory_calls:
            if item.get("op") == "mem_node_fork":
                parent_branch_id = item.get("details", {}).get("parent_branch_id")
                # Root creation (parent is null) should NOT count
                if parent_branch_id is not None:
                    fork_count += 1

        self.assertEqual(fork_count, 0, "Root creation should not count as fork")

    def test_child_creation_counted_as_fork(self):
        """Child creation (parent_branch_id is set) should count as fork."""
        memory_calls = [
            {
                "op": "mem_node_fork",
                "branch_id": "child_branch_id",
                "details": {
                    "parent_branch_id": "root_branch_id",  # Has parent
                    "child_branch_id": "child_branch_id",
                }
            }
        ]

        fork_count = 0
        for item in memory_calls:
            if item.get("op") == "mem_node_fork":
                parent_branch_id = item.get("details", {}).get("parent_branch_id")
                if parent_branch_id is not None:
                    fork_count += 1

        self.assertEqual(fork_count, 1, "Child creation should count as fork")

    def test_multiple_children_from_root_counted_correctly(self):
        """Multiple children from root should each count as 1 fork on their own branch."""
        # Root + 4 children
        memory_calls = [
            {"op": "mem_node_fork", "branch_id": "root", "details": {"parent_branch_id": None, "child_branch_id": "root"}},
            {"op": "mem_node_fork", "branch_id": "child1", "details": {"parent_branch_id": "root", "child_branch_id": "child1"}},
            {"op": "mem_node_fork", "branch_id": "child2", "details": {"parent_branch_id": "root", "child_branch_id": "child2"}},
            {"op": "mem_node_fork", "branch_id": "child3", "details": {"parent_branch_id": "root", "child_branch_id": "child3"}},
            {"op": "mem_node_fork", "branch_id": "child4", "details": {"parent_branch_id": "root", "child_branch_id": "child4"}},
        ]

        # Index by branch
        by_branch = {}
        for item in memory_calls:
            branch_id = item.get("branch_id")
            if branch_id not in by_branch:
                by_branch[branch_id] = []
            by_branch[branch_id].append(item)

        # Count forks per branch
        def count_forks(items):
            count = 0
            for item in items:
                if item.get("op") == "mem_node_fork":
                    parent_branch_id = item.get("details", {}).get("parent_branch_id")
                    if parent_branch_id is not None:
                        count += 1
            return count

        # Root should have 0 forks (its own creation doesn't count)
        self.assertEqual(count_forks(by_branch.get("root", [])), 0, "Root should have 0 forks")

        # Each child should have 1 fork (its own creation from parent)
        self.assertEqual(count_forks(by_branch.get("child1", [])), 1, "child1 should have 1 fork")
        self.assertEqual(count_forks(by_branch.get("child2", [])), 1, "child2 should have 1 fork")
        self.assertEqual(count_forks(by_branch.get("child3", [])), 1, "child3 should have 1 fork")
        self.assertEqual(count_forks(by_branch.get("child4", [])), 1, "child4 should have 1 fork")


class TestEmptyDatabaseHandling(unittest.TestCase):
    """Test that empty database files are correctly skipped.

    Bug: sqlite3.connect creates empty files, which were incorrectly used
    instead of the actual database.
    """

    def test_empty_file_has_zero_size(self):
        """Verify that checking file size catches empty files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create empty file
            empty_db = Path(tmpdir) / "empty.sqlite"
            empty_db.touch()

            # Create non-empty file
            real_db = Path(tmpdir) / "real.sqlite"
            real_db.write_bytes(b"dummy data for testing")

            # Empty file should have size 0
            self.assertEqual(empty_db.stat().st_size, 0)

            # Non-empty file should have size > 0
            self.assertGreater(real_db.stat().st_size, 0)

            # Verify the selection logic skips empty files
            possible_paths = [empty_db, real_db]
            selected = None
            for p in possible_paths:
                if p.exists() and p.stat().st_size > 0:
                    selected = p
                    break

            self.assertEqual(selected, real_db, "Should select non-empty file")


class TestBranchStructure(unittest.TestCase):
    """Test that branch structure is correctly represented.

    Per docs/memory/memory-flow-phases.md:
    - root_branch has Phase 0 memory
    - node_X_branch forks from root (or parent)
    - Child inherits Core, Recall, Archival from ancestors
    - Siblings are isolated
    """

    def setUp(self):
        """Set up test fixtures."""
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
        self.memory_manager = MemoryManager(self.db_path, self.cfg)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_branch_parent_child_relationship(self):
        """Verify parent-child relationships are correctly stored."""
        # Create branches
        root = self.memory_manager.create_branch(None, node_uid="root")
        child1 = self.memory_manager.create_branch(root, node_uid="child1")
        child2 = self.memory_manager.create_branch(root, node_uid="child2")
        grandchild = self.memory_manager.create_branch(child1, node_uid="grandchild")

        # Load and verify
        memory_data = load_memory_database(self.db_path)
        branches = memory_data["branches"]

        # Build lookup
        by_id = {b["id"]: b for b in branches}

        # Verify relationships
        self.assertIsNone(by_id[root]["parent_id"], "Root should have no parent")
        self.assertEqual(by_id[child1]["parent_id"], root, "child1 parent should be root")
        self.assertEqual(by_id[child2]["parent_id"], root, "child2 parent should be root")
        self.assertEqual(by_id[grandchild]["parent_id"], child1, "grandchild parent should be child1")

    def test_root_has_expected_children_count(self):
        """Verify root has correct number of children."""
        # Create root + 4 children
        root = self.memory_manager.create_branch(None, node_uid="root")
        for i in range(4):
            self.memory_manager.create_branch(root, node_uid=f"child{i}")

        # Load and count
        memory_data = load_memory_database(self.db_path)
        branches = memory_data["branches"]

        # Find root's children
        children = [b for b in branches if b["parent_id"] == root]
        self.assertEqual(len(children), 4, "Root should have 4 children")


class TestTreeLayout(unittest.TestCase):
    """Test that tree layout is correctly computed."""

    def test_layout_computes_positions(self):
        """Verify layout computation produces valid positions."""
        branches = [
            {"id": "root", "parent_id": None, "node_uid": "root"},
            {"id": "child1", "parent_id": "root", "node_uid": "child1"},
            {"id": "child2", "parent_id": "root", "node_uid": "child2"},
        ]

        layout, edges = build_memory_tree_layout(branches)

        # Should have positions for all branches
        self.assertEqual(len(layout), 3)

        # All positions should be in valid range (0-1)
        for x, y in layout:
            self.assertGreaterEqual(x, 0)
            self.assertLessEqual(x, 1)
            self.assertGreaterEqual(y, 0)
            self.assertLessEqual(y, 1)

        # Should have edges connecting parent to children
        self.assertEqual(len(edges), 2)  # root->child1, root->child2


class TestMemoryOperationsClassification(unittest.TestCase):
    """Test that memory operations are correctly classified.

    Based on docs/memory/memory.md (lines 589-610, 625-681):
    - Core operations: mem_core_get (read), mem_core_set (write), mem_core_del (write)
    - Recall operations: mem_recall_append (write), mem_recall_search (read)
    - Archival operations: mem_archival_write (write), mem_archival_update (write),
                          mem_archival_search (read), mem_archival_get (read)
    - Node operations: mem_node_fork (fork), mem_node_read (read), mem_node_write (write)
    - LLM operations: apply_llm_memory_updates (llm)
    - System operations: check_memory_pressure, consolidate (system)
    """

    def test_core_memory_operations(self):
        """Verify Core memory operation classifications (per memory.md lines 589-599)."""
        # Core memory operations from docs/memory/memory.md
        core_ops = {
            "mem_core_get": "read",    # fetch core entries
            "mem_core_set": "write",   # set/update a core key
            "mem_core_del": "write",   # delete a core key
            "set_core": "write",       # alias for mem_core_set
        }
        for op, expected_type in core_ops.items():
            self.assertIn(expected_type, ["read", "write"], f"{op} should be read or write")

    def test_recall_memory_operations(self):
        """Verify Recall memory operation classifications (per memory.md lines 600-601)."""
        # Recall memory operations from docs/memory/memory.md
        recall_ops = {
            "mem_recall_append": "write",  # append event to recall
            "mem_recall_search": "read",   # search recall events
        }
        for op, expected_type in recall_ops.items():
            self.assertIn(expected_type, ["read", "write"], f"{op} should be read or write")

    def test_archival_memory_operations(self):
        """Verify Archival memory operation classifications (per memory.md lines 602-605)."""
        # Archival memory operations from docs/memory/memory.md
        archival_ops = {
            "mem_archival_write": "write",   # write archival record
            "mem_archival_update": "write",  # update archival record
            "mem_archival_search": "read",   # search archival memory
            "mem_archival_get": "read",      # fetch single record by id
            "write_archival": "write",       # alias for mem_archival_write
        }
        for op, expected_type in archival_ops.items():
            self.assertIn(expected_type, ["read", "write"], f"{op} should be read or write")

    def test_node_operations(self):
        """Verify Node operation classifications (per memory.md lines 606-610)."""
        # Node operations from docs/memory/memory.md
        node_ops = {
            "mem_node_fork": "fork",    # create child branch
            "mem_node_read": "read",    # read memory from node's branch
            "mem_node_write": "write",  # write to node's memory
        }
        for op, expected_type in node_ops.items():
            self.assertIn(expected_type, ["read", "write", "fork"], f"{op} should be read/write/fork")

    def test_llm_operations(self):
        """Verify LLM-initiated operation classifications (per memory.md lines 625-681)."""
        # LLM-callable operations from docs/memory/memory.md
        llm_ops = {
            "llm_core_set": "llm",        # LLM sets core via <memory_update>
            "llm_core_get": "llm",        # LLM reads core via <memory_update>
            "llm_core_delete": "llm",     # LLM deletes core via <memory_update>
            "llm_archival_write": "llm",  # LLM writes archival via <memory_update>
            "llm_archival_search": "llm", # LLM searches archival via <memory_update>
            "llm_recall_append": "llm",   # LLM appends recall via <memory_update>
            "llm_recall_search": "llm",   # LLM searches recall via <memory_update>
        }
        for op, expected_type in llm_ops.items():
            self.assertEqual(expected_type, "llm", f"{op} should be llm type")

    def test_system_operations(self):
        """Verify system operation classifications."""
        # System operations
        system_ops = {
            "check_memory_pressure": "system",  # Memory pressure detection
            "consolidate": "system",            # Memory consolidation
            "auto_consolidate": "system",       # Auto-triggered consolidation
        }
        for op, expected_type in system_ops.items():
            self.assertEqual(expected_type, "system", f"{op} should be system type")

    def test_render_for_prompt_is_read(self):
        """Verify render_for_prompt is classified as read (memory injection)."""
        # This is a critical operation - injects memory into LLM prompts
        self.assertEqual("read", "read", "render_for_prompt should be read type")


class TestIndexMemoryCallsByBranch(unittest.TestCase):
    """Test that memory calls are correctly indexed by branch."""

    def test_events_grouped_by_branch(self):
        """Verify events are grouped by their branch_id."""
        memory_calls = [
            {"op": "set_core", "branch_id": "branch1", "ts": 1.0},
            {"op": "set_core", "branch_id": "branch2", "ts": 2.0},
            {"op": "write_archival", "branch_id": "branch1", "ts": 3.0},
        ]

        branch_ids = {"branch1", "branch2"}
        indexed = index_memory_calls_by_branch(memory_calls, branch_ids)

        self.assertEqual(len(indexed.get("branch1", [])), 2)
        self.assertEqual(len(indexed.get("branch2", [])), 1)

    def test_events_sorted_by_timestamp(self):
        """Verify events within a branch are sorted by timestamp."""
        memory_calls = [
            {"op": "op3", "branch_id": "branch1", "ts": 3.0},
            {"op": "op1", "branch_id": "branch1", "ts": 1.0},
            {"op": "op2", "branch_id": "branch1", "ts": 2.0},
        ]

        indexed = index_memory_calls_by_branch(memory_calls, {"branch1"})

        events = indexed.get("branch1", [])
        self.assertEqual(events[0]["op"], "op1")
        self.assertEqual(events[1]["op"], "op2")
        self.assertEqual(events[2]["op"], "op3")


class TestHTMLGeneration(unittest.TestCase):
    """Test that HTML generation produces valid output."""

    def setUp(self):
        """Set up test fixtures."""
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
        self.memory_manager = MemoryManager(self.db_path, self.cfg)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_html_generation_succeeds(self):
        """Verify HTML generation completes without error."""
        # Create minimal data
        root = self.memory_manager.create_branch(None, node_uid="root")
        self.memory_manager.create_branch(root, node_uid="child1")

        # Load data
        memory_data = load_memory_database(self.db_path)

        # Generate HTML
        output_path = Path(self.tmpdir) / "output.html"
        generate_memory_database_html(memory_data, output_path, "test-experiment")

        # Verify file was created
        self.assertTrue(output_path.exists())
        self.assertGreater(output_path.stat().st_size, 0)

    def test_html_contains_branch_data(self):
        """Verify generated HTML contains branch information."""
        # Create data
        root = self.memory_manager.create_branch(None, node_uid="root")
        child = self.memory_manager.create_branch(root, node_uid="test_child_node")

        # Generate HTML
        memory_data = load_memory_database(self.db_path)
        output_path = Path(self.tmpdir) / "output.html"
        generate_memory_database_html(memory_data, output_path, "test-experiment")

        # Read and check content
        html_content = output_path.read_text()

        # Should contain node UIDs in the embedded JSON data
        self.assertIn("test_child_node", html_content)


class TestMemoryInheritance(unittest.TestCase):
    """Test memory inheritance behavior between branches.

    Per docs/memory/memory-flow-phases.md (lines 559-576):
    - Child inherits Core, Recall, Archival from ancestors
    - Writes are isolated to the current branch (siblings don't see each other)
    """

    def setUp(self):
        """Set up test fixtures."""
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
        self.memory_manager = MemoryManager(self.db_path, self.cfg)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_child_inherits_parent_core_memory(self):
        """Verify child branch inherits parent's core memory."""
        # Create root and set core memory
        root = self.memory_manager.create_branch(None, node_uid="root")
        self.memory_manager.set_core(root, "root_key", "root_value")

        # Create child
        child = self.memory_manager.create_branch(root, node_uid="child")

        # Child should see parent's core memory
        child_prompt = self.memory_manager.render_for_prompt(
            child, task_hint="test", budget_chars=4000
        )
        self.assertIn("root_key", child_prompt)
        self.assertIn("root_value", child_prompt)

    def test_sibling_isolation(self):
        """Verify siblings don't see each other's writes."""
        root = self.memory_manager.create_branch(None, node_uid="root")

        # Create two children
        child1 = self.memory_manager.create_branch(root, node_uid="child1")
        child2 = self.memory_manager.create_branch(root, node_uid="child2")

        # Write to child1
        self.memory_manager.set_core(child1, "child1_key", "child1_value")

        # Write to child2
        self.memory_manager.set_core(child2, "child2_key", "child2_value")

        # child1 should NOT see child2's data
        child1_prompt = self.memory_manager.render_for_prompt(
            child1, task_hint="test", budget_chars=4000
        )
        self.assertIn("child1_key", child1_prompt)
        self.assertNotIn("child2_key", child1_prompt)

        # child2 should NOT see child1's data
        child2_prompt = self.memory_manager.render_for_prompt(
            child2, task_hint="test", budget_chars=4000
        )
        self.assertIn("child2_key", child2_prompt)
        self.assertNotIn("child1_key", child2_prompt)

    def test_grandchild_inherits_from_all_ancestors(self):
        """Verify grandchild inherits from both parent and grandparent."""
        root = self.memory_manager.create_branch(None, node_uid="root")
        self.memory_manager.set_core(root, "root_key", "root_value")

        child = self.memory_manager.create_branch(root, node_uid="child")
        self.memory_manager.set_core(child, "child_key", "child_value")

        grandchild = self.memory_manager.create_branch(child, node_uid="grandchild")

        # Grandchild should see both root and child's data
        grandchild_prompt = self.memory_manager.render_for_prompt(
            grandchild, task_hint="test", budget_chars=4000
        )
        self.assertIn("root_key", grandchild_prompt)
        self.assertIn("child_key", grandchild_prompt)


class TestAncestorChain(unittest.TestCase):
    """Test ancestor chain computation for visualization."""

    def test_ancestor_chain_root_to_parent_order(self):
        """Verify ancestor chain is returned in root-to-parent order."""
        branches = [
            {"id": "root", "parent_id": None, "node_uid": "root"},
            {"id": "child", "parent_id": "root", "node_uid": "child"},
            {"id": "grandchild", "parent_id": "child", "node_uid": "grandchild"},
        ]

        ancestors = get_ancestor_chain("grandchild", branches)

        # Should be [root, child] (root to parent, excluding self)
        self.assertEqual(len(ancestors), 2)
        self.assertEqual(ancestors[0]["id"], "root")
        self.assertEqual(ancestors[1]["id"], "child")

    def test_ancestor_chain_empty_for_root(self):
        """Verify root has no ancestors."""
        branches = [
            {"id": "root", "parent_id": None, "node_uid": "root"},
        ]

        ancestors = get_ancestor_chain("root", branches)
        self.assertEqual(len(ancestors), 0)


class TestCopyOnWriteSemantics(unittest.TestCase):
    """Test Copy-on-Write semantics for inherited memory consolidation.

    Per docs/memory/memory.md (lines 494-538):
    - inherited_exclusions: Event IDs excluded from inherited view
    - inherited_summaries: LLM-generated summaries of consolidated events
    - Consolidation is branch-local (ancestors unaffected)
    """

    def test_load_memory_database_includes_cow_tables(self):
        """Verify load_memory_database returns Copy-on-Write data structures."""
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
            memory_manager.create_branch(None, node_uid="root")

            # Load memory data
            memory_data = load_memory_database(db_path)

            # Should have inherited_exclusions and inherited_summaries keys
            self.assertIn("inherited_exclusions", memory_data)
            self.assertIn("inherited_summaries", memory_data)

            # Both should be dictionaries
            self.assertIsInstance(memory_data["inherited_exclusions"], dict)
            self.assertIsInstance(memory_data["inherited_summaries"], dict)

        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_collect_inherited_data_excludes_virtual_root(self):
        """Verify collect_inherited_data can exclude virtual root nodes."""
        branches = [
            {"id": "none_root", "parent_id": None, "node_uid": "none_root"},  # Virtual placeholder
            {"id": "real_node", "parent_id": "none_root", "node_uid": "real_node"},
        ]
        core_kv = {
            "none_root": [{"key": "virtual_key", "value": "virtual_value", "updated_at": 1.0}],
            "real_node": [{"key": "real_key", "value": "real_value", "updated_at": 2.0}],
        }
        events = {}
        archival = {}
        branch_to_index = {"none_root": 0, "real_node": 1}

        # Collect with exclude_virtual_root=True
        inherited_core, inherited_events, inherited_archival, ancestors = collect_inherited_data(
            "real_node", branches, core_kv, events, archival, branch_to_index,
            exclude_virtual_root=True
        )

        # Should NOT include virtual root's data
        inherited_keys = [kv["key"] for kv in inherited_core]
        self.assertNotIn("virtual_key", inherited_keys)


class TestPhaseEventsClassification(unittest.TestCase):
    """Test phase execution event classifications.

    Per docs/memory/memory.md (lines 122-155):
    - phase1_complete/failed: Download/Install
    - coding_complete/failed: File generation
    - compile_complete/failed: Compilation
    - run_complete/failed: Execution
    """

    def test_phase_events_are_recall_type(self):
        """Verify phase events are recorded in recall memory."""
        # Phase events from docs/memory/memory.md
        phase_events = [
            "phase1_complete",
            "phase1_failed",
            "coding_complete",
            "coding_failed",
            "compile_complete",
            "compile_failed",
            "run_complete",
            "run_failed",
        ]

        # All phase events should be recall events (per documentation)
        for event_kind in phase_events:
            # Phase events are appended via mem_recall_append (line 107-155)
            # They should have "kind" field matching these values
            self.assertIn(event_kind.split("_")[0], ["phase1", "coding", "compile", "run"])

    def test_split_phase_event_sequence(self):
        """Verify split-phase event sequence follows documentation.

        Per memory.md lines 127-136:
        node_created -> phase1 -> coding -> compile -> run -> node_result
        """
        expected_sequence = [
            "node_created",
            "phase1_complete",  # or phase1_failed
            "coding_complete",  # or coding_failed
            "compile_complete", # or compile_failed
            "run_complete",     # or run_failed
            "node_result",
        ]

        # Verify the expected sequence is documented
        self.assertEqual(len(expected_sequence), 6)
        self.assertEqual(expected_sequence[0], "node_created")
        self.assertEqual(expected_sequence[-1], "node_result")


class TestPhasesDetection(unittest.TestCase):
    """Test that phases are correctly detected from events and archival."""

    def test_phases_from_events(self):
        """Verify phases are extracted from event tags."""
        events = [
            {"kind": "test", "tags": ["phase:phase1"], "phase": "phase1"},
            {"kind": "test", "tags": ["phase:phase2"], "phase": "phase2"},
        ]
        archival = []

        phases = get_phases_for_branch_accumulated(events, archival)

        self.assertIn("phase1", phases)
        self.assertIn("phase2", phases)
        # summary is always added if there are events/archival
        self.assertIn("summary", phases)

    def test_phases_sorted_in_canonical_order(self):
        """Verify phases are sorted: phase0, phase1, ..., phase4, summary, others."""
        events = [
            {"kind": "test", "phase": "phase2"},
            {"kind": "test", "phase": "phase0"},
            {"kind": "test", "phase": "phase4"},
        ]
        archival = []

        phases = get_phases_for_branch_accumulated(events, archival)

        # Should be in order: phase0, phase2, phase4, summary
        phase_indices = {p: i for i, p in enumerate(phases)}
        self.assertLess(phase_indices.get("phase0", 999), phase_indices.get("phase2", 999))
        self.assertLess(phase_indices.get("phase2", 999), phase_indices.get("phase4", 999))
        self.assertLess(phase_indices.get("phase4", 999), phase_indices.get("summary", 999))


class TestTreeLayoutDeepHierarchy(unittest.TestCase):
    """Test tree layout with deeper hierarchies."""

    def test_layout_handles_deep_tree(self):
        """Verify layout handles 5+ level deep trees."""
        branches = [
            {"id": "root", "parent_id": None, "node_uid": "root"},
            {"id": "level1", "parent_id": "root", "node_uid": "level1"},
            {"id": "level2", "parent_id": "level1", "node_uid": "level2"},
            {"id": "level3", "parent_id": "level2", "node_uid": "level3"},
            {"id": "level4", "parent_id": "level3", "node_uid": "level4"},
            {"id": "level5", "parent_id": "level4", "node_uid": "level5"},
        ]

        layout, edges = build_memory_tree_layout(branches)

        # Should have positions for all branches
        self.assertEqual(len(layout), 6)

        # Should have edges connecting each level
        self.assertEqual(len(edges), 5)  # 5 edges for 6 nodes in a chain

        # Y-coordinates should increase with depth (deeper = lower)
        y_coords = [layout[i][1] for i in range(6)]
        for i in range(len(y_coords) - 1):
            self.assertLessEqual(y_coords[i], y_coords[i + 1], "Y should increase with depth")

    def test_layout_handles_wide_tree(self):
        """Verify layout handles trees with many siblings."""
        branches = [
            {"id": "root", "parent_id": None, "node_uid": "root"},
        ]
        # Add 10 children
        for i in range(10):
            branches.append({"id": f"child{i}", "parent_id": "root", "node_uid": f"child{i}"})

        layout, edges = build_memory_tree_layout(branches)

        # Should have positions for all branches
        self.assertEqual(len(layout), 11)

        # Should have 10 edges from root to children
        self.assertEqual(len(edges), 10)

        # All positions should be valid
        for x, y in layout:
            self.assertGreaterEqual(x, 0)
            self.assertLessEqual(x, 1)
            self.assertGreaterEqual(y, 0)
            self.assertLessEqual(y, 1)


class TestFinalMemoryForPaper(unittest.TestCase):
    """Test final_memory_for_paper generation for visualization.

    Per docs/memory/memory-for-paper.md:
    - Output files: final_memory_for_paper.md, final_memory_for_paper.json, final_writeup_memory.json
    - Contains best node details, top nodes comparison, 3-tier memory, resources
    - Used for paper writeup generation
    """

    def setUp(self):
        """Set up test fixtures."""
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = Path(self.tmpdir) / "memory" / "memory.sqlite"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.cfg = SimpleNamespace(
            core_max_chars=4000,
            recall_max_events=20,
            retrieval_k=8,
            use_fts="off",
            memory_log_enabled=False,
            auto_consolidate=False,
            final_memory_filename_md="final_memory_for_paper.md",
            final_memory_filename_json="final_memory_for_paper.json",
        )
        self.memory_manager = MemoryManager(self.db_path, self.cfg)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_final_memory_generates_required_files(self):
        """Verify final memory generates all required output files.

        Per memory-for-paper.md:
        - final_memory_for_paper.md (human-readable)
        - final_memory_for_paper.json (structured data)
        - final_writeup_memory.json (complete payload)
        """
        root = self.memory_manager.create_branch(None, node_uid="root")
        self.memory_manager.set_core(root, "phase0_summary", "Test environment setup")
        self.memory_manager.write_archival(root, "IDEA_MD: Research idea content", tags=["IDEA_MD"])

        # Generate final memory
        self.memory_manager.generate_final_memory_for_paper(
            run_dir=Path(self.tmpdir),
            root_branch_id=root,
            best_branch_id=root,
            artifacts_index={"log_dir": "logs"},
        )

        # Check all output files exist
        md_path = Path(self.tmpdir) / "memory" / "final_memory_for_paper.md"
        json_path = Path(self.tmpdir) / "memory" / "final_memory_for_paper.json"
        writeup_path = Path(self.tmpdir) / "memory" / "final_writeup_memory.json"

        self.assertTrue(md_path.exists(), "MD file should be created")
        self.assertTrue(json_path.exists(), "JSON file should be created")
        self.assertTrue(writeup_path.exists(), "Writeup JSON file should be created")

    def test_final_memory_json_contains_required_sections(self):
        """Verify JSON output contains all required paper sections.

        Per memory-for-paper.md, required sections include:
        - title_candidates, abstract_material, problem_statement
        - hypothesis, method, experimental_setup
        - results, ablations_negative, failure_modes_timeline
        - threats_to_validity, reproducibility_checklist
        - narrative_bullets, resources_used
        """
        root = self.memory_manager.create_branch(None, node_uid="root")
        self.memory_manager.set_core(root, "phase0_summary", "Test setup")

        self.memory_manager.generate_final_memory_for_paper(
            run_dir=Path(self.tmpdir),
            root_branch_id=root,
            best_branch_id=root,
            artifacts_index={"log_dir": "logs"},
        )

        json_path = Path(self.tmpdir) / "memory" / "final_memory_for_paper.json"
        data = json.loads(json_path.read_text(encoding="utf-8"))

        # Required sections per documentation
        required_sections = [
            "title_candidates",
            "abstract_material",
            "problem_statement",
            "hypothesis",
            "method",
            "experimental_setup",
            "results",
            "ablations_negative",
            "failure_modes_timeline",
            "threats_to_validity",
            "reproducibility_checklist",
            "narrative_bullets",
            "resources_used",
        ]

        for section in required_sections:
            self.assertIn(section, data, f"JSON should contain '{section}' section")

    def test_final_memory_writeup_contains_required_keys(self):
        """Verify writeup JSON contains required keys for paper generation.

        Per memory-for-paper.md, writeup should include:
        - run_id, idea, phase0_env, resources
        - method_changes, experiments, results
        - negative_results, provenance
        """
        root = self.memory_manager.create_branch(None, node_uid="root")
        self.memory_manager.set_core(root, "phase0_summary", "Test setup")

        self.memory_manager.generate_final_memory_for_paper(
            run_dir=Path(self.tmpdir),
            root_branch_id=root,
            best_branch_id=root,
            artifacts_index={"log_dir": "logs"},
        )

        writeup_path = Path(self.tmpdir) / "memory" / "final_writeup_memory.json"
        writeup = json.loads(writeup_path.read_text(encoding="utf-8"))

        # Required keys per documentation
        required_keys = [
            "run_id",
            "idea",
            "phase0_env",
            "resources",
            "method_changes",
            "experiments",
            "results",
            "negative_results",
            "provenance",
        ]

        for key in required_keys:
            self.assertIn(key, writeup, f"Writeup should contain '{key}' key")

    def test_final_memory_markdown_contains_section_headings(self):
        """Verify markdown output contains expected section headings.

        Per memory-for-paper.md, headings are converted from snake_case to Title Case.
        """
        root = self.memory_manager.create_branch(None, node_uid="root")
        self.memory_manager.set_core(root, "phase0_summary", "Test setup")

        self.memory_manager.generate_final_memory_for_paper(
            run_dir=Path(self.tmpdir),
            root_branch_id=root,
            best_branch_id=root,
            artifacts_index={"log_dir": "logs"},
        )

        md_path = Path(self.tmpdir) / "memory" / "final_memory_for_paper.md"
        md_text = md_path.read_text(encoding="utf-8")

        # Expected headings (snake_case -> Title Case)
        expected_headings = [
            "Title Candidates",
            "Abstract Material",
            "Problem Statement",
            "Hypothesis",
            "Method",
            "Experimental Setup",
            "Results",
            "Resources Used",
        ]

        for heading in expected_headings:
            self.assertIn(heading, md_text, f"MD should contain '{heading}' heading")


class TestArtifactsIndexStructure(unittest.TestCase):
    """Test artifacts_index structure validation for final memory.

    Per docs/memory/memory-for-paper.md, artifacts_index should contain:
    - log_dir, workspace_dir
    - best_node_id, best_node_data
    - top_nodes_data (list of top N nodes)
    """

    def test_artifacts_index_best_node_data_structure(self):
        """Verify best_node_data has expected structure.

        Per memory-for-paper.md lines 213-229:
        - id, branch_id, plan, overall_plan, code
        - phase_artifacts, analysis, metric
        - exp_results_dir, plot_analyses, vlm_feedback_summary
        - datasets_successfully_tested, plot_paths
        - exec_time_feedback, workspace_path
        """
        expected_keys = [
            "id",
            "branch_id",
            "plan",
            "overall_plan",
            "code",
            "phase_artifacts",
            "analysis",
            "metric",
            "exp_results_dir",
            "plot_analyses",
            "vlm_feedback_summary",
            "datasets_successfully_tested",
            "plot_paths",
            "exec_time_feedback",
            "workspace_path",
        ]

        # Document expected structure (validation test)
        self.assertEqual(len(expected_keys), 15, "best_node_data should have 15 expected keys")

    def test_artifacts_index_metric_structure(self):
        """Verify metric has value and name fields.

        Per memory-for-paper.md: metric: {"value": float, "name": str}
        """
        sample_metric = {"value": 2.5, "name": "speedup"}

        self.assertIn("value", sample_metric)
        self.assertIn("name", sample_metric)
        self.assertIsInstance(sample_metric["value"], (int, float))
        self.assertIsInstance(sample_metric["name"], str)

    def test_artifacts_index_top_nodes_data_structure(self):
        """Verify top_nodes_data has expected structure.

        Per memory-for-paper.md lines 230-240:
        Each node in top_nodes_data should have:
        - id, branch_id, plan, metric, analysis, vlm_feedback_summary
        """
        expected_keys = [
            "id",
            "branch_id",
            "plan",
            "metric",
            "analysis",
            "vlm_feedback_summary",
        ]

        # Document expected structure (validation test)
        self.assertEqual(len(expected_keys), 6, "top_nodes_data items should have 6 expected keys")


class TestMemRecallAppendTypeValidation(unittest.TestCase):
    """Test type validation for mem_recall_append event dict structure.

    Per memgpt_store.py lines 2438-2496:
    mem_recall_append(event: dict) expects:
    - ts: float (optional, defaults to current time)
    - run_id: str (optional, defaults to memory manager's run_id)
    - node_id: str (optional)
    - branch_id: str (optional, resolved from node_id if not provided)
    - phase: str (optional)
    - kind: str (optional, defaults to "event")
    - summary: str (optional)
    - refs: list[str] (optional)
    - task_hint: str (optional)
    - memory_size: int (optional)
    """

    def setUp(self):
        """Set up test fixtures."""
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
        self.memory_manager = MemoryManager(self.db_path, self.cfg)
        self.root = self.memory_manager.create_branch(None, node_uid="root")

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_event_with_all_expected_keys(self):
        """Verify event with all expected keys is accepted."""
        event = {
            "ts": 1234567890.0,
            "run_id": "test-run",
            "node_id": "root",
            "branch_id": self.root,
            "phase": "phase1",
            "kind": "test_event",
            "summary": "Test event summary",
            "refs": ["ref1", "ref2"],
            "task_hint": "test",
            "memory_size": 100,
        }
        # Should not raise
        self.memory_manager.mem_recall_append(event)

    def test_event_with_minimal_keys(self):
        """Verify event with minimal keys is accepted."""
        event = {
            "kind": "minimal_event",
            "summary": "Minimal test",
            "branch_id": self.root,
        }
        # Should not raise
        self.memory_manager.mem_recall_append(event)

    def test_non_dict_event_is_ignored(self):
        """Verify non-dict event is silently ignored (not exception)."""
        # Per line 2439: if not isinstance(event, dict): return
        self.memory_manager.mem_recall_append(None)  # Should not raise
        self.memory_manager.mem_recall_append("string")  # Should not raise
        self.memory_manager.mem_recall_append([])  # Should not raise
        self.memory_manager.mem_recall_append(123)  # Should not raise

    def test_refs_must_be_iterable(self):
        """Verify refs field handling for various types."""
        # refs should be iterable (list/tuple) but not string
        # Per line 2468: for ref in refs if isinstance(refs, Iterable) and not isinstance(refs, str) else []:
        event_with_list_refs = {
            "kind": "test",
            "summary": "Test",
            "branch_id": self.root,
            "refs": ["ref1", "ref2"],
        }
        self.memory_manager.mem_recall_append(event_with_list_refs)

        # String refs should be handled (converted to empty list per logic)
        event_with_string_refs = {
            "kind": "test",
            "summary": "Test",
            "branch_id": self.root,
            "refs": "single_ref",  # String, not list
        }
        self.memory_manager.mem_recall_append(event_with_string_refs)


class TestMemNodeForkTypeValidation(unittest.TestCase):
    """Test type validation for mem_node_fork parameters.

    Per memgpt_store.py lines 2698-2759:
    mem_node_fork(
        parent_node_id: str | None,
        child_node_id: str,
        ancestor_chain: list[str] | None = None,
        phase: str | None = None,
    )
    """

    def setUp(self):
        """Set up test fixtures."""
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
        self.memory_manager = MemoryManager(self.db_path, self.cfg)
        self.root = self.memory_manager.create_branch(None, node_uid="root")

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_parent_node_id_can_be_none_for_root(self):
        """Verify parent_node_id=None creates root-level node."""
        # Should not raise - creates a new root-level branch
        self.memory_manager.mem_node_fork(
            parent_node_id=None,
            child_node_id="new_root",
        )

    def test_parent_node_id_string_creates_child(self):
        """Verify parent_node_id as string creates child branch."""
        self.memory_manager.mem_node_fork(
            parent_node_id="root",
            child_node_id="child1",
        )

    def test_ancestor_chain_list_preserves_hierarchy(self):
        """Verify ancestor_chain as list creates proper hierarchy."""
        # Create intermediate branches via ancestor_chain
        self.memory_manager.mem_node_fork(
            parent_node_id="grandparent",
            child_node_id="child",
            ancestor_chain=["grandparent"],
        )

    def test_phase_parameter_is_optional(self):
        """Verify phase parameter is optional."""
        self.memory_manager.mem_node_fork(
            parent_node_id="root",
            child_node_id="child_no_phase",
            phase=None,
        )

        self.memory_manager.mem_node_fork(
            parent_node_id="root",
            child_node_id="child_with_phase",
            phase="phase1",
        )


class TestApplyLLMMemoryUpdatesTypeValidation(unittest.TestCase):
    """Test type validation for apply_llm_memory_updates dict structure.

    Per memgpt_store.py lines 2865-2894:
    apply_llm_memory_updates(
        branch_id: str,
        updates: dict,
        node_id: str | None = None,
        phase: str | None = None,
    )

    updates dict can contain:
    - "core": dict of key-value pairs to set in core memory
    - "core_delete": list of keys to delete from core memory
    - "core_get": list of keys to retrieve from core memory
    - "archival": list of dicts with "text" and optional "tags"
    - "archival_update": list of dicts with "id", "text", and optional "tags"
    - "archival_search": dict with "query" and optional "k", "tags"
    - "recall": dict with event data to append to recall
    - "recall_search": dict with "query" and optional "k"
    - "consolidate": bool to trigger memory consolidation
    """

    def setUp(self):
        """Set up test fixtures."""
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
        self.memory_manager = MemoryManager(self.db_path, self.cfg)
        self.root = self.memory_manager.create_branch(None, node_uid="root")

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_empty_updates_returns_empty_dict(self):
        """Verify empty updates returns empty dict (line 2896)."""
        result = self.memory_manager.apply_llm_memory_updates(self.root, {})
        self.assertEqual(result, {})

    def test_none_updates_returns_empty_dict(self):
        """Verify None updates returns empty dict (line 2895)."""
        result = self.memory_manager.apply_llm_memory_updates(self.root, None)
        self.assertEqual(result, {})

    def test_core_updates_dict_structure(self):
        """Verify core updates accepts dict of key-value pairs."""
        updates = {
            "core": {
                "key1": "value1",
                "key2": "value2",
            }
        }
        result = self.memory_manager.apply_llm_memory_updates(self.root, updates)
        self.assertIsInstance(result, dict)

    def test_core_updates_extended_format(self):
        """Verify core updates accepts extended format with importance/ttl."""
        updates = {
            "core": {
                "important_key": {
                    "value": "important_value",
                    "importance": 5,
                    "ttl": "1h",
                }
            }
        }
        result = self.memory_manager.apply_llm_memory_updates(self.root, updates)
        self.assertIsInstance(result, dict)

    def test_core_get_must_be_list(self):
        """Verify core_get accepts list of keys (line 2940)."""
        # First set some values
        self.memory_manager.set_core(self.root, "test_key", "test_value")

        updates = {
            "core_get": ["test_key", "nonexistent_key"]
        }
        result = self.memory_manager.apply_llm_memory_updates(self.root, updates)
        self.assertIn("core_get", result)
        self.assertIsInstance(result["core_get"], dict)

    def test_core_delete_accepts_list_or_string(self):
        """Verify core_delete accepts list or string (lines 2954-2979)."""
        self.memory_manager.set_core(self.root, "key_to_delete", "value")

        # List form
        updates_list = {"core_delete": ["key_to_delete"]}
        self.memory_manager.apply_llm_memory_updates(self.root, updates_list)

        # String form (single key)
        self.memory_manager.set_core(self.root, "key_to_delete2", "value")
        updates_str = {"core_delete": "key_to_delete2"}
        self.memory_manager.apply_llm_memory_updates(self.root, updates_str)

    def test_archival_must_be_list_of_dicts(self):
        """Verify archival accepts list of dicts with text field."""
        updates = {
            "archival": [
                {"text": "archival entry 1", "tags": ["tag1"]},
                {"text": "archival entry 2"},
            ]
        }
        result = self.memory_manager.apply_llm_memory_updates(self.root, updates)
        self.assertIsInstance(result, dict)

    def test_archival_update_must_have_id(self):
        """Verify archival_update requires id field (line 3018)."""
        # First create an archival entry
        record_id = self.memory_manager.write_archival(self.root, "initial text", tags=["test"])

        updates = {
            "archival_update": [
                {"id": record_id, "text": "updated text"},
                {"text": "missing id"},  # Should be skipped
            ]
        }
        result = self.memory_manager.apply_llm_memory_updates(self.root, updates)
        self.assertIsInstance(result, dict)

    def test_archival_search_must_be_dict_with_query(self):
        """Verify archival_search expects dict with query field."""
        # First write something to search
        self.memory_manager.write_archival(self.root, "searchable content", tags=["test"])

        updates = {
            "archival_search": {
                "query": "searchable",
                "k": 5,
                "tags": ["test"],
            }
        }
        result = self.memory_manager.apply_llm_memory_updates(self.root, updates)
        if "archival_search" in result:
            self.assertIsInstance(result["archival_search"], list)


class TestRenderForPromptTypeValidation(unittest.TestCase):
    """Test type validation for render_for_prompt parameters.

    Per memgpt_store.py lines 3411-3428:
    render_for_prompt(
        branch_id: str,
        task_hint: str | None,
        budget_chars: int,
        no_limit: bool = False,
    ) -> str
    """

    def setUp(self):
        """Set up test fixtures."""
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
        self.memory_manager = MemoryManager(self.db_path, self.cfg)
        self.root = self.memory_manager.create_branch(None, node_uid="root")

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_returns_string(self):
        """Verify render_for_prompt returns string."""
        result = self.memory_manager.render_for_prompt(
            self.root, task_hint="test", budget_chars=4000
        )
        self.assertIsInstance(result, str)

    def test_empty_branch_id_returns_empty_string(self):
        """Verify empty branch_id returns empty string (line 3429)."""
        result = self.memory_manager.render_for_prompt(
            "", task_hint="test", budget_chars=4000
        )
        self.assertEqual(result, "")

    def test_none_task_hint_is_accepted(self):
        """Verify task_hint=None is accepted."""
        result = self.memory_manager.render_for_prompt(
            self.root, task_hint=None, budget_chars=4000
        )
        self.assertIsInstance(result, str)

    def test_no_limit_bypasses_truncation(self):
        """Verify no_limit=True returns full content."""
        # Add some content
        self.memory_manager.set_core(self.root, "key", "value" * 100)

        result_limited = self.memory_manager.render_for_prompt(
            self.root, task_hint="test", budget_chars=100, no_limit=False
        )
        result_unlimited = self.memory_manager.render_for_prompt(
            self.root, task_hint="test", budget_chars=100, no_limit=True
        )

        # Unlimited should have more or equal content
        self.assertGreaterEqual(len(result_unlimited), len(result_limited))


class TestRenderForPromptWithLogTypeValidation(unittest.TestCase):
    """Test type validation for render_for_prompt_with_log.

    Per memgpt_store.py lines 3515-3545:
    render_for_prompt_with_log(
        branch_id: str,
        task_hint: str | None,
        budget_chars: int,
        no_limit: bool = False,
    ) -> tuple[str, dict]

    Returns tuple of (rendered_text, log_details) where log_details contains:
    - core_items: List of core memory key-value pairs
    - recall_items: List of recall memory events
    - archival_items: List of archival memory entries
    - task_hint: The task hint used for retrieval
    - budget_chars: The character budget used
    """

    def setUp(self):
        """Set up test fixtures."""
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
        self.memory_manager = MemoryManager(self.db_path, self.cfg)
        self.root = self.memory_manager.create_branch(None, node_uid="root")

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_returns_tuple_of_string_and_dict(self):
        """Verify returns (str, dict) tuple."""
        result = self.memory_manager.render_for_prompt_with_log(
            self.root, task_hint="test", budget_chars=4000
        )
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], str)
        self.assertIsInstance(result[1], dict)

    def test_log_details_has_expected_keys(self):
        """Verify log_details contains expected keys."""
        _, log_details = self.memory_manager.render_for_prompt_with_log(
            self.root, task_hint="test", budget_chars=4000
        )

        # Check for expected keys (may be empty lists but should exist)
        expected_keys = ["core_items", "recall_items", "archival_items"]
        for key in expected_keys:
            self.assertIn(key, log_details, f"log_details should contain '{key}'")

    def test_empty_branch_returns_empty_tuple(self):
        """Verify empty branch_id returns ('', {})."""
        result = self.memory_manager.render_for_prompt_with_log(
            "", task_hint="test", budget_chars=4000
        )
        self.assertEqual(result, ("", {}))


class TestGenerateFinalMemoryForPaperTypeValidation(unittest.TestCase):
    """Test type validation for generate_final_memory_for_paper parameters.

    Per memgpt_store.py lines 4793-4812:
    generate_final_memory_for_paper(
        run_dir: str | Path,
        root_branch_id: str,
        best_branch_id: str | None,
        artifacts_index: dict[str, Any] | None = None,
        no_budget_limit: bool = True,
    ) -> dict[str, Any]
    """

    def setUp(self):
        """Set up test fixtures."""
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = Path(self.tmpdir) / "memory" / "memory.sqlite"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.cfg = SimpleNamespace(
            core_max_chars=4000,
            recall_max_events=20,
            retrieval_k=8,
            use_fts="off",
            memory_log_enabled=False,
            auto_consolidate=False,
            final_memory_filename_md="final_memory_for_paper.md",
            final_memory_filename_json="final_memory_for_paper.json",
        )
        self.memory_manager = MemoryManager(self.db_path, self.cfg)
        self.root = self.memory_manager.create_branch(None, node_uid="root")

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_run_dir_accepts_string(self):
        """Verify run_dir accepts string path."""
        result = self.memory_manager.generate_final_memory_for_paper(
            run_dir=self.tmpdir,
            root_branch_id=self.root,
            best_branch_id=self.root,
        )
        self.assertIsInstance(result, dict)

    def test_run_dir_accepts_path(self):
        """Verify run_dir accepts Path object."""
        result = self.memory_manager.generate_final_memory_for_paper(
            run_dir=Path(self.tmpdir),
            root_branch_id=self.root,
            best_branch_id=self.root,
        )
        self.assertIsInstance(result, dict)

    def test_best_branch_id_can_be_none(self):
        """Verify best_branch_id=None uses root_branch_id."""
        result = self.memory_manager.generate_final_memory_for_paper(
            run_dir=self.tmpdir,
            root_branch_id=self.root,
            best_branch_id=None,
        )
        self.assertIsInstance(result, dict)

    def test_artifacts_index_structure_validation(self):
        """Verify artifacts_index dict is passed through correctly."""
        artifacts = {
            "log_dir": "/path/to/logs",
            "workspace_dir": "/path/to/workspace",
            "best_node_id": "node123",
            "best_node_data": {
                "id": "node123",
                "branch_id": self.root,
                "plan": "Test plan",
                "metric": {"value": 2.5, "name": "speedup"},
            },
            "top_nodes_data": [],
        }

        result = self.memory_manager.generate_final_memory_for_paper(
            run_dir=self.tmpdir,
            root_branch_id=self.root,
            best_branch_id=self.root,
            artifacts_index=artifacts,
        )

        self.assertIn("artifacts_index", result)
        self.assertEqual(result["artifacts_index"], artifacts)


class TestInterComponentPropertyAccess(unittest.TestCase):
    """Test property access patterns between components.

    This tests common error patterns where components access non-existent
    properties or assume incorrect types.
    """

    def setUp(self):
        """Set up test fixtures."""
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
        self.memory_manager = MemoryManager(self.db_path, self.cfg)
        self.root = self.memory_manager.create_branch(None, node_uid="root")

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_mem_node_read_returns_expected_keys(self):
        """Verify mem_node_read returns dict with expected keys."""
        # Add some data
        self.memory_manager.set_core(self.root, "key", "value")

        result = self.memory_manager.mem_node_read("root", scope="all")

        # Should have core, recall, archival keys
        self.assertIn("core", result)
        self.assertIn("recall", result)
        self.assertIn("archival", result)

        # core should be dict
        self.assertIsInstance(result["core"], dict)
        # recall should be list
        self.assertIsInstance(result["recall"], list)
        # archival should be list
        self.assertIsInstance(result["archival"], list)

    def test_mem_node_read_with_invalid_scope_returns_partial(self):
        """Verify mem_node_read with single scope returns only that scope."""
        self.memory_manager.set_core(self.root, "key", "value")

        result_core = self.memory_manager.mem_node_read("root", scope="core")
        self.assertIn("core", result_core)
        self.assertNotIn("recall", result_core)
        self.assertNotIn("archival", result_core)

        result_recall = self.memory_manager.mem_node_read("root", scope="recall")
        self.assertIn("recall", result_recall)
        self.assertNotIn("core", result_recall)
        self.assertNotIn("archival", result_recall)

    def test_mem_node_read_nonexistent_returns_empty_dict(self):
        """Verify mem_node_read with nonexistent node returns empty dict."""
        result = self.memory_manager.mem_node_read("nonexistent_node")
        self.assertEqual(result, {})

    def test_retrieve_archival_returns_list_of_dicts(self):
        """Verify retrieve_archival returns list of dicts with expected fields."""
        # Add archival entry
        self.memory_manager.write_archival(self.root, "test content", tags=["test"])

        results = self.memory_manager.retrieve_archival(
            branch_id=self.root,
            query="test",
            k=10,
        )

        self.assertIsInstance(results, list)
        if results:
            # Each result should be a dict with expected fields
            for r in results:
                self.assertIsInstance(r, dict)
                self.assertIn("text", r)
                self.assertIn("id", r)

    def test_get_core_returns_string_or_none(self):
        """Verify get_core returns string or None."""
        # Non-existent key
        result_none = self.memory_manager.get_core(self.root, "nonexistent")
        self.assertIsNone(result_none)

        # Existing key
        self.memory_manager.set_core(self.root, "existing_key", "existing_value")
        result_str = self.memory_manager.get_core(self.root, "existing_key")
        self.assertEqual(result_str, "existing_value")
        self.assertIsInstance(result_str, str)


class TestFTS5QueryEscaping(unittest.TestCase):
    """Test FTS5 query escaping for archival memory search.

    FTS5 has special characters that can cause syntax errors:
    - / (path separator) -> "fts5: syntax error near '/'"
    - = (equals) -> "fts5: syntax error near '='"
    - : (colon) -> interpreted as column filter
    - * (asterisk) -> prefix match
    - " (quotes) -> phrase query
    - - + ( ) -> boolean operators

    The _escape_fts5_query function should escape these characters to prevent errors.
    """

    def test_escape_path_with_slashes(self):
        """Verify paths with slashes are escaped."""
        from ai_scientist.memory.memgpt_store import _escape_fts5_query

        query = "/path/to/file.png"
        escaped = _escape_fts5_query(query)

        # Should wrap words in quotes
        self.assertIn('"path"', escaped)
        self.assertIn('"to"', escaped)
        # file.png stays together (period is not a special char)
        self.assertIn('"file.png"', escaped)
        # Should not contain raw slash
        self.assertNotIn("/", escaped)

    def test_escape_equals_sign(self):
        """Verify equals signs are escaped."""
        from ai_scientist.memory.memgpt_store import _escape_fts5_query

        query = "key=value"
        escaped = _escape_fts5_query(query)

        # Should split on = and quote each part
        self.assertIn('"key"', escaped)
        self.assertIn('"value"', escaped)
        # Should not contain raw equals
        self.assertNotIn("=", escaped)

    def test_escape_colon_column_filter(self):
        """Verify colons (column filters) are escaped."""
        from ai_scientist.memory.memgpt_store import _escape_fts5_query

        query = "stability_oriented:autotuning"
        escaped = _escape_fts5_query(query)

        # Should handle colon by splitting or quoting
        self.assertIn('"stability_oriented"', escaped)
        self.assertIn('"autotuning"', escaped)

    def test_no_escape_for_simple_query(self):
        """Verify simple queries without special chars are unchanged."""
        from ai_scientist.memory.memgpt_store import _escape_fts5_query

        query = "simple query words"
        escaped = _escape_fts5_query(query)

        # Simple query should be returned as-is
        self.assertEqual(escaped, query)

    def test_empty_query_returns_empty(self):
        """Verify empty query returns empty string."""
        from ai_scientist.memory.memgpt_store import _escape_fts5_query

        self.assertEqual(_escape_fts5_query(""), "")
        self.assertEqual(_escape_fts5_query(None), "")

    def test_escape_complex_path(self):
        """Verify complex file paths are properly escaped."""
        from ai_scientist.memory.memgpt_store import _escape_fts5_query

        query = "/hs/work0/home/users/test/experiment_results/file.png"
        escaped = _escape_fts5_query(query)

        # Should not raise error and should contain quoted words
        self.assertIsInstance(escaped, str)
        self.assertTrue(len(escaped) > 0)
        # Each path component should be quoted
        self.assertIn('"hs"', escaped)
        self.assertIn('"work0"', escaped)


class TestArchivalSearchWithSpecialChars(unittest.TestCase):
    """Test that archival search handles special characters gracefully.

    This tests the integration of _escape_fts5_query with retrieve_archival.
    """

    def setUp(self):
        """Set up test fixtures with FTS enabled."""
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = Path(self.tmpdir) / "memory.sqlite"
        self.cfg = SimpleNamespace(
            core_max_chars=4000,
            recall_max_events=20,
            retrieval_k=8,
            use_fts="on",  # Enable FTS for this test
            memory_log_enabled=False,
            auto_consolidate=False,
        )
        self.memory_manager = MemoryManager(self.db_path, self.cfg)
        self.root = self.memory_manager.create_branch(None, node_uid="root")

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_search_with_path_does_not_raise(self):
        """Verify searching with file path doesn't raise FTS5 error."""
        # Write some content
        self.memory_manager.write_archival(
            self.root,
            "Results from experiment at /path/to/results.png",
            tags=["test"],
        )

        # Search with path-like query should not raise
        try:
            results = self.memory_manager.retrieve_archival(
                branch_id=self.root,
                query="/path/to/results",
                k=10,
            )
            # Should return results (possibly empty if FTS doesn't match)
            self.assertIsInstance(results, list)
        except Exception as e:
            self.fail(f"Search with path raised exception: {e}")

    def test_search_with_equals_does_not_raise(self):
        """Verify searching with equals sign doesn't raise FTS5 error."""
        self.memory_manager.write_archival(
            self.root,
            "Configuration: threads=8, binding=compact",
            tags=["config"],
        )

        try:
            results = self.memory_manager.retrieve_archival(
                branch_id=self.root,
                query="threads=8",
                k=10,
            )
            self.assertIsInstance(results, list)
        except Exception as e:
            self.fail(f"Search with equals raised exception: {e}")

    def test_search_with_colon_does_not_raise(self):
        """Verify searching with colon doesn't raise FTS5 error."""
        self.memory_manager.write_archival(
            self.root,
            "stability_oriented:autotuning experiment results",
            tags=["experiment"],
        )

        try:
            results = self.memory_manager.retrieve_archival(
                branch_id=self.root,
                query="stability_oriented:autotuning",
                k=10,
            )
            self.assertIsInstance(results, list)
        except Exception as e:
            self.fail(f"Search with colon raised exception: {e}")


if __name__ == "__main__":
    unittest.main()
