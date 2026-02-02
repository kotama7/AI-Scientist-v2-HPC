"""Test to verify that best_node_id.txt matches inherited_from_node_id.

This test verifies the fix for the issue where the best node saved in
best_node_id.txt did not match the node actually inherited to the next stage
(recorded in inherited_from_node_id).

Root cause:
- run_io.save_run() called journal.get_best_node() to save best_node_id.txt
- agent_manager._get_best_implementation() also called journal.get_best_node()
  to determine which node to inherit to the next stage
- These two calls could return different nodes (LLM non-determinism, different params)

Fix:
- agent_manager._get_best_implementation() now saves best_node_id.txt directly
  after determining the best node, ensuring consistency
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, create_autospec
import shutil

from ai_scientist.treesearch.journal import Journal, Node
from ai_scientist.treesearch.utils.metric import MetricValue


class TestBestNodeConsistency(unittest.TestCase):
    """Test that _get_best_implementation saves best_node_id.txt correctly."""

    def test_get_best_implementation_saves_best_node_id(self):
        """_get_best_implementation should save best_node_id.txt with the selected node ID."""

        from ai_scientist.treesearch.agent_manager import AgentManager

        # Create a mock config with required attributes
        cfg = Mock()
        cfg.log_dir = Path(tempfile.mkdtemp())

        # Create a test stage directory
        stage_name = "1_initial_implementation_1_preliminary"
        stage_dir = cfg.log_dir / stage_name
        stage_dir.mkdir(parents=True, exist_ok=True)

        # Create test journal with nodes
        journal = Journal()

        # Create nodes with different metrics
        node1 = Node(
            id="node1_id",
            code="code1",
            plan="plan1",
            is_buggy=False,
            metric=MetricValue(value=0.5, maximize=True, name="accuracy"),
        )
        node2 = Node(
            id="node2_id",
            code="code2",
            plan="plan2",
            is_buggy=False,
            metric=MetricValue(value=0.8, maximize=True, name="accuracy"),
        )

        journal.append(node1)
        journal.append(node2)

        # Create a minimal AgentManager mock
        manager = Mock(spec=AgentManager)
        manager.cfg = cfg
        manager.journals = {stage_name: journal}

        # Mock _memory_context method
        def mock_memory_context(j, context_type):
            return ""
        manager._memory_context = mock_memory_context

        # Call the actual _get_best_implementation method
        # We need to bind it to our mock instance
        from ai_scientist.treesearch.agent_manager import AgentManager as RealAgentManager

        # Create a real method bound to our mock
        get_best_impl = RealAgentManager._get_best_implementation.__get__(manager, type(manager))

        # Mock journal.get_best_node to return node2
        with patch.object(journal, 'get_best_node', return_value=node2):
            # Call the method
            best_node = get_best_impl(stage_name)

        # Verify that best_node_id.txt was created
        best_node_id_file = stage_dir / "best_node_id.txt"
        self.assertTrue(
            best_node_id_file.exists(),
            "best_node_id.txt should be created by _get_best_implementation"
        )

        # Verify that the saved ID matches the returned node's original ID
        saved_id = best_node_id_file.read_text().strip()
        self.assertEqual(
            saved_id,
            "node2_id",
            "Saved best_node_id should match the selected node's ID"
        )

        # Verify that the returned node has inherited_from_node_id set
        self.assertIsNotNone(best_node)
        self.assertEqual(
            best_node.inherited_from_node_id,
            "node2_id",
            "inherited_from_node_id should be set to original node ID"
        )

        # Clean up
        shutil.rmtree(cfg.log_dir)


if __name__ == "__main__":
    unittest.main()
