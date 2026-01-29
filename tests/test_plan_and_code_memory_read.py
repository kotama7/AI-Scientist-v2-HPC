"""Tests for plan_and_code_query memory read re-query handling.

Verifies that both MinimalAgent.plan_and_code_query and
ParallelAgent.plan_and_code_query correctly:

1. Detect memory read results from apply_llm_memory_updates.
2. Inject read results into the prompt and re-query the LLM.
3. Do NOT consume a retry slot for re-queries.
4. Terminate via the hard call ceiling if re-queries keep happening.
"""

import unittest
from unittest.mock import MagicMock, patch, PropertyMock


def _make_memory_update_response(with_code: bool = False) -> str:
    """Build a fake LLM response with a <memory_update> block."""
    mem_block = (
        "<memory_update>\n"
        '{"mem_archival_search": {"query": "performance data", "k": 5}}\n'
        "</memory_update>\n"
    )
    if with_code:
        return (
            mem_block
            + "Here is the plan.\n\n"
            "```python\nprint('hello')\n```\n"
        )
    return mem_block


def _make_code_response() -> str:
    """LLM response with plan + code (no memory)."""
    return "Here is the plan.\n\n```python\nprint('hello')\n```\n"


def _make_agent_stub():
    """Minimal mock satisfying plan_and_code_query requirements."""
    agent = MagicMock()
    agent.code_language = "python"
    agent.cfg.agent.code.model = "test-model"
    agent.cfg.agent.code.temp = 0.0
    agent.cfg.memory.max_memory_read_rounds = 2
    agent.branch_id = "branch-1"
    agent._current_node_id = "node-1"
    agent.prompt_log_dir = None
    agent._last_memory_operation_results = None
    agent.memory_manager = MagicMock()
    return agent


# ---------------------------------------------------------------------------
# MinimalAgent.plan_and_code_query
# ---------------------------------------------------------------------------

class TestMinimalAgentPlanAndCodeMemoryRead(unittest.TestCase):

    @patch("ai_scientist.treesearch.parallel_agent.query")
    def test_read_results_injected_and_requeried(self, mock_query):
        """When memory read results are returned, the LLM should be re-queried
        with results injected into the prompt, without consuming a retry."""
        from ai_scientist.treesearch.parallel_agent import MinimalAgent

        mock_query.side_effect = [
            _make_memory_update_response(with_code=False),  # read-only, triggers re-query
            _make_code_response(),                           # success
        ]

        agent = _make_agent_stub()
        agent.plan_and_code_query = MinimalAgent.plan_and_code_query.__get__(agent)
        agent._log_prompt = MagicMock()
        agent._log_memory_operations_detail = MagicMock()

        agent.memory_manager.apply_llm_memory_updates.side_effect = [
            {"archival_search": [{"text": "info", "tags": ["T"]}]},  # has read results
            {},  # no read results on second call
        ]

        prompt = {"task": "do something"}
        nl, code = agent.plan_and_code_query(
            prompt=prompt,
            retries=1,
            code_language="python",
            log_label="test",
        )

        # Should succeed with code
        self.assertIn("hello", code)
        # query called twice (read-only + success), but only 1 retry consumed
        self.assertEqual(mock_query.call_count, 2)
        # Memory Read Results should have been cleaned up
        self.assertNotIn("Memory Read Results", prompt)

    @patch("ai_scientist.treesearch.parallel_agent.query")
    def test_no_read_results_no_requery(self, mock_query):
        """When memory has no read results, proceed normally without re-query."""
        from ai_scientist.treesearch.parallel_agent import MinimalAgent

        mock_query.side_effect = [
            _make_memory_update_response(with_code=True),  # write + code
        ]

        agent = _make_agent_stub()
        agent.plan_and_code_query = MinimalAgent.plan_and_code_query.__get__(agent)
        agent._log_prompt = MagicMock()
        agent._log_memory_operations_detail = MagicMock()

        # No read results — only writes
        agent.memory_manager.apply_llm_memory_updates.return_value = {}

        nl, code = agent.plan_and_code_query(
            prompt={"task": "do something"},
            retries=3,
            code_language="python",
            log_label="test",
        )

        self.assertIn("hello", code)
        self.assertEqual(mock_query.call_count, 1)

    @patch("ai_scientist.treesearch.parallel_agent.query")
    def test_hard_ceiling_prevents_infinite_loop(self, mock_query):
        """Hard call ceiling must terminate the loop."""
        from ai_scientist.treesearch.parallel_agent import MinimalAgent

        # All responses are memory-read-only with no code
        mock_query.side_effect = [
            _make_memory_update_response(with_code=False) for _ in range(50)
        ]

        agent = _make_agent_stub()
        agent.plan_and_code_query = MinimalAgent.plan_and_code_query.__get__(agent)
        agent._log_prompt = MagicMock()
        agent._log_memory_operations_detail = MagicMock()

        agent.memory_manager.apply_llm_memory_updates.return_value = {
            "archival_search": [{"text": "info", "tags": ["T"]}],
        }

        nl, code = agent.plan_and_code_query(
            prompt={"task": "do something"},
            retries=3,
            code_language="python",
            log_label="test",
        )

        # Should terminate; max calls = retries(3) + max_memory_read_rounds(2) + 1 = 6
        self.assertLessEqual(mock_query.call_count, 6)

    @patch("ai_scientist.treesearch.parallel_agent.query")
    def test_stale_parsing_feedback_cleared(self, mock_query):
        """When memory read re-query happens, stale Parsing Feedback is cleared."""
        from ai_scientist.treesearch.parallel_agent import MinimalAgent

        mock_query.side_effect = [
            _make_memory_update_response(with_code=False),  # read-only
            _make_code_response(),                           # success
        ]

        agent = _make_agent_stub()
        agent.plan_and_code_query = MinimalAgent.plan_and_code_query.__get__(agent)
        agent._log_prompt = MagicMock()
        agent._log_memory_operations_detail = MagicMock()

        agent.memory_manager.apply_llm_memory_updates.side_effect = [
            {"archival_search": [{"text": "info", "tags": ["T"]}]},
            {},
        ]

        prompt = {"task": "do something", "Parsing Feedback": "old stale error"}
        nl, code = agent.plan_and_code_query(
            prompt=prompt,
            retries=3,
            code_language="python",
            log_label="test",
        )

        self.assertIn("hello", code)
        self.assertNotIn("Parsing Feedback", prompt)


# ---------------------------------------------------------------------------
# ParallelAgent.plan_and_code_query
# ---------------------------------------------------------------------------

class TestParallelAgentPlanAndCodeMemoryRead(unittest.TestCase):

    @patch("ai_scientist.treesearch.parallel_agent.query")
    def test_read_results_injected_and_requeried(self, mock_query):
        """ParallelAgent variant: re-query with read results."""
        from ai_scientist.treesearch.parallel_agent import ParallelAgent

        mock_query.side_effect = [
            _make_memory_update_response(with_code=False),
            _make_code_response(),
        ]

        agent = _make_agent_stub()
        agent.plan_and_code_query = ParallelAgent.plan_and_code_query.__get__(agent)
        agent._log_memory_operations_detail = MagicMock()

        agent.memory_manager.apply_llm_memory_updates.side_effect = [
            {"archival_search": [{"text": "info", "tags": ["T"]}]},
            {},
        ]

        prompt = {"task": "do something"}
        nl, code = agent.plan_and_code_query(
            prompt=prompt,
            retries=1,
            code_language="python",
        )

        self.assertIn("hello", code)
        self.assertEqual(mock_query.call_count, 2)
        self.assertNotIn("Memory Read Results", prompt)

    @patch("ai_scientist.treesearch.parallel_agent.query")
    def test_hard_ceiling_prevents_infinite_loop(self, mock_query):
        """ParallelAgent variant: hard ceiling."""
        from ai_scientist.treesearch.parallel_agent import ParallelAgent

        mock_query.side_effect = [
            _make_memory_update_response(with_code=False) for _ in range(50)
        ]

        agent = _make_agent_stub()
        agent.plan_and_code_query = ParallelAgent.plan_and_code_query.__get__(agent)
        agent._log_memory_operations_detail = MagicMock()

        agent.memory_manager.apply_llm_memory_updates.return_value = {
            "archival_search": [{"text": "info", "tags": ["T"]}],
        }

        nl, code = agent.plan_and_code_query(
            prompt={"task": "do something"},
            retries=3,
            code_language="python",
        )

        self.assertLessEqual(mock_query.call_count, 6)


if __name__ == "__main__":
    unittest.main()
