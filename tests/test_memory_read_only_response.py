"""Tests for the memory read-only response handling fix.

This verifies that when the LLM returns a <memory_update> block containing
only read operations (mem_archival_search, mem_core_get, etc.) with no
phase_artifacts JSON body, the system:

1.  Raises MemoryReadOnlyResponse (not PhasePlanError) from
    extract_phase_artifacts.
2.  The helper _is_memory_read_only_request correctly classifies read-only
    vs. write-containing vs. mixed updates.
3.  In generate_phase_artifacts, a read-only response does NOT consume a
    retry slot — the retry counter stays the same after handling it.
"""

import json
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

from ai_scientist.treesearch.utils.phase_plan import (
    PhasePlanError,
    MemoryReadOnlyResponse,
    MissingMemoryUpdateError,
    _is_memory_read_only_request,
    extract_phase_artifacts,
)


# ---------------------------------------------------------------------------
# _is_memory_read_only_request
# ---------------------------------------------------------------------------

class TestIsMemoryReadOnlyRequest(unittest.TestCase):
    """Unit tests for the _is_memory_read_only_request helper."""

    def test_none(self):
        self.assertFalse(_is_memory_read_only_request(None))

    def test_empty_dict(self):
        self.assertFalse(_is_memory_read_only_request({}))

    def test_read_only_archival_search(self):
        self.assertTrue(
            _is_memory_read_only_request({"mem_archival_search": {"query": "foo", "k": 5}})
        )

    def test_read_only_core_get(self):
        self.assertTrue(
            _is_memory_read_only_request({"mem_core_get": ["key1"]})
        )

    def test_read_only_recall_search(self):
        self.assertTrue(
            _is_memory_read_only_request({"mem_recall_search": {"query": "bar"}})
        )

    def test_read_only_multiple_reads(self):
        self.assertTrue(
            _is_memory_read_only_request({
                "mem_archival_search": {"query": "x", "k": 3},
                "mem_core_get": ["a"],
            })
        )

    def test_write_only(self):
        self.assertFalse(
            _is_memory_read_only_request({"mem_core_set": {"key": "val"}})
        )

    def test_mixed_read_and_write(self):
        """A request that has both reads and writes is NOT read-only."""
        self.assertFalse(
            _is_memory_read_only_request({
                "mem_archival_search": {"query": "x", "k": 3},
                "mem_core_set": {"key": "val"},
            })
        )

    def test_archival_write_only(self):
        self.assertFalse(
            _is_memory_read_only_request({
                "mem_archival_write": [{"text": "info", "tags": ["T"]}],
            })
        )


# ---------------------------------------------------------------------------
# extract_phase_artifacts — MemoryReadOnlyResponse detection
# ---------------------------------------------------------------------------

class TestExtractPhaseArtifactsReadOnly(unittest.TestCase):
    """Verify extract_phase_artifacts raises MemoryReadOnlyResponse for
    read-only memory requests with no JSON body."""

    def _make_read_only_response(self, updates_dict: dict) -> str:
        """Build a raw LLM response that has only <memory_update> + no JSON."""
        return (
            "<memory_update>\n"
            + json.dumps(updates_dict, indent=2)
            + "\n</memory_update>\n"
        )

    def test_read_only_archival_search_raises(self):
        raw = self._make_read_only_response(
            {"mem_archival_search": {"query": "timeout sweep", "k": 5}}
        )
        with self.assertRaises(MemoryReadOnlyResponse) as ctx:
            extract_phase_artifacts(raw, default_language="python", require_memory_update=True)
        self.assertIn("mem_archival_search", ctx.exception.memory_updates)

    def test_read_only_core_get_raises(self):
        raw = self._make_read_only_response(
            {"mem_core_get": ["algorithm_approach"]}
        )
        with self.assertRaises(MemoryReadOnlyResponse) as ctx:
            extract_phase_artifacts(raw, default_language="python", require_memory_update=True)
        self.assertIn("mem_core_get", ctx.exception.memory_updates)

    def test_read_plus_json_body_does_not_raise(self):
        """When read ops AND valid JSON body are present, should parse normally."""
        body = {
            "phase_artifacts": {
                "download": {"commands": [], "notes": ""},
                "coding": {"workspace": {"root": "/workspace", "tree": [], "files": [
                    {"path": "src/main.py", "mode": "0644", "encoding": "utf-8",
                     "content": "print('hello')"}
                ]}},
                "compile": {"build_plan": {"language": "python"}},
                "run": {"commands": ["python3 src/main.py"], "expected_outputs": ["working/out.npy"]},
            }
        }
        raw = (
            "<memory_update>\n"
            '{"mem_archival_search": {"query": "test", "k": 3}}\n'
            "</memory_update>\n"
            + json.dumps(body)
        )
        result = extract_phase_artifacts(raw, default_language="python", require_memory_update=True)
        self.assertIn("phase_artifacts", result)

    def test_write_only_with_no_json_raises_phase_plan_error(self):
        """Write-only memory update with no JSON should be a normal PhasePlanError,
        NOT MemoryReadOnlyResponse."""
        raw = (
            "<memory_update>\n"
            '{"mem_core_set": {"key": "val"}}\n'
            "</memory_update>\n"
        )
        with self.assertRaises(PhasePlanError) as ctx:
            extract_phase_artifacts(raw, default_language="python", require_memory_update=True)
        self.assertNotIsInstance(ctx.exception, MemoryReadOnlyResponse)

    def test_empty_memory_update_with_no_json_raises_phase_plan_error(self):
        """Empty {} memory block with no JSON body => PhasePlanError."""
        raw = "<memory_update>{}</memory_update>\n"
        with self.assertRaises(PhasePlanError):
            extract_phase_artifacts(raw, default_language="python", require_memory_update=True)

    def test_no_memory_update_at_all_raises_missing(self):
        """No <memory_update> block when required => MissingMemoryUpdateError."""
        raw = '{"phase_artifacts": {}}'
        with self.assertRaises(MissingMemoryUpdateError):
            extract_phase_artifacts(raw, default_language="python", require_memory_update=True)


# ---------------------------------------------------------------------------
# generate_phase_artifacts — retry budget preservation
# ---------------------------------------------------------------------------

class TestGeneratePhaseArtifactsRetryBudget(unittest.TestCase):
    """Integration-level tests verifying that MemoryReadOnlyResponse does NOT
    consume a retry slot in generate_phase_artifacts."""

    def _make_agent_stub(self):
        """Build a minimal mock that satisfies generate_phase_artifacts."""
        agent = MagicMock()
        agent.code_language = "python"
        agent.experiment_name = "test_exp"
        agent.experiment_output_path = "working/test_exp_data.npy"
        agent.cfg.agent.code.model = "test-model"
        agent.cfg.agent.code.temp = 0.0
        agent.branch_id = "branch-1"
        agent._current_node_id = "node-1"
        agent.prompt_log_dir = None
        type(agent)._is_memory_enabled = PropertyMock(return_value=True)

        # Memory manager mock
        agent.memory_manager = MagicMock()
        agent.memory_manager.apply_llm_memory_updates.return_value = {
            "archival_search": [
                {"text": "Found info about timeout bug", "tags": ["BUG"]},
            ]
        }

        return agent

    def _valid_phase_response(self, with_memory: bool = True) -> str:
        body = {
            "phase_artifacts": {
                "download": {"commands": [], "notes": ""},
                "coding": {"workspace": {"root": "/workspace", "tree": [], "files": [
                    {"path": "src/main.py", "mode": "0644", "encoding": "utf-8",
                     "content": "print('ok')"}
                ]}},
                "compile": {"build_plan": {"language": "python"}},
                "run": {"commands": ["python3 src/main.py"],
                        "expected_outputs": ["working/test_exp_data.npy"]},
            }
        }
        if with_memory:
            return (
                "<memory_update>\n"
                '{"mem_core_set": {"done": true}}\n'
                "</memory_update>\n"
                + json.dumps(body)
            )
        return json.dumps(body)

    def _read_only_response(self) -> str:
        return (
            "<memory_update>\n"
            '{"mem_archival_search": {"query": "timeout sweep", "k": 5}}\n'
            "</memory_update>\n"
        )

    @patch("ai_scientist.treesearch.parallel_agent.query")
    def test_read_only_then_success_uses_one_retry(self, mock_query):
        """Scenario: LLM returns read-only response first, then valid response.
        Only 1 retry slot should be consumed (the successful parse), not 2."""
        from ai_scientist.treesearch.parallel_agent import MinimalAgent

        mock_query.side_effect = [
            self._read_only_response(),     # 1st call: read-only (should NOT consume retry)
            self._valid_phase_response(),    # 2nd call: valid (consumes 1 retry)
        ]

        agent = self._make_agent_stub()
        # Bind the real method
        agent.generate_phase_artifacts = MinimalAgent.generate_phase_artifacts.__get__(agent)
        agent._validate_phase_language = MagicMock()
        agent._log_prompt = MagicMock()
        agent._log_response = MagicMock()
        agent._log_memory_operations_detail = MagicMock()
        agent._last_memory_operation_results = None
        agent.last_phase_artifacts_response = ""

        # Memory read results that indicate read data was found
        agent.memory_manager.apply_llm_memory_updates.side_effect = [
            # First call (read-only): return archival_search results
            {"archival_search": [{"text": "info", "tags": ["T"]}]},
            # Second call (write from valid response): no read results
            {},
        ]

        # With retries=1, if the read-only consumed a retry this would fallback
        result = agent.generate_phase_artifacts(
            prompt={"test": "prompt"},
            retries=1,
            log_label="test",
            max_memory_read_rounds=2,
        )

        # Should succeed, NOT fall through to _fallback_phase_artifacts
        self.assertIn("phase_artifacts", result)
        # query was called exactly 2 times
        self.assertEqual(mock_query.call_count, 2)

    @patch("ai_scientist.treesearch.parallel_agent.query")
    def test_three_read_only_then_fallback_with_retries_1(self, mock_query):
        """When memory rounds are exhausted AND retries=1, the system should
        fall through to fallback on the retry attempt (not hang forever)."""
        from ai_scientist.treesearch.parallel_agent import MinimalAgent

        mock_query.side_effect = [
            self._read_only_response(),   # read-only, serviced
            self._read_only_response(),   # read-only again but max_rounds=1 exceeded
            # retry 1: another read-only, can't service → PhasePlanError feedback
            self._read_only_response(),
        ]

        agent = self._make_agent_stub()
        agent.generate_phase_artifacts = MinimalAgent.generate_phase_artifacts.__get__(agent)
        agent._validate_phase_language = MagicMock()
        agent._log_prompt = MagicMock()
        agent._log_response = MagicMock()
        agent._log_memory_operations_detail = MagicMock()
        agent._last_memory_operation_results = None
        agent.last_phase_artifacts_response = ""
        agent._fallback_phase_artifacts = MagicMock(return_value={"fallback": True})

        agent.memory_manager.apply_llm_memory_updates.return_value = {
            "archival_search": [{"text": "info", "tags": ["T"]}],
        }

        result = agent.generate_phase_artifacts(
            prompt={"test": "prompt"},
            retries=1,
            log_label="test",
            max_memory_read_rounds=1,  # only 1 read round allowed
        )

        # Should have called fallback
        agent._fallback_phase_artifacts.assert_called_once()
        self.assertTrue(result.get("fallback"))

    @patch("ai_scientist.treesearch.parallel_agent.query")
    def test_stale_parsing_feedback_cleared_after_read_only(self, mock_query):
        """After a successful memory read cycle, any stale 'Parsing Feedback'
        from a prior error should be removed from the prompt."""
        from ai_scientist.treesearch.parallel_agent import MinimalAgent

        mock_query.side_effect = [
            self._read_only_response(),     # read-only, serviced
            self._valid_phase_response(),   # valid
        ]

        agent = self._make_agent_stub()
        agent.generate_phase_artifacts = MinimalAgent.generate_phase_artifacts.__get__(agent)
        agent._validate_phase_language = MagicMock()
        agent._log_prompt = MagicMock()
        agent._log_response = MagicMock()
        agent._log_memory_operations_detail = MagicMock()
        agent._last_memory_operation_results = None
        agent.last_phase_artifacts_response = ""

        agent.memory_manager.apply_llm_memory_updates.side_effect = [
            {"archival_search": [{"text": "info", "tags": ["T"]}]},
            {},
        ]

        prompt = {"test": "prompt", "Parsing Feedback": "old stale error"}
        result = agent.generate_phase_artifacts(
            prompt=prompt,
            retries=3,
            log_label="test",
            max_memory_read_rounds=2,
        )

        self.assertIn("phase_artifacts", result)
        # The stale Parsing Feedback should have been removed
        self.assertNotIn("Parsing Feedback", prompt)


    @patch("ai_scientist.treesearch.parallel_agent.query")
    def test_hard_ceiling_prevents_infinite_loop(self, mock_query):
        """Even if every LLM call returns a read-only response that somehow
        bypasses the memory_read_round guard, the hard call ceiling
        (max_total_calls) must terminate the loop."""
        from ai_scientist.treesearch.parallel_agent import MinimalAgent

        # Supply more responses than the ceiling should ever allow
        mock_query.side_effect = [self._read_only_response() for _ in range(50)]

        agent = self._make_agent_stub()
        agent.generate_phase_artifacts = MinimalAgent.generate_phase_artifacts.__get__(agent)
        agent._validate_phase_language = MagicMock()
        agent._log_prompt = MagicMock()
        agent._log_response = MagicMock()
        agent._log_memory_operations_detail = MagicMock()
        agent._last_memory_operation_results = None
        agent.last_phase_artifacts_response = ""
        agent._fallback_phase_artifacts = MagicMock(return_value={"fallback": True})

        # Always return read results to keep memory_read_round incrementing
        agent.memory_manager.apply_llm_memory_updates.return_value = {
            "archival_search": [{"text": "info", "tags": ["T"]}],
        }

        result = agent.generate_phase_artifacts(
            prompt={"test": "prompt"},
            retries=3,
            log_label="test",
            max_memory_read_rounds=2,
        )

        # Must terminate and call fallback
        agent._fallback_phase_artifacts.assert_called_once()
        self.assertTrue(result.get("fallback"))
        # Total LLM calls should be at most retries + max_memory_read_rounds + 1 = 6
        self.assertLessEqual(mock_query.call_count, 3 + 2 + 1)


if __name__ == "__main__":
    unittest.main()
