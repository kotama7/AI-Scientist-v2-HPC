"""Tests for memory handling in response.py and phase_plan.py.

These tests verify that:
1. <memory_results> tags incorrectly output by LLM are properly removed
2. <memory_update> tags are properly extracted and removed
3. The JSON parsing works correctly after tag removal
"""
import json
import pytest

from ai_scientist.treesearch.utils.response import (
    remove_memory_update_tags,
    extract_memory_updates,
)
from ai_scientist.treesearch.utils.phase_plan import (
    extract_memory_update_block,
    extract_phase_artifacts,
    PhasePlanError,
)


class TestRemoveMemoryUpdateTags:
    """Tests for remove_memory_update_tags function."""

    def test_removes_normal_memory_update(self):
        """Test removal of normal <memory_update> tags."""
        text = """<memory_update>
{"mem_core_set": {"key": "value"}}
</memory_update>
{"phase_artifacts": {}}"""
        result = remove_memory_update_tags(text)
        assert "<memory_update>" not in result
        assert "</memory_update>" not in result
        assert '{"phase_artifacts": {}}' in result

    def test_removes_memory_results_tags(self):
        """Test removal of <memory_results> tags that LLM incorrectly outputs."""
        text = """<memory_results>
{"mem_archival_search": []}
</memory_results>
{"phase_artifacts": {}}"""
        result = remove_memory_update_tags(text)
        assert "<memory_results>" not in result
        assert "</memory_results>" not in result
        assert '{"phase_artifacts": {}}' in result

    def test_removes_both_memory_update_and_results(self):
        """Test removal of both <memory_update> and <memory_results> tags."""
        text = """<memory_update>
{"mem_archival_search": {"query": "test", "k": 5}}
</memory_update>
<memory_results>
{"mem_archival_search": []}
</memory_results>
<memory_update>
{"mem_core_set": {"key": "value"}}
</memory_update>
{"phase_artifacts": {"download": {}}}"""
        result = remove_memory_update_tags(text)
        assert "<memory_update>" not in result
        assert "</memory_update>" not in result
        assert "<memory_results>" not in result
        assert "</memory_results>" not in result
        # Only the JSON should remain
        assert '{"phase_artifacts":' in result

    def test_handles_empty_text(self):
        """Test handling of empty text."""
        assert remove_memory_update_tags("") == ""
        assert remove_memory_update_tags(None) is None

    def test_handles_text_without_tags(self):
        """Test handling of text without any memory tags."""
        text = '{"phase_artifacts": {"download": {}}}'
        result = remove_memory_update_tags(text)
        assert result == text


class TestExtractMemoryUpdateBlock:
    """Tests for extract_memory_update_block function."""

    def test_extracts_memory_update_and_removes_results(self):
        """Test that memory_update is extracted and memory_results is removed."""
        text = """<memory_update>
{"mem_archival_search": {"query": "test", "k": 5}}
</memory_update>
<memory_results>
{"mem_archival_search": []}
</memory_results>
{"phase_artifacts": {"download": {"commands": []}}}"""

        updates, remaining = extract_memory_update_block(text)

        # Memory update should be extracted
        assert updates is not None
        assert "mem_archival_search" in updates

        # memory_results should be removed from remaining text
        assert "<memory_results>" not in remaining
        assert "</memory_results>" not in remaining

        # JSON should be parseable
        assert remaining.startswith("{")
        parsed = json.loads(remaining)
        assert "phase_artifacts" in parsed

    def test_handles_multiple_memory_updates_with_results(self):
        """Test handling of multiple memory_update blocks with memory_results between them."""
        text = """<memory_update>
{"mem_archival_search": {"query": "test"}}
</memory_update>
<memory_results>
{"mem_archival_search": [{"text": "result1"}]}
</memory_results>
<memory_update>
{"mem_core_set": {"found": "value"}}
</memory_update>
{"phase_artifacts": {}}"""

        updates, remaining = extract_memory_update_block(text)

        # First memory_update should be extracted
        assert updates is not None
        assert "mem_archival_search" in updates

        # memory_results should be removed
        assert "<memory_results>" not in remaining

        # The second memory_update will still be in remaining (only first is extracted)
        # But memory_results should be removed

    def test_handles_no_memory_update(self):
        """Test handling when no memory_update block is present."""
        text = '{"phase_artifacts": {}}'
        updates, remaining = extract_memory_update_block(text)
        assert updates is None
        assert remaining == text


class TestExtractPhaseArtifactsWithMemoryResults:
    """Tests for extract_phase_artifacts with LLM-generated memory_results."""

    def test_parses_json_after_removing_memory_results(self):
        """Test that JSON is correctly parsed after memory_results removal."""
        # Simulate LLM output with memory_results (which shouldn't be there)
        text = """<memory_update>
{"mem_archival_search": {"query": "compiler", "k": 5}}
</memory_update>
<memory_results>
{"mem_archival_search": []}
</memory_results>
{
  "phase_artifacts": {
    "download": {"commands": [], "notes": "test"},
    "coding": {
      "workspace": {
        "root": "/workspace",
        "tree": ["workspace/", "workspace/src/"],
        "files": [
          {"path": "src/main.c", "mode": "0644", "encoding": "utf-8", "content": "int main() { return 0; }"}
        ]
      },
      "notes": "test"
    },
    "compile": {
      "build_plan": {
        "language": "c",
        "compiler_selected": "gcc",
        "cflags": [],
        "ldflags": [],
        "workdir": "/workspace",
        "output": "bin/a.out"
      },
      "commands": ["gcc -o bin/a.out src/main.c"],
      "notes": "test"
    },
    "run": {
      "commands": ["./bin/a.out"],
      "expected_outputs": ["working/test_data.npy"],
      "notes": "test"
    }
  },
  "constraints": {}
}"""

        result = extract_phase_artifacts(text, require_memory_update=True)

        assert "phase_artifacts" in result
        assert "memory_update" in result
        assert result["memory_update"]["mem_archival_search"]["query"] == "compiler"

    def test_fails_without_memory_update_when_required(self):
        """Test that missing memory_update raises error when required."""
        text = '{"phase_artifacts": {"download": {}, "coding": {}, "compile": {}, "run": {}}}'

        with pytest.raises(Exception):  # MissingMemoryUpdateError
            extract_phase_artifacts(text, require_memory_update=True)

    def test_python_experiment_skips_compiler_validation(self):
        """Test that Python experiments don't require compiler_selected."""
        text = """<memory_update>
{"mem_core_set": {"test": "value"}}
</memory_update>
{
  "phase_artifacts": {
    "download": {"commands": [], "notes": "test"},
    "coding": {
      "workspace": {
        "root": "/workspace",
        "tree": ["workspace/", "workspace/src/"],
        "files": [
          {"path": "src/main.py", "mode": "0644", "encoding": "utf-8", "content": "print('hello')"}
        ]
      },
      "notes": "test"
    },
    "compile": {
      "build_plan": {
        "language": "python",
        "compiler_selected": "",
        "cflags": [],
        "ldflags": [],
        "workdir": "/workspace",
        "output": "working/test_data.npy"
      },
      "commands": [],
      "notes": "no compile needed for python"
    },
    "run": {
      "commands": ["python3 src/main.py"],
      "expected_outputs": ["working/test_data.npy"],
      "notes": "test"
    }
  },
  "constraints": {}
}"""

        # Should not raise PhasePlanError for missing compiler_selected
        result = extract_phase_artifacts(text, require_memory_update=True)
        assert result["phase_artifacts"]["compile"]["build_plan"]["language"] == "python"
        assert result["phase_artifacts"]["compile"]["build_plan"]["compiler_selected"] == ""


class TestRealWorldScenario:
    """Test with real-world LLM output patterns that caused parse failures."""

    def test_debug_phase_with_memory_results(self):
        """Test the exact pattern that caused parse failures in debug phase."""
        # This is the actual pattern from the experiment logs
        text = """<memory_update>
{
  "mem_archival_search": {
    "query": "compiler_selected 'python' not in available_compilers",
    "k": 5
  }
}
</memory_update>
<memory_results>
{
  "mem_archival_search": []
}
</memory_results>
<memory_update>
{
  "mem_core_set": {
    "last_bug_type": "orchestrator_compiler_selection_mismatch"
  }
}
</memory_update>
{
  "phase_artifacts": {
    "download": {
      "commands": ["command -v gcc"],
      "notes": "test"
    },
    "coding": {
      "workspace": {
        "root": "/workspace",
        "tree": ["workspace/", "workspace/src/"],
        "files": [
          {"path": "src/main.c", "mode": "0644", "encoding": "utf-8", "content": "int main() { return 0; }"}
        ]
      },
      "notes": "test"
    },
    "compile": {
      "build_plan": {
        "language": "c",
        "compiler_selected": "gcc",
        "cflags": [],
        "ldflags": [],
        "workdir": "/workspace",
        "output": "bin/a.out"
      },
      "commands": ["gcc -o bin/a.out src/main.c"],
      "notes": "test"
    },
    "run": {
      "commands": ["./bin/a.out"],
      "expected_outputs": ["working/test_data.npy"],
      "notes": "test"
    }
  },
  "constraints": {}
}"""

        # This should NOT raise a parse error anymore
        result = extract_phase_artifacts(text, require_memory_update=True)

        assert "phase_artifacts" in result
        assert "memory_update" in result
        # First memory_update should be extracted (with read operation)
        assert "mem_archival_search" in result["memory_update"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
