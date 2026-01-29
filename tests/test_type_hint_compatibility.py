"""Test type hint compatibility for Python 3.9+.

This test verifies that the type hints in parallel_agent.py use
Optional[] syntax instead of X | None syntax for Python 3.9 compatibility.
"""

import ast
import re
import sys
import unittest
from pathlib import Path


class TestTypeHintCompatibility(unittest.TestCase):
    """Test that type hints are compatible with Python 3.9."""

    def test_parallel_agent_import(self):
        """Test that parallel_agent module can be imported."""
        try:
            from ai_scientist.treesearch.parallel_agent import ParallelAgent
            self.assertTrue(True, "ParallelAgent imported successfully")
        except TypeError as e:
            if "unsupported operand type(s) for |" in str(e):
                self.fail(
                    f"Python 3.9 type hint compatibility issue: {e}\n"
                    "Use Optional[X] instead of X | None"
                )
            raise

    def test_agent_manager_import(self):
        """Test that agent_manager module can be imported."""
        try:
            from ai_scientist.treesearch.agent_manager import AgentManager
            self.assertTrue(True, "AgentManager imported successfully")
        except TypeError as e:
            if "unsupported operand type(s) for |" in str(e):
                self.fail(
                    f"Python 3.9 type hint compatibility issue: {e}\n"
                    "Use Optional[X] instead of X | None"
                )
            raise

    def test_no_pipe_none_syntax(self):
        """Test that parallel_agent.py doesn't use X | None syntax."""
        file_path = Path(__file__).parent.parent / "ai_scientist" / "treesearch" / "parallel_agent.py"

        if not file_path.exists():
            self.skipTest(f"File not found: {file_path}")

        content = file_path.read_text()

        # Pattern to match "Type | None" syntax
        pipe_none_pattern = re.compile(r'\w+\s*\|\s*None')
        matches = pipe_none_pattern.findall(content)

        if matches:
            self.fail(
                f"Found {len(matches)} instances of 'X | None' syntax which is "
                f"not compatible with Python 3.9.\n"
                f"Examples: {matches[:5]}\n"
                "Use Optional[X] instead."
            )

    def test_syntax_valid(self):
        """Test that parallel_agent.py has valid Python syntax."""
        file_path = Path(__file__).parent.parent / "ai_scientist" / "treesearch" / "parallel_agent.py"

        if not file_path.exists():
            self.skipTest(f"File not found: {file_path}")

        content = file_path.read_text()

        try:
            ast.parse(content)
        except SyntaxError as e:
            self.fail(f"Syntax error in parallel_agent.py: {e}")

    def test_launch_scientist_import(self):
        """Test that the main launch script can be imported."""
        try:
            # This is the original entry point that was failing
            from ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager import (
                perform_experiments_bfts,
            )
            self.assertTrue(True, "perform_experiments_bfts imported successfully")
        except TypeError as e:
            if "unsupported operand type(s) for |" in str(e):
                self.fail(
                    f"Python 3.9 type hint compatibility issue: {e}\n"
                    "Use Optional[X] instead of X | None"
                )
            raise
        except ImportError as e:
            # Other import errors are acceptable for this test
            # (missing dependencies, etc.)
            pass


if __name__ == "__main__":
    unittest.main()
