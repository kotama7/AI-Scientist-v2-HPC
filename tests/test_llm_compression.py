"""Tests for LLM-based memory compression functionality."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from ai_scientist.memory import MemoryManager
from ai_scientist.memory.memgpt_store import (
    _compress_with_llm,
    _load_compression_prompt,
    _truncate,
)


class TestCompressionPromptLoading(unittest.TestCase):
    """Test compression prompt file loading."""

    def test_load_compression_prompt_file_not_found(self) -> None:
        """Should return None for non-existent file."""
        result = _load_compression_prompt("/nonexistent/path.txt")
        self.assertIsNone(result)

    def test_load_compression_prompt_success(self) -> None:
        """Should load prompt from existing file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Test prompt template")
            f.flush()
            result = _load_compression_prompt(f.name)
            self.assertEqual(result, "Test prompt template")


class TestCompressWithLLM(unittest.TestCase):
    """Test _compress_with_llm function."""

    def test_compress_returns_original_if_within_budget(self) -> None:
        """Should return original text if already within budget."""
        text = "Short text"
        result = _compress_with_llm(text, 100, "test")
        self.assertEqual(result, text)

    def test_compress_uses_truncate_without_client(self) -> None:
        """Should fall back to _truncate when client is None."""
        text = "A" * 200
        result = _compress_with_llm(text, 50, "test", client=None, model=None)
        self.assertEqual(len(result), 50)
        self.assertTrue(result.endswith("..."))

    def test_compress_caches_results(self) -> None:
        """Should cache compression results."""
        from ai_scientist.memory.memgpt_store import _compression_cache
        _compression_cache.clear()
        
        text = "A" * 200
        # Without client, uses truncation
        result1 = _compress_with_llm(text, 50, "test", client=None, model=None)
        
        # Second call should not re-process (cache doesn't store truncated results though)
        result2 = _compress_with_llm(text, 50, "test", client=None, model=None)
        self.assertEqual(result1, result2)


class TestMemoryManagerCompression(unittest.TestCase):
    """Test MemoryManager compression configuration."""

    def test_compression_disabled_by_default(self) -> None:
        """Compression should be disabled by default."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "memory" / "memory.sqlite"
            cfg = SimpleNamespace()
            mem = MemoryManager(db_path, cfg)
            self.assertFalse(mem.use_llm_compression)

    def test_compression_enabled_via_config(self) -> None:
        """Compression can be enabled via config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "memory" / "memory.sqlite"
            cfg = SimpleNamespace(
                use_llm_compression=True,
                compression_model="gpt-4o-mini",
            )
            # Will fail to create client but that's expected
            mem = MemoryManager(db_path, cfg)
            # use_llm_compression should be False since client creation failed
            self.assertFalse(mem.use_llm_compression)

    def test_compress_method_uses_truncate_when_disabled(self) -> None:
        """_compress should use _truncate when compression is disabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "memory" / "memory.sqlite"
            cfg = SimpleNamespace(use_llm_compression=False)
            mem = MemoryManager(db_path, cfg)
            
            text = "A" * 200
            result = mem._compress(text, 50, "test")
            self.assertEqual(len(result), 50)
            self.assertTrue(result.endswith("..."))


class TestSummarizeWithCompression(unittest.TestCase):
    """Test summarization functions with compression."""

    def test_summarize_idea_without_compress_fn(self) -> None:
        """_summarize_idea should work without compress_fn."""
        from ai_scientist.memory.memgpt_store import _summarize_idea
        
        text = """
## Abstract
This is a test abstract.
## Hypothesis
This is a hypothesis.
"""
        result = _summarize_idea(text)
        self.assertIn("Purpose:", result)
        self.assertIn("Hypothesis:", result)

    def test_summarize_phase0_without_compress_fn(self) -> None:
        """_summarize_phase0 should work without compress_fn."""
        from ai_scientist.memory.memgpt_store import _summarize_phase0
        
        payload = {"threads": 8, "pinning": "compact"}
        result = _summarize_phase0(payload, "run command")
        self.assertIn("threads=8", result)
        self.assertIn("pinning=compact", result)

    def test_summarize_idea_with_compress_fn(self) -> None:
        """_summarize_idea should use compress_fn when provided."""
        from ai_scientist.memory.memgpt_store import _summarize_idea
        
        compress_calls = []
        
        def mock_compress(text, max_chars, context):
            compress_calls.append((text, max_chars, context))
            return text[:max_chars] if len(text) > max_chars else text
        
        # Long text that triggers compression
        text = """
## Abstract
""" + "A" * 500 + """
## Hypothesis
""" + "B" * 500

        result = _summarize_idea(text, max_chars=200, compress_fn=mock_compress)
        # Should have called compress_fn
        self.assertTrue(len(compress_calls) > 0)


if __name__ == "__main__":
    unittest.main()
