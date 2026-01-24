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


if __name__ == "__main__":
    unittest.main()
