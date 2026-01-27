"""Tests for ai_scientist.prompt_loader module."""

import tempfile
import unittest
from pathlib import Path

from ai_scientist.prompt_loader import (
    load_prompt,
    load_prompt_lines,
    format_prompt,
    load_prompt_from_dir,
    write_prompt,
    PromptNotFoundError,
    _resolve_prompt_path,
    _clear_load_prompt_cache,
)


class TestPromptLoader(unittest.TestCase):
    """Test cases for prompt loading functions."""

    def setUp(self):
        """Clear cache before each test."""
        _clear_load_prompt_cache()

    def test_load_existing_prompt(self):
        """Test loading an existing prompt file."""
        # Test loading a known existing prompt
        content = load_prompt("core/system")
        self.assertIsInstance(content, str)
        self.assertGreater(len(content), 0)

    def test_load_prompt_with_extension(self):
        """Test loading a prompt with explicit .txt extension."""
        content = load_prompt("core/system.txt")
        self.assertIsInstance(content, str)
        self.assertGreater(len(content), 0)

    def test_load_nonexistent_prompt_raises_error(self):
        """Test that loading a nonexistent prompt raises PromptNotFoundError."""
        with self.assertRaises(PromptNotFoundError):
            load_prompt("nonexistent/prompt/file")

    def test_load_prompt_lines(self):
        """Test loading a prompt as lines."""
        lines = load_prompt_lines("core/system")
        self.assertIsInstance(lines, list)
        self.assertGreater(len(lines), 0)
        for line in lines:
            self.assertIsInstance(line, str)

    def test_format_prompt(self):
        """Test formatting a prompt with variables."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            # Create a test prompt
            test_prompt = "Hello {name}, you are a {role}."
            prompt_file = tmpdir_path / "test_format.txt"
            prompt_file.write_text(test_prompt)

            # Load and format
            content = load_prompt_from_dir("test_format", tmpdir_path)
            formatted = content.format(name="Alice", role="researcher")
            self.assertEqual(formatted, "Hello Alice, you are a researcher.")

    def test_load_prompt_from_custom_dir(self):
        """Test loading a prompt from a custom directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            test_content = "Custom prompt content"
            prompt_file = tmpdir_path / "custom.txt"
            prompt_file.write_text(test_content)

            content = load_prompt_from_dir("custom", tmpdir_path)
            self.assertEqual(content, test_content)

    def test_write_prompt(self):
        """Test writing a prompt file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            test_content = "Written prompt content"

            write_prompt("new_prompt", test_content, base_dir=tmpdir_path)

            # Verify file was created
            prompt_file = tmpdir_path / "new_prompt.txt"
            self.assertTrue(prompt_file.exists())
            self.assertEqual(prompt_file.read_text(), test_content)

    def test_write_prompt_creates_subdirectories(self):
        """Test that write_prompt creates necessary subdirectories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            test_content = "Nested prompt content"

            write_prompt("nested/subdir/prompt", test_content, base_dir=tmpdir_path)

            prompt_file = tmpdir_path / "nested" / "subdir" / "prompt.txt"
            self.assertTrue(prompt_file.exists())
            self.assertEqual(prompt_file.read_text(), test_content)

    def test_resolve_prompt_path_adds_extension(self):
        """Test that _resolve_prompt_path adds .txt extension."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            path = _resolve_prompt_path("test", base_dir=tmpdir_path)
            self.assertEqual(path.suffix, ".txt")

    def test_resolve_prompt_path_preserves_extension(self):
        """Test that _resolve_prompt_path preserves existing extension."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            path = _resolve_prompt_path("test.json", base_dir=tmpdir_path)
            self.assertEqual(path.suffix, ".json")

    def test_cache_clearing(self):
        """Test that cache clearing works."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            prompt_file = tmpdir_path / "cached_test.txt"
            prompt_file.write_text("Initial content")

            # Load once
            content1 = load_prompt_from_dir("cached_test", tmpdir_path)
            self.assertEqual(content1, "Initial content")

            # Modify file
            prompt_file.write_text("Modified content")

            # Should still return cached value
            content2 = load_prompt_from_dir("cached_test", tmpdir_path)
            self.assertEqual(content2, "Initial content")

            # Clear cache and reload
            _clear_load_prompt_cache()
            content3 = load_prompt_from_dir("cached_test", tmpdir_path)
            self.assertEqual(content3, "Modified content")


class TestPromptFiles(unittest.TestCase):
    """Test that expected prompt files exist."""

    def test_phase0_prompts_exist(self):
        """Test that Phase 0 prompts exist."""
        content = load_prompt("config/phases/phase0_planning")
        self.assertIsInstance(content, str)
        self.assertGreater(len(content), 0)

        content_memory = load_prompt("config/phases/phase0_planning_with_memory")
        self.assertIsInstance(content_memory, str)
        self.assertGreater(len(content_memory), 0)

    def test_phase1_prompts_exist(self):
        """Test that Phase 1 prompts exist."""
        content = load_prompt("config/phases/phase1_installer")
        self.assertIsInstance(content, str)
        self.assertGreater(len(content), 0)

        content_memory = load_prompt("config/phases/phase1_installer_with_memory")
        self.assertIsInstance(content_memory, str)
        self.assertGreater(len(content_memory), 0)

    def test_task_prompts_exist(self):
        """Test that task prompts exist."""
        task_prompts = [
            "agent/parallel/tasks/draft/introduction",
            "agent/parallel/tasks/debug/introduction",
            "agent/parallel/tasks/improve/introduction",
            "agent/parallel/tasks/summary/introduction",
            "agent/parallel/tasks/parse_metrics/introduction",
            "agent/parallel/tasks/execution_review/introduction",
        ]
        for prompt_name in task_prompts:
            content = load_prompt(prompt_name)
            self.assertIsInstance(content, str, f"Failed for {prompt_name}")
            self.assertGreater(len(content), 0, f"Empty content for {prompt_name}")

    def test_memory_enabled_task_prompts_exist(self):
        """Test that memory-enabled task prompts exist."""
        memory_prompts = [
            "agent/parallel/tasks/draft/introduction_with_memory",
            "agent/parallel/tasks/debug/introduction_with_memory",
            "agent/parallel/tasks/improve/introduction_with_memory",
            "agent/parallel/tasks/summary/introduction_with_memory",
            "agent/parallel/tasks/parse_metrics/introduction_with_memory",
            "agent/parallel/tasks/execution_review/introduction_with_memory",
            "agent/parallel/nodes/hyperparam/introduction_with_memory",
            "agent/parallel/nodes/ablation/introduction_with_memory",
            "agent/parallel/vlm_analysis_with_memory",
        ]
        for prompt_name in memory_prompts:
            content = load_prompt(prompt_name)
            self.assertIsInstance(content, str, f"Failed for {prompt_name}")
            self.assertGreater(len(content), 0, f"Empty content for {prompt_name}")
            # Memory-enabled prompts should contain memory_update instructions
            self.assertIn("memory_update", content.lower(), f"No memory_update in {prompt_name}")

    def test_memory_config_prompts_exist(self):
        """Test that memory configuration prompts exist."""
        memory_prompts = [
            "config/memory/compression",
            "config/memory/compression_system_message",
            "config/memory/importance_evaluation",
            "config/memory/importance_evaluation_system_message",
            "config/memory/consolidation",
            "config/memory/consolidation_system_message",
            "config/memory/paper_section_generation",
            "config/memory/paper_section_generation_system_message",
            "config/memory/paper_section_outline",
            "config/memory/paper_section_outline_system_message",
            "config/memory/paper_section_fill",
            "config/memory/paper_section_fill_system_message",
            "config/memory/keyword_extraction",
            "config/memory/keyword_extraction_system_message",
        ]
        for prompt_name in memory_prompts:
            content = load_prompt(prompt_name)
            self.assertIsInstance(content, str, f"Failed for {prompt_name}")
            self.assertGreater(len(content), 0, f"Empty content for {prompt_name}")


if __name__ == "__main__":
    unittest.main()
