"""Tests for ai_scientist.treesearch.utils.config module."""

import tempfile
import unittest
from pathlib import Path

import yaml
from omegaconf import OmegaConf

from ai_scientist.treesearch.utils.config import (
    MemoryConfig,
    ExecConfig,
    Config,
    _auto_detect_language,
)


class TestMemoryConfig(unittest.TestCase):
    """Test cases for MemoryConfig dataclass."""

    def test_default_values(self):
        """Test that MemoryConfig has correct default values."""
        cfg = MemoryConfig()
        # Note: enabled is False by default in the dataclass,
        # but in bfts_config.yaml it's set to True
        self.assertFalse(cfg.enabled)
        self.assertIsNone(cfg.db_path)
        self.assertEqual(cfg.core_max_chars, 2000)
        self.assertEqual(cfg.recall_max_events, 20)
        self.assertEqual(cfg.retrieval_k, 8)
        self.assertEqual(cfg.use_fts, "auto")
        self.assertTrue(cfg.final_memory_enabled)
        self.assertEqual(cfg.final_memory_filename_md, "final_memory_for_paper.md")
        self.assertEqual(cfg.final_memory_filename_json, "final_memory_for_paper.json")
        self.assertTrue(cfg.redact_secrets)
        self.assertEqual(cfg.max_memory_read_rounds, 2)

    def test_custom_values(self):
        """Test creating MemoryConfig with custom values."""
        cfg = MemoryConfig(
            enabled=True,
            core_max_chars=4000,
            recall_max_events=50,
            max_memory_read_rounds=5,
        )
        self.assertTrue(cfg.enabled)
        self.assertEqual(cfg.core_max_chars, 4000)
        self.assertEqual(cfg.recall_max_events, 50)
        self.assertEqual(cfg.max_memory_read_rounds, 5)


class TestExecConfig(unittest.TestCase):
    """Test cases for ExecConfig dataclass."""

    def test_default_values(self):
        """Test that ExecConfig has correct default values."""
        cfg = ExecConfig()
        self.assertEqual(cfg.timeout, 3600)
        self.assertEqual(cfg.agent_file_name, "runfile.py")
        self.assertTrue(cfg.format_tb_ipython)
        self.assertEqual(cfg.language, "auto")
        self.assertEqual(cfg.phase_mode, "split")
        self.assertTrue(cfg.use_gpu)
        self.assertEqual(cfg.workspace_mount, "/workspace")
        self.assertEqual(cfg.phase1_max_steps, 12)
        self.assertTrue(cfg.log_prompts)

    def test_custom_values(self):
        """Test creating ExecConfig with custom values."""
        cfg = ExecConfig(
            timeout=7200,
            language="cpp",
            phase_mode="single",
            phase1_max_steps=20,
        )
        self.assertEqual(cfg.timeout, 7200)
        self.assertEqual(cfg.language, "cpp")
        self.assertEqual(cfg.phase_mode, "single")
        self.assertEqual(cfg.phase1_max_steps, 20)


class TestBftsConfigYaml(unittest.TestCase):
    """Test cases for bfts_config.yaml loading."""

    @classmethod
    def setUpClass(cls):
        """Load the actual bfts_config.yaml file."""
        config_path = Path(__file__).parent.parent / "bfts_config.yaml"
        if config_path.exists():
            cls.config = OmegaConf.load(config_path)
        else:
            cls.config = None

    def test_config_file_exists(self):
        """Test that bfts_config.yaml exists."""
        config_path = Path(__file__).parent.parent / "bfts_config.yaml"
        self.assertTrue(config_path.exists(), "bfts_config.yaml not found")

    def test_memory_section_exists(self):
        """Test that memory section exists in config."""
        if self.config is None:
            self.skipTest("Config file not loaded")
        self.assertIn("memory", self.config)

    def test_memory_enabled_default_true(self):
        """Test that memory.enabled defaults to true in bfts_config.yaml."""
        if self.config is None:
            self.skipTest("Config file not loaded")
        # In the actual config file, memory.enabled should be true
        self.assertTrue(self.config.memory.enabled)

    def test_exec_section_exists(self):
        """Test that exec section exists in config."""
        if self.config is None:
            self.skipTest("Config file not loaded")
        self.assertIn("exec", self.config)

    def test_agent_section_exists(self):
        """Test that agent section exists in config."""
        if self.config is None:
            self.skipTest("Config file not loaded")
        self.assertIn("agent", self.config)

    def test_section_budgets_flat_structure(self):
        """Test that section budgets use flat structure under memory."""
        if self.config is None:
            self.skipTest("Config file not loaded")
        memory = self.config.memory

        # These should be flat keys under memory, not nested
        budget_keys = [
            "datasets_tested_budget_chars",
            "metrics_extraction_budget_chars",
            "plotting_code_budget_chars",
            "plot_selection_budget_chars",
            "vlm_analysis_budget_chars",
            "node_summary_budget_chars",
            "parse_metrics_budget_chars",
            "archival_snippet_budget_chars",
            "results_budget_chars",
        ]

        for key in budget_keys:
            self.assertIn(key, memory, f"Missing budget key: {key}")
            self.assertIsInstance(memory[key], int, f"Budget {key} should be int")


class TestAutoDetectLanguage(unittest.TestCase):
    """Test cases for language auto-detection."""

    def test_python_detection_from_files(self):
        """Test Python detection from .py files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            # Create Python files
            (tmpdir_path / "main.py").touch()
            (tmpdir_path / "utils.py").touch()

            # Create minimal config
            cfg = OmegaConf.create({
                "data_dir": str(tmpdir_path),
                "log_dir": str(tmpdir_path),
                "workspace_dir": str(tmpdir_path),
                "copy_data": False,
                "exp_name": "test",
                "exec": {"timeout": 3600},
                "generate_report": False,
                "report": {"model": "gpt-4", "temp": 0.7},
                "agent": {
                    "steps": 10,
                    "stages": {},
                    "code": {"model": "gpt-4", "temp": 0.7},
                    "feedback": {"model": "gpt-4", "temp": 0.7},
                    "vlm_feedback": {"model": "gpt-4", "temp": 0.7},
                    "search": {"max_debug_depth": 3, "debug_prob": 0.5, "num_drafts": 3},
                    "num_workers": 1,
                    "type": "minimal",
                    "multi_seed_eval": {},
                },
                "experiment": {"num_syn_datasets": 1},
                "desc_file": None,
                "goal": None,
            })

            language = _auto_detect_language(cfg)
            self.assertEqual(language, "python")

    def test_cpp_detection_from_description(self):
        """Test C++ detection from task description."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            cfg = OmegaConf.create({
                "data_dir": str(tmpdir_path),
                "log_dir": str(tmpdir_path),
                "workspace_dir": str(tmpdir_path),
                "copy_data": False,
                "exp_name": "test",
                "exec": {"timeout": 3600},
                "generate_report": False,
                "report": {"model": "gpt-4", "temp": 0.7},
                "agent": {
                    "steps": 10,
                    "stages": {},
                    "code": {"model": "gpt-4", "temp": 0.7},
                    "feedback": {"model": "gpt-4", "temp": 0.7},
                    "vlm_feedback": {"model": "gpt-4", "temp": 0.7},
                    "search": {"max_debug_depth": 3, "debug_prob": 0.5, "num_drafts": 3},
                    "num_workers": 1,
                    "type": "minimal",
                    "multi_seed_eval": {},
                },
                "experiment": {"num_syn_datasets": 1},
                "desc_file": None,
                "goal": "Implement a C++ matrix multiplication library",
            })

            language = _auto_detect_language(cfg)
            self.assertEqual(language, "cpp")

    def test_default_python_when_no_indicators(self):
        """Test that Python is default when no indicators present."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            cfg = OmegaConf.create({
                "data_dir": str(tmpdir_path),
                "log_dir": str(tmpdir_path),
                "workspace_dir": str(tmpdir_path),
                "copy_data": False,
                "exp_name": "test",
                "exec": {"timeout": 3600},
                "generate_report": False,
                "report": {"model": "gpt-4", "temp": 0.7},
                "agent": {
                    "steps": 10,
                    "stages": {},
                    "code": {"model": "gpt-4", "temp": 0.7},
                    "feedback": {"model": "gpt-4", "temp": 0.7},
                    "vlm_feedback": {"model": "gpt-4", "temp": 0.7},
                    "search": {"max_debug_depth": 3, "debug_prob": 0.5, "num_drafts": 3},
                    "num_workers": 1,
                    "type": "minimal",
                    "multi_seed_eval": {},
                },
                "experiment": {"num_syn_datasets": 1},
                "desc_file": None,
                "goal": "Generic task",
            })

            language = _auto_detect_language(cfg)
            self.assertEqual(language, "python")


if __name__ == "__main__":
    unittest.main()
