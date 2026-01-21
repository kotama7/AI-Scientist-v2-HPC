import subprocess
import sys
import unittest
from pathlib import Path

from ai_scientist import prompt_loader


class SplitSmokeTests(unittest.TestCase):
    def test_import_split_entrypoints(self) -> None:
        import ai_scientist.treesearch.parallel_agent
        import ai_scientist.treesearch.utils.phase_execution
        import ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager

    def test_import_resource_module(self) -> None:
        from ai_scientist.treesearch.utils.resource import (
            ResourceConfig,
            load_resources,
            build_local_binds,
            build_resources_context,
        )

    def test_prompt_loading(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        prompt_root = repo_root / "prompt"
        prompt_loader.PROMPT_DIR = prompt_root
        prompt_loader.load_prompt.cache_clear()

        for path in prompt_root.rglob("*.txt"):
            rel = path.relative_to(prompt_root).as_posix()
            content = prompt_loader.load_prompt(rel)
            self.assertIsInstance(content, str)

    def test_resources_prompt_exists(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        prompt_root = repo_root / "prompt"
        resources_prompt = prompt_root / "config" / "environment" / "resources_injection.txt"
        self.assertTrue(resources_prompt.exists(), "config/environment/resources_injection.txt should exist")

    def test_cli_help(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, str(repo_root / "launch_scientist_bfts.py"), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0)

    def test_cli_resources_in_help(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, str(repo_root / "launch_scientist_bfts.py"), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertIn("--resources", result.stdout)

    def test_phase1_prompt_exists(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        prompt_root = repo_root / "prompt"
        phase1_prompt = prompt_root / "config" / "phases" / "phase1_installer.txt"
        self.assertTrue(phase1_prompt.exists(), "config/phases/phase1_installer.txt should exist")
        content = phase1_prompt.read_text()
        self.assertIn("Confirmation before done=true", content)
        self.assertIn("curl", content)
        self.assertIn("wget", content)


class DownloadInfoExtractionTests(unittest.TestCase):
    def test_curl_output_option(self) -> None:
        from ai_scientist.treesearch.utils.phase_execution import extract_download_info

        cmd = "curl -fsSL https://example.com/file.tar.gz -o /workspace/deps/file.tar.gz"
        result = extract_download_info(cmd)
        self.assertIsNotNone(result)
        self.assertEqual(result["url"], "https://example.com/file.tar.gz")
        self.assertEqual(result["dest"], "/workspace/deps/file.tar.gz")

    def test_wget_output_option(self) -> None:
        from ai_scientist.treesearch.utils.phase_execution import extract_download_info

        cmd = "wget -q https://example.com/data.zip -O /workspace/data/data.zip"
        result = extract_download_info(cmd)
        self.assertIsNotNone(result)
        self.assertEqual(result["url"], "https://example.com/data.zip")
        self.assertEqual(result["dest"], "/workspace/data/data.zip")

    def test_no_download_command(self) -> None:
        from ai_scientist.treesearch.utils.phase_execution import extract_download_info

        cmd = "apt-get update && apt-get install -y zlib1g-dev"
        result = extract_download_info(cmd)
        self.assertIsNone(result)

    def test_curl_url_before_output(self) -> None:
        from ai_scientist.treesearch.utils.phase_execution import extract_download_info

        cmd = "curl https://github.com/repo/archive.tar.gz --output /workspace/archive.tar.gz"
        result = extract_download_info(cmd)
        self.assertIsNotNone(result)
        self.assertEqual(result["url"], "https://github.com/repo/archive.tar.gz")
        self.assertEqual(result["dest"], "/workspace/archive.tar.gz")


if __name__ == "__main__":
    unittest.main()

