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

    def test_prompt_loading(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        prompt_root = repo_root / "prompt"
        prompt_loader.PROMPT_DIR = prompt_root
        prompt_loader.load_prompt.cache_clear()

        for path in prompt_root.rglob("*.txt"):
            rel = path.relative_to(prompt_root).as_posix()
            content = prompt_loader.load_prompt(rel)
            self.assertIsInstance(content, str)

    def test_cli_help(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, str(repo_root / "launch_scientist_bfts.py"), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
