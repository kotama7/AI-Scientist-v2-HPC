import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from ai_scientist.memory import MemoryManager


class TestIdeaMdPersistedAndInjected(unittest.TestCase):
    def test_idea_md_archival_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            idea_path = Path(tmpdir) / "idea.md"
            idea_path.write_text(
                "## Title\nTest Idea\n\n"
                "## Abstract\nWe study memory inheritance.\n\n"
                "## Short Hypothesis\nChild nodes inherit memory.\n\n"
                "## Experiments\nRun A/B tests.\n",
                encoding="utf-8",
            )
            db_path = Path(tmpdir) / "memory.sqlite"
            cfg = SimpleNamespace(
                core_max_chars=2000,
                recall_max_events=5,
                retrieval_k=5,
                use_fts="off",
                always_inject_idea_summary=True,
                always_inject_phase0_summary=False,
            )
            mem = MemoryManager(db_path, cfg)
            root = mem.create_branch(None, node_uid="root")
            mem.ingest_idea_md(root, node_uid="root", idea_path=idea_path, is_root=True)

            archival = mem.retrieve_archival(
                root,
                query="inherit",
                k=5,
                include_ancestors=True,
                tags_filter=["IDEA_MD"],
            )
            self.assertTrue(len(archival) >= 1)
            prompt = mem.render_for_prompt(root, task_hint="inherit", budget_chars=3000)
            self.assertIn("Idea summary", prompt)
