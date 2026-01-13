import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from ai_scientist.memory import MemoryManager


class TestMemGPTBranchInheritance(unittest.TestCase):
    def test_branch_inheritance_and_isolation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "memory.sqlite"
            cfg = SimpleNamespace(
                core_max_chars=2000,
                recall_max_events=5,
                retrieval_k=5,
                use_fts="off",
                always_inject_idea_summary=False,
                always_inject_phase0_summary=False,
            )
            mem = MemoryManager(db_path, cfg)
            root = mem.create_branch(None, node_uid="root")
            child = mem.create_branch(root, node_uid="child")
            sibling = mem.create_branch(root, node_uid="sibling")

            mem.set_core(root, "constraint", "no network")
            mem.write_event(root, "note", "parent event", tags=["PARENT"])
            mem.write_archival(root, "parent archival", tags=["PARENT_ARCH"])

            prompt_child = mem.render_for_prompt(child, task_hint="parent", budget_chars=4000)
            self.assertIn("constraint", prompt_child)
            self.assertIn("parent event", prompt_child)
            self.assertIn("parent archival", prompt_child)

            mem.write_event(child, "note", "child event", tags=["CHILD"])
            prompt_sibling = mem.render_for_prompt(sibling, task_hint="child", budget_chars=4000)
            self.assertNotIn("child event", prompt_sibling)
