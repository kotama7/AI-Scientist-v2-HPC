import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from ai_scientist.memory import MemoryManager


class TestPhase0InternalPersisted(unittest.TestCase):
    def test_phase0_internal_info_ingestion(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "memory.sqlite"
            cfg = SimpleNamespace(
                core_max_chars=2000,
                recall_max_events=5,
                retrieval_k=5,
                use_fts="off",
                always_inject_idea_summary=False,
                always_inject_phase0_summary=True,
            )
            mem = MemoryManager(db_path, cfg)
            root = mem.create_branch(None, node_uid="root")
            payload = {
                "threads": 8,
                "pinning": "compact",
                "numa": "interleave",
                "plan": {"compile": {"commands": ["make"]}},
            }
            mem.ingest_phase0_internal_info(
                root,
                node_uid="node0",
                phase0_payload=payload,
                artifact_paths=[],
                command_str="run.sh --threads 8",
            )

            json_path = Path(tmpdir) / "phase0_internal_info.json"
            md_path = Path(tmpdir) / "phase0_internal_info.md"
            self.assertFalse(json_path.exists())
            self.assertFalse(md_path.exists())

            archival = mem.retrieve_archival(
                root,
                query="threads",
                k=5,
                include_ancestors=True,
                tags_filter=["PHASE0_INTERNAL"],
            )
            self.assertTrue(len(archival) >= 1)

            child = mem.create_branch(root, node_uid="child")
            prompt_child = mem.render_for_prompt(child, task_hint="threads", budget_chars=3000)
            self.assertIn("Phase 0 internal summary", prompt_child)
