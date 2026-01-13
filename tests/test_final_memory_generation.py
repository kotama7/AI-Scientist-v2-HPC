import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from ai_scientist.memory import MemoryManager


class TestFinalMemoryGeneration(unittest.TestCase):
    def test_final_memory_files_created(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            db_path = run_dir / "memory" / "memory.sqlite"
            cfg = SimpleNamespace(
                core_max_chars=2000,
                recall_max_events=5,
                retrieval_k=5,
                use_fts="off",
                always_inject_idea_summary=False,
                always_inject_phase0_summary=False,
                final_memory_filename_md="final_memory_for_paper.md",
                final_memory_filename_json="final_memory_for_paper.json",
            )
            mem = MemoryManager(db_path, cfg)
            root = mem.create_branch(None, node_uid="root")
            mem.set_core(root, "phase0_summary", "threads=8 | pinning=compact")
            mem.write_archival(root, "IDEA_MD: Test idea content", tags=["IDEA_MD"])

            sections = mem.generate_final_memory_for_paper(
                run_dir=run_dir,
                root_branch_id=root,
                best_branch_id=root,
                artifacts_index={"log_dir": "logs"},
            )
            md_path = run_dir / "memory" / "final_memory_for_paper.md"
            json_path = run_dir / "memory" / "final_memory_for_paper.json"
            writeup_path = run_dir / "memory" / "final_writeup_memory.json"
            self.assertTrue(md_path.exists())
            self.assertTrue(json_path.exists())
            self.assertTrue(writeup_path.exists())
            md_text = md_path.read_text(encoding="utf-8")
            for heading in (
                "Title Candidates / Abstract Material",
                "Problem Statement / Motivation",
                "Hypothesis",
                "Method",
                "Experimental Setup",
                "Phase0 Internal Info Summary",
                "Results",
                "Ablations / Negative Results",
                "Failure Modes & Debugging Timeline",
                "Threats to Validity",
                "Reproducibility Checklist",
                "Narrative Bullets",
                "Resources Used",
            ):
                self.assertIn(heading, md_text)
            data = json.loads(json_path.read_text(encoding="utf-8"))
            for key in (
                "title_candidates",
                "abstract_material",
                "problem_statement",
                "hypothesis",
                "method",
                "experimental_setup",
                "phase0_internal_info_summary",
                "results",
                "ablations_negative",
                "failure_modes_timeline",
                "threats_to_validity",
                "reproducibility_checklist",
                "narrative_bullets",
                "resources_used",
            ):
                self.assertIn(key, data)
            self.assertIn("artifacts_index", sections)
            writeup = json.loads(writeup_path.read_text(encoding="utf-8"))
            for key in (
                "run_id",
                "idea",
                "phase0_env",
                "resources",
                "method_changes",
                "experiments",
                "results",
                "negative_results",
                "provenance",
            ):
                self.assertIn(key, writeup)
