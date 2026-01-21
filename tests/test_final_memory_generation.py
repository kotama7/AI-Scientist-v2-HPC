import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from ai_scientist.memory import MemoryManager
from ai_scientist.memory.memgpt_store import _summarize_phase0


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


class TestSummarizePhase0(unittest.TestCase):
    """Tests for _summarize_phase0 function to ensure PC info extraction works."""

    def test_extract_env_context_from_direct_payload(self) -> None:
        """Test extracting environment_context directly from payload."""
        payload = {
            "environment_context": {
                "cpu_info": "x86_64; 256 CPUs online; AMD EPYC 9534",
                "os_release": 'PRETTY_NAME="Ubuntu 22.04.5 LTS"',
                "available_compilers": [
                    {"name": "gcc", "version": "11.4.0"},
                    {"name": "g++", "version": "11.4.0"},
                ],
                "container_runtime": "singularity",
            }
        }
        result = _summarize_phase0(payload, None)
        self.assertIn("OS=Ubuntu 22.04.5 LTS", result)
        self.assertIn("compilers=[gcc:11.4.0, g++:11.4.0]", result)
        self.assertIn("container=singularity", result)

    def test_extract_env_context_from_nested_artifacts(self) -> None:
        """Test extracting environment_context from nested artifacts array."""
        env_ctx = {
            "cpu_info": "x86_64; 256 CPUs online; AMD EPYC 9534 64-Core Processor; 2 sockets",
            "memory_info": "RAM total 1.5TiB",
            "gpu_info": "4x NVIDIA H100 PCIe",
            "os_release": 'PRETTY_NAME="Ubuntu 22.04.5 LTS (jammy)"',
            "available_compilers": [
                {"name": "gcc", "version": "11.4.0 (Ubuntu 22.04)"},
                {"name": "g++", "version": "11.4.0 (Ubuntu 22.04)"},
                {"name": "nvcc", "version": "NVIDIA CUDA compiler driver"},
            ],
            "container_runtime": "singularity",
        }
        artifact_content = json.dumps({"environment_context": env_ctx})
        payload = {
            "plan": {
                "goal_summary": "Create a minimal prototype",
            },
            "artifacts": [
                {"path": "/some/plan.json", "content": '{"foo": "bar"}'},
                {"path": "/some/history.json", "content": artifact_content},
            ],
        }
        result = _summarize_phase0(payload, None)
        self.assertIn("OS=Ubuntu 22.04.5 LTS (jammy)", result)
        self.assertIn("compilers=[gcc:11.4.0, g++:11.4.0, nvcc:NVIDIA]", result)
        self.assertIn("container=singularity", result)

    def test_no_env_context_fallback(self) -> None:
        """Test fallback when no environment_context is found."""
        payload = {
            "plan": {"goal_summary": "Do something"},
        }
        result = _summarize_phase0(payload, None)
        self.assertIn("No structured Phase 0 info captured", result)

    def test_command_string_included(self) -> None:
        """Test that command string is included in summary."""
        payload = {}
        result = _summarize_phase0(payload, "sbatch run.sh")
        self.assertIn("command=sbatch run.sh", result)

    def test_condensed_cpu_and_simple_os_format(self) -> None:
        """Test parsing condensed CPU info and simple OS string format."""
        env_ctx = {
            "cpu_info": "x86_64; 256 CPUs online (0-255); AMD EPYC 9534 64-Core Processor; "
                        "2 sockets, 64 cores/socket, 2 threads/core; "
                        "NUMA nodes:2 (node0:0-63,128-191; node1:64-127,192-255)",
            "os_release": "Ubuntu 22.04.5 LTS (jammy)",
            "available_compilers": [
                {"name": "gcc", "version": "11.4.0"},
            ],
            "container_runtime": "singularity",
        }
        artifact_content = json.dumps({"environment_context": env_ctx})
        payload = {
            "artifacts": [
                {"path": "/some/history.json", "content": artifact_content},
            ],
        }
        result = _summarize_phase0(payload, None)
        self.assertIn("CPU:AMD EPYC 9534 64-Core Processor", result)
        self.assertIn("socket", result.lower())
        self.assertIn("numa", result.lower())
        self.assertIn("OS=Ubuntu 22.04.5 LTS (jammy)", result)
        self.assertIn("compilers=[gcc:11.4.0]", result)
        self.assertIn("container=singularity", result)
