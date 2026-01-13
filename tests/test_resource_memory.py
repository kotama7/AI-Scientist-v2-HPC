import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from ai_scientist.memory import MemoryManager
from ai_scientist.memory.resource_memory import (
    RESOURCE_DIGEST_KEY,
    RESOURCE_INDEX_KEY,
    RESOURCE_ITEM_MAX_CHARS,
    build_resource_snapshot,
    persist_resource_snapshot_to_ltm,
    update_resource_snapshot_if_changed,
)
from ai_scientist.treesearch.utils.resource import load_resources, stage_resource_items


class TestResourceMemory(unittest.TestCase):
    def _make_memory(self, root: Path) -> tuple[MemoryManager, str]:
        db_path = root / "memory.sqlite"
        cfg = SimpleNamespace(
            core_max_chars=2000,
            recall_max_events=5,
            retrieval_k=5,
            use_fts="off",
            always_inject_idea_summary=False,
            always_inject_phase0_summary=False,
            root_branch_id=None,
        )
        mem = MemoryManager(db_path, cfg)
        branch_id = mem.create_branch(None, node_uid="root")
        cfg.root_branch_id = branch_id
        return mem, branch_id

    def test_snapshot_persist_and_digest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            workspace_root = root / "workspace"
            workspace_root.mkdir()
            local_root = root / "local"
            local_root.mkdir()
            template_dir = local_root / "template"
            template_dir.mkdir()
            (template_dir / "main.py").write_text("print('hello')\n", encoding="utf-8")
            (template_dir / "README.md").write_text("Template docs\n", encoding="utf-8")
            doc_path = local_root / "doc.md"
            doc_path.write_text("Document content\n", encoding="utf-8")

            resource_data = {
                "local": [
                    {
                        "name": "local_data",
                        "host_path": str(local_root),
                        "mount_path": "/workspace/input/local_data",
                    }
                ],
                "github": [],
                "huggingface": [],
                "items": [
                    {
                        "name": "baseline_template",
                        "class": "template",
                        "source": "local",
                        "resource": "local_data",
                        "path": "template",
                        "include_tree": True,
                        "include_files": ["main.py"],
                    },
                    {
                        "name": "doc_item",
                        "class": "document",
                        "source": "local",
                        "resource": "local_data",
                        "path": "doc.md",
                        "include_content": True,
                    },
                ],
            }
            resource_path = root / "resources.json"
            resource_path.write_text(json.dumps(resource_data), encoding="utf-8")

            resources_cfg = load_resources(resource_path)
            stage_resource_items(resources_cfg, workspace_root / "resources")

            mem, branch_id = self._make_memory(root)
            snapshot = build_resource_snapshot(
                resource_path,
                workspace_root=workspace_root,
                ai_scientist_root=None,
                phase_mode="single",
                log=None,
            )
            persist_resource_snapshot_to_ltm(snapshot, mem)

            index = mem.get_core(branch_id, RESOURCE_INDEX_KEY)
            self.assertTrue(index)
            staged_path = workspace_root / "resources" / "template" / "baseline_template"
            self.assertIn(str(staged_path), index)

            items = mem.retrieve_archival(
                branch_id,
                query="baseline_template",
                k=5,
                include_ancestors=True,
                tags_filter=["RESOURCE_ITEM"],
            )
            self.assertTrue(items)
            for row in items:
                self.assertLessEqual(len(row.get("text", "")), RESOURCE_ITEM_MAX_CHARS)

            snapshot_again = build_resource_snapshot(
                resource_path,
                workspace_root=workspace_root,
                ai_scientist_root=None,
                phase_mode="single",
                log=None,
            )
            self.assertEqual(snapshot.resource_digest, snapshot_again.resource_digest)

            (template_dir / "main.py").write_text("print('changed')\n", encoding="utf-8")
            snapshot_changed = build_resource_snapshot(
                resource_path,
                workspace_root=workspace_root,
                ai_scientist_root=None,
                phase_mode="single",
                log=None,
            )
            self.assertNotEqual(snapshot.resource_digest, snapshot_changed.resource_digest)

    def test_pending_github_item_updates_after_fetch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            workspace_root = root / "workspace"
            workspace_root.mkdir()

            resource_data = {
                "local": [],
                "github": [
                    {
                        "name": "gh_lib",
                        "repo": "https://github.com/example/repo.git",
                        "ref": "main",
                        "dest": "/workspace/gh_lib",
                    }
                ],
                "huggingface": [],
                "items": [
                    {
                        "name": "gh_item",
                        "class": "library",
                        "source": "github",
                        "resource": "gh_lib",
                        "path": ".",
                        "include_tree": True,
                        "include_files": ["README.md"],
                        "include_content": True,
                    }
                ],
            }
            resource_path = root / "resources.json"
            resource_path.write_text(json.dumps(resource_data), encoding="utf-8")

            mem, branch_id = self._make_memory(root)
            snapshot = build_resource_snapshot(
                resource_path,
                workspace_root=workspace_root,
                ai_scientist_root=None,
                phase_mode="single",
                log=None,
            )
            persist_resource_snapshot_to_ltm(snapshot, mem)

            pending = mem.retrieve_archival(
                branch_id,
                query="gh_item",
                k=1,
                include_ancestors=True,
                tags_filter=["RESOURCE_ITEM"],
            )
            self.assertTrue(pending)
            self.assertIn("pending fetch", pending[0].get("text", "").lower())

            gh_root = workspace_root / "gh_lib"
            gh_root.mkdir(parents=True, exist_ok=True)
            (gh_root / "README.md").write_text("Fetched content\n", encoding="utf-8")

            updated = build_resource_snapshot(
                resource_path,
                workspace_root=workspace_root,
                ai_scientist_root=None,
                phase_mode="single",
                log=None,
            )
            update_resource_snapshot_if_changed(updated, mem)

            refreshed = mem.retrieve_archival(
                branch_id,
                query="gh_item",
                k=1,
                include_ancestors=True,
                tags_filter=["RESOURCE_ITEM"],
            )
            self.assertTrue(refreshed)
            self.assertNotIn("pending fetch", refreshed[0].get("text", "").lower())
            self.assertIn("Fetched content", refreshed[0].get("text", ""))
            self.assertEqual(mem.get_core(branch_id, RESOURCE_DIGEST_KEY), updated.resource_digest)


if __name__ == "__main__":
    unittest.main()
