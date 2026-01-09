"""Tests for ai_scientist.treesearch.utils.resource module."""

import json
import tempfile
import unittest
from pathlib import Path

from ai_scientist.treesearch.utils.resource import (
    LocalResource,
    GitHubResource,
    HuggingFaceResource,
    ResourceItem,
    ResourceConfig,
    load_resources,
    build_local_binds,
    get_github_fetch_commands,
    get_huggingface_fetch_commands,
    build_resources_context,
)


class TestLocalResource(unittest.TestCase):
    def test_valid_local_resource(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            res = LocalResource(
                name="test_data",
                host_path=tmpdir,
                mount_path="/workspace/input/test",
                read_only=True,
            )
            errors = res.validate()
            self.assertEqual(errors, [])

    def test_invalid_mount_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            res = LocalResource(
                name="test_data",
                host_path=tmpdir,
                mount_path="/tmp/outside_workspace",  # Invalid
                read_only=True,
            )
            errors = res.validate()
            self.assertEqual(len(errors), 1)
            self.assertIn("/workspace", errors[0])

    def test_missing_host_path(self) -> None:
        res = LocalResource(
            name="test_data",
            host_path="/nonexistent/path/abc123",
            mount_path="/workspace/input/test",
            read_only=True,
        )
        errors = res.validate()
        self.assertEqual(len(errors), 1)
        self.assertIn("does not exist", errors[0])


class TestGitHubResource(unittest.TestCase):
    def test_valid_github_resource(self) -> None:
        res = GitHubResource(
            name="mylib",
            repo="https://github.com/user/repo.git",
            ref="v1.0.0",
            as_="library",
            dest="/workspace/third_party/mylib",
        )
        errors = res.validate()
        self.assertEqual(errors, [])

    def test_missing_ref_warns(self) -> None:
        res = GitHubResource(
            name="mylib",
            repo="https://github.com/user/repo.git",
            ref=None,  # Should warn
            dest="/workspace/third_party/mylib",
        )
        # Validation doesn't fail, just warns
        errors = res.validate()
        self.assertEqual(errors, [])


class TestHuggingFaceResource(unittest.TestCase):
    def test_valid_hf_resource(self) -> None:
        res = HuggingFaceResource(
            name="hf_model",
            type="model",
            repo_id="org/model",
            revision="abc123",
            dest="/workspace/input/hf_model",
        )
        errors = res.validate()
        self.assertEqual(errors, [])

    def test_invalid_type(self) -> None:
        res = HuggingFaceResource(
            name="hf_data",
            type="invalid",  # Not model or dataset
            repo_id="org/data",
            dest="/workspace/input/hf_data",
        )
        errors = res.validate()
        self.assertEqual(len(errors), 1)
        self.assertIn("type must be", errors[0])


class TestLoadResources(unittest.TestCase):
    def test_load_valid_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            (tmp_path / "template.txt").write_text("hello", encoding="utf-8")
            data = {
                "local": [
                    {"name": "data", "host_path": tmpdir, "mount_path": "/workspace/input/data"}
                ],
                "github": [
                    {"name": "lib", "repo": "https://github.com/a/b.git", "ref": "main", "dest": "/workspace/lib"}
                ],
                "huggingface": [],
                "items": [
                    {
                        "name": "tmpl",
                        "class": "template",
                        "source": "local",
                        "resource": "data",
                        "path": "template.txt",
                        "include_content": True,
                    }
                ],
            }
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                json.dump(data, f)
                f.flush()
                path = Path(f.name)

            try:
                config = load_resources(path)
                self.assertEqual(len(config.local), 1)
                self.assertEqual(len(config.github), 1)
                self.assertEqual(len(config.items), 1)
                self.assertEqual(config.local[0].name, "data")
            finally:
                path.unlink()

    def test_load_invalid_dest(self) -> None:
        data = {
            "local": [],
            "github": [
                {"name": "lib", "repo": "https://github.com/a/b.git", "dest": "/tmp/outside"}
            ],
            "huggingface": [],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            path = Path(f.name)

        try:
            with self.assertRaises(ValueError) as ctx:
                load_resources(path)
            self.assertIn("/workspace", str(ctx.exception))
        finally:
            path.unlink()


class TestBuildLocalBinds(unittest.TestCase):
    def test_empty_resources(self) -> None:
        binds = build_local_binds(None)
        self.assertEqual(binds, [])

    def test_single_bind(self) -> None:
        config = ResourceConfig(
            local=[LocalResource(name="data", host_path="/host/data", mount_path="/workspace/data", read_only=True)]
        )
        binds = build_local_binds(config)
        self.assertEqual(binds, ["/host/data:/workspace/data:ro"])

    def test_writable_bind(self) -> None:
        config = ResourceConfig(
            local=[LocalResource(name="out", host_path="/host/out", mount_path="/workspace/out", read_only=False)]
        )
        binds = build_local_binds(config)
        self.assertEqual(binds, ["/host/out:/workspace/out"])


class TestGitHubFetchCommands(unittest.TestCase):
    def test_shallow_clone_with_tag(self) -> None:
        config = ResourceConfig(
            github=[GitHubResource(name="lib", repo="https://github.com/a/b.git", ref="v1.0", dest="/workspace/lib")]
        )
        commands = get_github_fetch_commands(config)
        self.assertEqual(len(commands), 2)  # Clone + verify
        self.assertIn("--depth 1", commands[0]["command"])
        self.assertIn("--branch v1.0", commands[0]["command"])

    def test_full_sha_fetch(self) -> None:
        sha = "a" * 40
        config = ResourceConfig(
            github=[GitHubResource(name="lib", repo="https://github.com/a/b.git", ref=sha, dest="/workspace/lib")]
        )
        commands = get_github_fetch_commands(config)
        self.assertEqual(len(commands), 2)
        self.assertIn("--no-checkout", commands[0]["command"])
        self.assertIn(f"git checkout {sha}", commands[0]["command"])


class TestBuildResourcesContext(unittest.TestCase):
    def test_empty_resources(self) -> None:
        ctx = build_resources_context(None)
        self.assertFalse(ctx["has_resources"])

    def test_with_resources(self) -> None:
        config = ResourceConfig(
            local=[LocalResource(name="data", host_path="/host", mount_path="/workspace/data")],
            github=[GitHubResource(name="lib", repo="https://github.com/a/b.git", ref="main", dest="/workspace/lib")],
            items=[
                ResourceItem(
                    name="tmpl",
                    class_="template",
                    source="local",
                    resource="data",
                    path=".",
                )
            ],
        )
        ctx = build_resources_context(config)
        self.assertTrue(ctx["has_resources"])
        self.assertEqual(len(ctx["local_mounts"]), 1)
        self.assertEqual(len(ctx["github_resources"]), 1)
        self.assertEqual(len(ctx["items"]), 1)


if __name__ == "__main__":
    unittest.main()
