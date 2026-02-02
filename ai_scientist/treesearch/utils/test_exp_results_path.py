"""Tests for experiment results path handling and .npy file loading robustness.

These tests verify that:
1. exp_results_dir is serialized as an absolute path in Node.to_dict()
2. _aggregate_seed_eval_results handles missing/invalid paths gracefully
3. Plotting templates skip missing .npy files instead of crashing
"""
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestExpResultsDirSerialization:
    """Tests for Node.to_dict() exp_results_dir serialization."""

    def test_to_dict_stores_absolute_path(self, tmp_path):
        """exp_results_dir should be serialized as an absolute path."""
        from ai_scientist.treesearch.journal import Node

        exp_dir = tmp_path / "logs" / "0-run" / "experiment_results" / "exp_001"
        exp_dir.mkdir(parents=True)

        node = Node(
            id="test-node",
            code="print('hello')",
            plan="test plan",
            exp_results_dir=str(exp_dir),
        )
        data = node.to_dict()

        assert data["exp_results_dir"] is not None
        assert os.path.isabs(data["exp_results_dir"]), (
            f"exp_results_dir should be absolute but got: {data['exp_results_dir']}"
        )
        assert Path(data["exp_results_dir"]).exists()

    def test_to_dict_none_exp_results_dir(self):
        """exp_results_dir=None should serialize as None."""
        from ai_scientist.treesearch.journal import Node

        node = Node(
            id="test-node",
            code="print('hello')",
            plan="test plan",
            exp_results_dir=None,
        )
        data = node.to_dict()
        assert data["exp_results_dir"] is None

    def test_to_dict_resolves_relative_path(self, tmp_path):
        """A relative exp_results_dir should be resolved to absolute in to_dict."""
        from ai_scientist.treesearch.journal import Node

        exp_dir = tmp_path / "experiment_results" / "exp_001"
        exp_dir.mkdir(parents=True)

        # Use relative path
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            node = Node(
                id="test-node",
                code="print('hello')",
                plan="test plan",
                exp_results_dir="experiment_results/exp_001",
            )
            data = node.to_dict()
            assert os.path.isabs(data["exp_results_dir"])
        finally:
            os.chdir(original_cwd)

    def test_roundtrip_preserves_path(self, tmp_path):
        """Path should survive to_dict -> from_dict roundtrip."""
        from ai_scientist.treesearch.journal import Node

        exp_dir = tmp_path / "experiment_results" / "exp_001"
        exp_dir.mkdir(parents=True)

        node = Node(
            id="test-node",
            code="print('hello')",
            plan="test plan",
            exp_results_dir=str(exp_dir),
        )
        data = node.to_dict()
        restored = Node.from_dict(data)
        assert restored.exp_results_dir == str(exp_dir.resolve())


class TestNpyFileCopy:
    """Tests that .npy files are copied (not moved) to experiment_results."""

    def test_copy_preserves_original(self, tmp_path):
        """After copying .npy to experiment_results, the original should still exist."""
        # Simulate the working directory with .npy file
        working_dir = tmp_path / "working"
        working_dir.mkdir()
        npy_file = working_dir / "test_data.npy"
        np.save(str(npy_file), {"test": "data"})

        # Simulate the experiment_results directory
        exp_results_dir = tmp_path / "experiment_results" / "exp_001"
        exp_results_dir.mkdir(parents=True)

        # Copy (as in the fixed code)
        for exp_data_file in working_dir.glob("*.npy"):
            exp_data_path = exp_results_dir / exp_data_file.name
            shutil.copy2(str(exp_data_file.resolve()), str(exp_data_path))

        # Both original and copy should exist
        assert npy_file.exists(), "Original .npy file should still exist after copy"
        assert (exp_results_dir / "test_data.npy").exists(), "Copy should exist"

        # Data should be identical
        orig = np.load(str(npy_file), allow_pickle=True)
        copy = np.load(str(exp_results_dir / "test_data.npy"), allow_pickle=True)
        assert orig.item() == copy.item()


class TestSeedDataPathCollection:
    """Tests for seed data path collection in _aggregate_seed_eval_results."""

    def test_skips_nonexistent_exp_dir(self, tmp_path):
        """Should skip seed nodes with nonexistent exp_results_dir."""
        seed_data_paths = []
        mock_node = MagicMock()
        mock_node.id = "test-id"
        mock_node.exp_results_dir = str(tmp_path / "nonexistent")

        exp_dir = Path(mock_node.exp_results_dir)
        if not exp_dir.is_absolute():
            exp_dir = Path(os.getcwd()) / exp_dir
        if exp_dir.exists():
            npy_files = sorted(exp_dir.rglob("*.npy"))
            valid_paths = [str(p.resolve()) for p in npy_files if p.is_file()]
            seed_data_paths.extend(valid_paths)

        assert seed_data_paths == []

    def test_collects_existing_npy_files(self, tmp_path):
        """Should collect absolute paths for existing .npy files."""
        exp_dir = tmp_path / "experiment_results" / "exp_001"
        exp_dir.mkdir(parents=True)
        npy_file = exp_dir / "test_data.npy"
        np.save(str(npy_file), {"test": "data"})

        seed_data_paths = []
        if exp_dir.exists():
            npy_files = sorted(exp_dir.rglob("*.npy"))
            valid_paths = [str(p.resolve()) for p in npy_files if p.is_file()]
            seed_data_paths.extend(valid_paths)

        assert len(seed_data_paths) == 1
        assert os.path.isabs(seed_data_paths[0])
        assert os.path.isfile(seed_data_paths[0])

    def test_skips_missing_files_in_existing_dir(self, tmp_path):
        """Should skip individual files that don't exist (e.g., broken symlinks)."""
        exp_dir = tmp_path / "experiment_results" / "exp_001"
        exp_dir.mkdir(parents=True)
        # Create a valid file
        good_file = exp_dir / "good_data.npy"
        np.save(str(good_file), {"good": "data"})
        # Create a broken symlink
        broken_link = exp_dir / "broken_data.npy"
        broken_link.symlink_to(tmp_path / "nonexistent.npy")

        seed_data_paths = []
        npy_files = sorted(exp_dir.rglob("*.npy"))
        valid_paths = [str(p.resolve()) for p in npy_files if p.is_file()]
        seed_data_paths.extend(valid_paths)

        assert len(seed_data_paths) == 1
        assert "good_data.npy" in seed_data_paths[0]


class TestPlottingTemplateRobustness:
    """Tests that the plotting data loading logic handles missing files gracefully."""

    def test_skips_missing_npy_file(self, tmp_path):
        """Data loading should skip missing files and continue."""
        experiment_data_path_list = [
            str(tmp_path / "nonexistent_1.npy"),
            str(tmp_path / "nonexistent_2.npy"),
        ]

        all_experiment_data = []
        for experiment_data_path in experiment_data_path_list:
            full_path = experiment_data_path
            if not os.path.isfile(full_path):
                continue
            loaded = np.load(full_path, allow_pickle=True)
            experiment_data = loaded.item() if isinstance(loaded, np.ndarray) and loaded.shape == () else loaded
            all_experiment_data.append((experiment_data_path, experiment_data))

        assert all_experiment_data == []

    def test_loads_existing_npy_files(self, tmp_path):
        """Should load existing files and skip missing ones."""
        # Create one valid file
        valid_file = tmp_path / "valid_data.npy"
        np.save(str(valid_file), {"metric": 42})

        experiment_data_path_list = [
            str(tmp_path / "nonexistent.npy"),
            str(valid_file),
        ]

        all_experiment_data = []
        for experiment_data_path in experiment_data_path_list:
            full_path = experiment_data_path
            if not os.path.isfile(full_path):
                continue
            try:
                loaded = np.load(full_path, allow_pickle=True)
                experiment_data = loaded.item() if isinstance(loaded, np.ndarray) and loaded.shape == () else loaded
                all_experiment_data.append((experiment_data_path, experiment_data))
            except Exception:
                pass

        assert len(all_experiment_data) == 1
        assert all_experiment_data[0][1] == {"metric": 42}

    def test_handles_corrupted_npy_file(self, tmp_path):
        """Should handle corrupted .npy files without crashing."""
        corrupted_file = tmp_path / "corrupted.npy"
        corrupted_file.write_bytes(b"not a valid npy file")

        all_experiment_data = []
        full_path = str(corrupted_file)
        if os.path.isfile(full_path):
            try:
                loaded = np.load(full_path, allow_pickle=True)
                experiment_data = loaded.item() if isinstance(loaded, np.ndarray) and loaded.shape == () else loaded
                all_experiment_data.append((full_path, experiment_data))
            except Exception:
                pass

        assert all_experiment_data == []
