"""Tests for experiment output filename utilities."""

import pytest
from ai_scientist.treesearch.utils.file_utils import (
    sanitize_experiment_name,
    get_experiment_output_filename,
    get_experiment_output_pattern,
    DEFAULT_EXPERIMENT_OUTPUT_FILENAME,
)


class TestSanitizeExperimentName:
    """Tests for sanitize_experiment_name function."""

    def test_removes_timestamp_prefix(self):
        """Should remove leading timestamp prefix from experiment name."""
        name = "2026-01-28_17-46-11_stability_oriented_autotuning_v2_attempt_0"
        result = sanitize_experiment_name(name)
        assert result == "stability_oriented_autotuning_v2"

    def test_removes_attempt_suffix(self):
        """Should remove trailing _attempt_N suffix."""
        name = "stability_oriented_autotuning_v2_attempt_0"
        result = sanitize_experiment_name(name)
        assert result == "stability_oriented_autotuning_v2"

    def test_removes_both_timestamp_and_attempt(self):
        """Should remove both timestamp prefix and attempt suffix."""
        name = "2026-01-28_10-00-00_matrix_multiplication_opt_attempt_3"
        result = sanitize_experiment_name(name)
        assert result == "matrix_multiplication_opt"

    def test_replaces_special_characters(self):
        """Should replace non-alphanumeric characters with underscores."""
        name = "test@experiment#with$special%chars"
        result = sanitize_experiment_name(name)
        assert "@" not in result
        assert "#" not in result
        assert "$" not in result
        assert "%" not in result

    def test_collapses_multiple_underscores(self):
        """Should collapse multiple consecutive underscores."""
        name = "test___multiple___underscores"
        result = sanitize_experiment_name(name)
        assert "___" not in result
        assert "__" not in result

    def test_lowercases_result(self):
        """Should lowercase the result."""
        name = "TestExperiment"
        result = sanitize_experiment_name(name)
        assert result == result.lower()

    def test_truncates_long_names(self):
        """Should truncate names exceeding max_length."""
        name = "a" * 100
        result = sanitize_experiment_name(name, max_length=50)
        assert len(result) <= 50

    def test_handles_empty_string(self):
        """Should return 'experiment' for empty string."""
        result = sanitize_experiment_name("")
        assert result == "experiment"

    def test_handles_none(self):
        """Should return 'experiment' for None input."""
        result = sanitize_experiment_name(None)
        assert result == "experiment"

    def test_strips_leading_trailing_underscores(self):
        """Should strip leading and trailing underscores."""
        name = "___test_name___"
        result = sanitize_experiment_name(name)
        assert not result.startswith("_")
        assert not result.endswith("_")

    def test_realistic_experiment_name(self):
        """Test with realistic experiment directory name."""
        name = "2026-01-28_17-46-11_stability_oriented_autotuning_v2_attempt_0"
        result = sanitize_experiment_name(name)
        # Should be clean, lowercase, and descriptive
        assert result == "stability_oriented_autotuning_v2"
        assert result.islower() or "_" in result  # underscore-separated lowercase


class TestGetExperimentOutputFilename:
    """Tests for get_experiment_output_filename function."""

    def test_generates_filename_from_experiment_name(self):
        """Should generate filename from experiment name."""
        name = "2026-01-28_17-46-11_stability_autotuning_attempt_0"
        result = get_experiment_output_filename(name)
        assert result == "stability_autotuning_data.npy"

    def test_returns_default_for_none(self):
        """Should return default filename for None input."""
        result = get_experiment_output_filename(None)
        assert result == DEFAULT_EXPERIMENT_OUTPUT_FILENAME

    def test_returns_default_for_empty_string(self):
        """Should return default filename for empty string."""
        result = get_experiment_output_filename("")
        assert result == DEFAULT_EXPERIMENT_OUTPUT_FILENAME

    def test_filename_ends_with_data_npy(self):
        """Output filename should end with _data.npy."""
        result = get_experiment_output_filename("test_experiment")
        assert result.endswith("_data.npy")

    def test_realistic_experiment_name(self):
        """Test with realistic experiment directory name."""
        name = "2026-01-28_17-46-11_matrix_multiplication_optimization_attempt_0"
        result = get_experiment_output_filename(name)
        assert result == "matrix_multiplication_optimization_data.npy"


class TestGetExperimentOutputPattern:
    """Tests for get_experiment_output_pattern function."""

    def test_returns_glob_pattern(self):
        """Should return a glob pattern for matching experiment output files."""
        pattern = get_experiment_output_pattern()
        assert "*" in pattern
        assert pattern.endswith("_data.npy")

    def test_pattern_matches_expected_files(self):
        """Pattern should be '*_data.npy'."""
        pattern = get_experiment_output_pattern()
        assert pattern == "*_data.npy"


class TestDefaultFilename:
    """Tests for DEFAULT_EXPERIMENT_OUTPUT_FILENAME constant."""

    def test_default_filename_value(self):
        """Default filename should be experiment_data.npy for backwards compatibility."""
        assert DEFAULT_EXPERIMENT_OUTPUT_FILENAME == "experiment_data.npy"
