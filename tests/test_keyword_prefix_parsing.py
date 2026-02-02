"""Tests for _parse_keyword_prefix_response case-insensitive matching.

Regression test: the parser previously required exact case matching on keyword
prefixes, causing LLM responses with mixed casing (e.g. "Reasoning:" instead of
"REASONING:") to fail parsing.
"""

from ai_scientist.treesearch.parallel_agent import _parse_keyword_prefix_response


class TestParseKeywordPrefixResponse:
    """Verify keyword prefix parsing handles case variations."""

    def test_exact_case_match(self):
        response = "REASONING: The plots look correct\nSUCCESSFULLY_TESTED_DATASETS: dataset_a, dataset_b"
        name, desc = _parse_keyword_prefix_response(
            response, "REASONING:", "SUCCESSFULLY_TESTED_DATASETS:"
        )
        assert name == "The plots look correct"
        assert desc == "dataset_a, dataset_b"

    def test_lowercase_keywords(self):
        response = "reasoning: The plots look correct\nsuccessfully_tested_datasets: dataset_a"
        name, desc = _parse_keyword_prefix_response(
            response, "REASONING:", "SUCCESSFULLY_TESTED_DATASETS:"
        )
        assert name == "The plots look correct"
        assert desc == "dataset_a"

    def test_mixed_case_keywords(self):
        response = "Reasoning: The plots look correct\nSuccessfully_Tested_Datasets: dataset_a"
        name, desc = _parse_keyword_prefix_response(
            response, "REASONING:", "SUCCESSFULLY_TESTED_DATASETS:"
        )
        assert name == "The plots look correct"
        assert desc == "dataset_a"

    def test_missing_keyword_returns_none(self):
        response = "Some unrelated text\nNo keywords here"
        name, desc = _parse_keyword_prefix_response(
            response, "REASONING:", "SUCCESSFULLY_TESTED_DATASETS:"
        )
        assert name is None
        assert desc is None

    def test_multiline_description(self):
        response = (
            "REASONING: The plots are valid\n"
            "SUCCESSFULLY_TESTED_DATASETS: dataset_a\n"
            "and also dataset_b was tested"
        )
        name, desc = _parse_keyword_prefix_response(
            response, "REASONING:", "SUCCESSFULLY_TESTED_DATASETS:"
        )
        assert name == "The plots are valid"
        assert "dataset_a" in desc
        assert "dataset_b" in desc

    def test_empty_value(self):
        response = "REASONING: some reasoning\nSUCCESSFULLY_TESTED_DATASETS:"
        name, desc = _parse_keyword_prefix_response(
            response, "REASONING:", "SUCCESSFULLY_TESTED_DATASETS:"
        )
        assert name == "some reasoning"
        assert desc == ""
