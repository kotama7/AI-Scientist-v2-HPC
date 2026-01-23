"""Utilities for the treesearch module.

This module provides various utility functions for file operations,
parsing, artifact management, and more.
"""

import logging

logger = logging.getLogger("ai-scientist")

# Re-export from submodules (order matters to avoid circular imports)
# file_ops has no dependencies on other treesearch modules
from ai_scientist.treesearch.utils.file_ops import (
    copytree,
    clean_up_dataset,
    extract_archives,
    preproc_data,
)
from ai_scientist.treesearch.utils.parsing import (
    normalize_language,
    strip_json_wrappers,
    parse_json_object,
    normalize_phase0_plan,
)
from ai_scientist.treesearch.utils.file_utils import (
    read_text,
    summarize_file,
    find_previous_run_dir,
    summarize_phase1_steps,
    extract_error_lines,
    summarize_phase_logs,
    summarize_journal_outputs,
)
from ai_scientist.treesearch.utils.artifacts import (
    resolve_run_root,
    copy_artifact,
    format_prompt_log_name,
    render_prompt_for_log,
    write_prompt_log,
    save_phase_execution_artifacts,
)
