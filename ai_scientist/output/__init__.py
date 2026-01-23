"""Output module for AI Scientist.

This module provides functionality for generating plots, LaTeX writeups,
and managing citations.
"""

from ai_scientist.output.plotting import (
    aggregate_plots,
    extract_code_snippet,
    run_aggregator_script,
    build_aggregator_prompt,
)
from ai_scientist.output.writeup import (
    perform_writeup,
    load_idea_text,
    load_exp_summaries,
    filter_experiment_summaries,
)
from ai_scientist.output.latex_utils import (
    compile_latex,
    extract_latex_snippet,
    detect_pages_before_impact,
)
from ai_scientist.output.citation import (
    gather_citations,
    get_citation_addition,
    remove_accents_and_clean,
    CITATION_SYSTEM_MSG_TEMPLATE,
    CITATION_FIRST_PROMPT_TEMPLATE,
    CITATION_SECOND_PROMPT_TEMPLATE,
)

__all__ = [
    # Plotting
    "aggregate_plots",
    "extract_code_snippet",
    "run_aggregator_script",
    "build_aggregator_prompt",
    # Writeup
    "perform_writeup",
    "load_idea_text",
    "load_exp_summaries",
    "filter_experiment_summaries",
    # LaTeX utilities
    "compile_latex",
    "extract_latex_snippet",
    "detect_pages_before_impact",
    # Citation
    "gather_citations",
    "get_citation_addition",
    "remove_accents_and_clean",
]
