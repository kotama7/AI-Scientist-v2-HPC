"""Paper writeup - backward compatibility wrapper.

This module re-exports functionality from the new ai_scientist.output module
and provides the CLI interface for backward compatibility.

New code should import directly from ai_scientist.output:
    from ai_scientist.output import perform_writeup, gather_citations
"""

import argparse

# Re-export from new modules
from ai_scientist.output import (
    perform_writeup,
    load_idea_text,
    load_exp_summaries,
    filter_experiment_summaries,
    gather_citations,
    compile_latex,
    extract_latex_snippet,
    detect_pages_before_impact,
    remove_accents_and_clean,
    get_citation_addition,
)
from ai_scientist.llm import AVAILABLE_LLMS

__all__ = [
    "perform_writeup",
    "load_idea_text",
    "load_exp_summaries",
    "filter_experiment_summaries",
    "gather_citations",
    "compile_latex",
    "extract_latex_snippet",
    "detect_pages_before_impact",
    "remove_accents_and_clean",
    "get_citation_addition",
]


if __name__ == "__main__":
    import traceback

    parser = argparse.ArgumentParser(description="Perform writeup for a project")
    parser.add_argument("--folder", type=str, help="Project folder", required=True)
    parser.add_argument("--no-writing", action="store_true", help="Only generate")
    parser.add_argument("--num-cite-rounds", type=int, default=20)
    parser.add_argument(
        "--model",
        type=str,
        default="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
        choices=AVAILABLE_LLMS,
        help="Model to use for citation collection (small model).",
    )
    parser.add_argument(
        "--big-model",
        type=str,
        default="o1-2024-12-17",
        choices=AVAILABLE_LLMS,
        help="Model to use for final writeup (big model).",
    )
    parser.add_argument(
        "--writeup-reflections",
        type=int,
        default=3,
        help="Number of reflection steps for the final LaTeX writeup.",
    )
    parser.add_argument(
        "--page-limit",
        type=int,
        default=8,
        help="Target page limit for the main paper (excluding references, impact statement, etc.); use 0 to disable.",
    )
    args = parser.parse_args()

    try:
        success = perform_writeup(
            base_folder=args.folder,
            no_writing=args.no_writing,
            num_cite_rounds=args.num_cite_rounds,
            small_model=args.model,
            big_model=args.big_model,
            n_writeup_reflections=args.writeup_reflections,
            page_limit=args.page_limit if args.page_limit > 0 else None,
        )
        if not success:
            print("Writeup process did not complete successfully.")
    except Exception:
        print("EXCEPTION in main:")
        print(traceback.format_exc())
