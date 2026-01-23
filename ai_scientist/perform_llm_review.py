"""LLM paper review - backward compatibility wrapper.

This module re-exports functionality from the new ai_scientist.review module
for backward compatibility.

New code should import directly from ai_scientist.review:
    from ai_scientist.review import perform_review, load_paper
"""

# Re-export from new modules
from ai_scientist.review import (
    perform_review,
    get_review_fewshot_examples,
    get_meta_review,
    load_paper,
    load_review,
    reviewer_system_prompt_base,
    reviewer_system_prompt_neg,
    reviewer_system_prompt_pos,
    neurips_form,
)

__all__ = [
    "perform_review",
    "get_review_fewshot_examples",
    "get_meta_review",
    "load_paper",
    "load_review",
    "reviewer_system_prompt_base",
    "reviewer_system_prompt_neg",
    "reviewer_system_prompt_pos",
    "neurips_form",
]
