"""Review module for AI Scientist.

This module provides functionality for reviewing papers using both
LLM (text-based) and VLM (vision-based) approaches.
"""

from ai_scientist.review.llm_review import (
    perform_review,
    get_review_fewshot_examples,
    get_meta_review,
    reviewer_system_prompt_base,
    reviewer_system_prompt_neg,
    reviewer_system_prompt_pos,
    neurips_form,
)
from ai_scientist.review.vlm_review import (
    perform_imgs_cap_ref_review,
    perform_imgs_cap_ref_review_selection,
    detect_duplicate_figures,
    generate_vlm_img_review,
    generate_vlm_img_cap_ref_review,
    generate_vlm_img_selection_review,
    extract_figure_screenshots,
    extract_abstract,
)
from ai_scientist.review.pdf_utils import (
    load_paper,
    load_review,
)

__all__ = [
    # LLM Review
    "perform_review",
    "get_review_fewshot_examples",
    "get_meta_review",
    "reviewer_system_prompt_base",
    "reviewer_system_prompt_neg",
    "reviewer_system_prompt_pos",
    "neurips_form",
    # VLM Review
    "perform_imgs_cap_ref_review",
    "perform_imgs_cap_ref_review_selection",
    "detect_duplicate_figures",
    "generate_vlm_img_review",
    "generate_vlm_img_cap_ref_review",
    "generate_vlm_img_selection_review",
    "extract_figure_screenshots",
    "extract_abstract",
    # PDF Utilities
    "load_paper",
    "load_review",
]
