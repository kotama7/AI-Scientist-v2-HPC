"""VLM paper review - backward compatibility wrapper.

This module re-exports functionality from the new ai_scientist.review module
for backward compatibility.

New code should import directly from ai_scientist.review:
    from ai_scientist.review import perform_imgs_cap_ref_review, generate_vlm_img_review
"""

# Re-export from new modules
from ai_scientist.review import (
    perform_imgs_cap_ref_review,
    perform_imgs_cap_ref_review_selection,
    detect_duplicate_figures,
    generate_vlm_img_review,
    generate_vlm_img_cap_ref_review,
    generate_vlm_img_selection_review,
    extract_figure_screenshots,
    extract_abstract,
)

__all__ = [
    "perform_imgs_cap_ref_review",
    "perform_imgs_cap_ref_review_selection",
    "detect_duplicate_figures",
    "generate_vlm_img_review",
    "generate_vlm_img_cap_ref_review",
    "generate_vlm_img_selection_review",
    "extract_figure_screenshots",
    "extract_abstract",
]
