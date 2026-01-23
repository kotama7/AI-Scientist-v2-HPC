"""LLM-based paper review functionality."""

import os
import json
import numpy as np

from ai_scientist.llm import (
    get_response_from_llm,
    get_batch_responses_from_llm,
    extract_json_between_markers,
)
from ai_scientist.prompt_loader import load_prompt
from ai_scientist.review.pdf_utils import load_paper, load_review

# Load prompt templates
REVIEW_BASE_PROMPT = load_prompt("review/llm/system_base").strip()
REVIEW_NEG_SUFFIX = load_prompt("review/llm/system_neg_suffix").strip()
REVIEW_POS_SUFFIX = load_prompt("review/llm/system_pos_suffix").strip()
template_instructions = load_prompt("review/llm/template_instructions")
neurips_form = load_prompt("review/llm/neurips_form") + "\n" + template_instructions
reviewer_reflection_prompt = load_prompt("review/llm/reflection")
meta_reviewer_system_prompt = load_prompt("review/llm/meta_reviewer_system")
ENSEMBLE_AGGREGATION_THOUGHT_TEMPLATE = load_prompt(
    "review/llm/ensemble_aggregation"
).strip()
REVIEW_PAPER_PROMPT_TEMPLATE = load_prompt("review/llm/paper_review")
FEWSHOT_INTRO_TEMPLATE = load_prompt("review/llm/fewshot_intro")
FEWSHOT_EXAMPLE_TEMPLATE = load_prompt("review/llm/fewshot_example")
META_REVIEW_ENTRY_TEMPLATE = load_prompt("review/llm/meta_review_entry")

# System prompts
reviewer_system_prompt_base = REVIEW_BASE_PROMPT
reviewer_system_prompt_neg = f"{reviewer_system_prompt_base} {REVIEW_NEG_SUFFIX}"
reviewer_system_prompt_pos = f"{reviewer_system_prompt_base} {REVIEW_POS_SUFFIX}"

# Paths for fewshot examples
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(dir_path)

fewshot_papers = [
    os.path.join(parent_dir, "fewshot_examples/132_automated_relational.pdf"),
    os.path.join(parent_dir, "fewshot_examples/attention.pdf"),
    os.path.join(parent_dir, "fewshot_examples/2_carpe_diem.pdf"),
]

fewshot_reviews = [
    os.path.join(parent_dir, "fewshot_examples/132_automated_relational.json"),
    os.path.join(parent_dir, "fewshot_examples/attention.json"),
    os.path.join(parent_dir, "fewshot_examples/2_carpe_diem.json"),
]


def perform_review(
    text,
    model,
    client,
    num_reflections=1,
    num_fs_examples=1,
    num_reviews_ensemble=1,
    temperature=0.75,
    msg_history=None,
    return_msg_history=False,
    reviewer_system_prompt=reviewer_system_prompt_neg,
    review_instruction_form=neurips_form,
):
    """Perform an LLM-based review of a paper.

    Args:
        text: Paper text content to review.
        model: LLM model to use.
        client: LLM client instance.
        num_reflections: Number of reflection rounds.
        num_fs_examples: Number of few-shot examples.
        num_reviews_ensemble: Number of reviews for ensemble.
        temperature: Sampling temperature.
        msg_history: Optional message history.
        return_msg_history: Whether to return message history.
        reviewer_system_prompt: System prompt for the reviewer.
        review_instruction_form: Review form template.

    Returns:
        Review dictionary, or tuple of (review, msg_history) if return_msg_history is True.
    """
    if num_fs_examples > 0:
        fs_prompt = get_review_fewshot_examples(num_fs_examples)
        base_prompt = review_instruction_form + fs_prompt
    else:
        base_prompt = review_instruction_form

    base_prompt += REVIEW_PAPER_PROMPT_TEMPLATE.format(paper_text=text)

    if num_reviews_ensemble > 1:
        llm_reviews, msg_histories = get_batch_responses_from_llm(
            base_prompt,
            model=model,
            client=client,
            system_message=reviewer_system_prompt,
            print_debug=False,
            msg_history=msg_history,
            temperature=0.75,
            n_responses=num_reviews_ensemble,
        )
        parsed_reviews = []
        for idx, rev in enumerate(llm_reviews):
            try:
                parsed_reviews.append(extract_json_between_markers(rev))
            except Exception as e:
                print(f"Ensemble review {idx} failed: {e}")
        parsed_reviews = [r for r in parsed_reviews if r is not None]
        review = get_meta_review(model, client, temperature, parsed_reviews)
        if review is None:
            review = parsed_reviews[0]
        for score, limits in [
            ("Originality", (1, 4)),
            ("Quality", (1, 4)),
            ("Clarity", (1, 4)),
            ("Significance", (1, 4)),
            ("Soundness", (1, 4)),
            ("Presentation", (1, 4)),
            ("Contribution", (1, 4)),
            ("Overall", (1, 10)),
            ("Confidence", (1, 5)),
        ]:
            scores = []
            for r in parsed_reviews:
                if score in r and limits[0] <= r[score] <= limits[1]:
                    scores.append(r[score])
            if scores:
                review[score] = int(round(np.mean(scores)))
        msg_history = msg_histories[0][:-1]
        aggregation_thought = ENSEMBLE_AGGREGATION_THOUGHT_TEMPLATE.format(
            num_reviews=num_reviews_ensemble,
            review_json=json.dumps(review),
        )
        msg_history += [
            {
                "role": "assistant",
                "content": aggregation_thought,
            }
        ]
    else:
        llm_review, msg_history = get_response_from_llm(
            base_prompt,
            model=model,
            client=client,
            system_message=reviewer_system_prompt,
            print_debug=False,
            msg_history=msg_history,
            temperature=temperature,
        )
        review = extract_json_between_markers(llm_review)

    if num_reflections > 1:
        for j in range(num_reflections - 1):
            text, msg_history = get_response_from_llm(
                reviewer_reflection_prompt,
                client=client,
                model=model,
                system_message=reviewer_system_prompt,
                msg_history=msg_history,
                temperature=temperature,
            )
            review = extract_json_between_markers(text)
            assert review is not None, "Failed to extract JSON from LLM output"
            if "I am done" in text:
                break

    if return_msg_history:
        return review, msg_history
    else:
        return review


def get_review_fewshot_examples(num_fs_examples: int = 1) -> str:
    """Get few-shot examples for paper review.

    Args:
        num_fs_examples: Number of examples to include.

    Returns:
        Formatted few-shot prompt string.
    """
    fewshot_prompt = FEWSHOT_INTRO_TEMPLATE
    for paper_path, review_path in zip(
        fewshot_papers[:num_fs_examples], fewshot_reviews[:num_fs_examples]
    ):
        txt_path = paper_path.replace(".pdf", ".txt")
        if os.path.exists(txt_path):
            with open(txt_path, "r") as f:
                paper_text = f.read()
        else:
            paper_text = load_paper(paper_path)
        review_text = load_review(review_path)
        fewshot_prompt += FEWSHOT_EXAMPLE_TEMPLATE.format(
            paper_text=paper_text,
            review_text=review_text,
        )
    return fewshot_prompt


def get_meta_review(model, client, temperature, reviews):
    """Generate a meta-review from multiple reviews.

    Args:
        model: LLM model to use.
        client: LLM client instance.
        temperature: Sampling temperature.
        reviews: List of review dictionaries.

    Returns:
        Meta-review dictionary, or None if generation fails.
    """
    review_text = ""
    for i, r in enumerate(reviews):
        review_text += META_REVIEW_ENTRY_TEMPLATE.format(
            index=i + 1,
            total=len(reviews),
            review_json=json.dumps(r),
        )
    base_prompt = neurips_form + review_text
    llm_review, _ = get_response_from_llm(
        base_prompt,
        model=model,
        client=client,
        system_message=meta_reviewer_system_prompt.format(reviewer_count=len(reviews)),
        print_debug=False,
        msg_history=None,
        temperature=temperature,
    )
    meta_review = extract_json_between_markers(llm_review)
    return meta_review
