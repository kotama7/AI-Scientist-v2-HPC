#!/usr/bin/env python3
"""Generate few-shot examples for paper review.

This script takes PDF papers and generates:
1. Extracted text (.txt)
2. Generated review (.json)

Usage:
    python generate_fewshot_examples.py \
        --pdf path/to/paper.pdf \
        --output-dir ai_scientist/fewshot_examples \
        --model gpt-4o-2024-11-20 \
        --custom-prompt "Review this HPC paper focusing on scalability and performance"
"""

import argparse
import json
import os
from pathlib import Path

from ai_scientist.llm import create_client
from ai_scientist.review import perform_review, load_paper, reviewer_system_prompt_neg


def generate_fewshot_example(
    pdf_path: str,
    output_dir: str,
    model: str = "gpt-4o-2024-11-20",
    custom_system_prompt: str = None,
    custom_review_prompt: str = None,
    num_reflections: int = 3,
) -> dict:
    """Generate a few-shot example from a PDF paper.

    Args:
        pdf_path: Path to the PDF paper.
        output_dir: Output directory for generated files.
        model: Model to use for review generation.
        custom_system_prompt: Custom system prompt for domain-specific review.
        custom_review_prompt: Additional instructions for the review.
        num_reflections: Number of reflection rounds for review quality.

    Returns:
        Dictionary with paths to generated files.
    """
    pdf_path = Path(pdf_path).resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get base name without extension
    base_name = pdf_path.stem

    # Step 1: Extract text from PDF
    print(f"Extracting text from {pdf_path.name}...")
    paper_text = load_paper(str(pdf_path))

    txt_path = output_dir / f"{base_name}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(paper_text)
    print(f"✓ Saved text to {txt_path}")

    # Step 2: Generate review
    print(f"Generating review with {model}...")
    client, client_model = create_client(model)

    # Use custom system prompt if provided
    system_prompt = custom_system_prompt if custom_system_prompt else reviewer_system_prompt_neg

    # Add custom instructions to paper text if provided
    review_text = paper_text
    if custom_review_prompt:
        review_text = f"{custom_review_prompt}\n\n{paper_text}"

    review = perform_review(
        text=review_text,
        model=client_model,
        client=client,
        num_reflections=num_reflections,
        num_fs_examples=0,  # Don't use existing few-shot examples
        reviewer_system_prompt=system_prompt,
    )

    # Step 3: Save review as JSON
    json_path = output_dir / f"{base_name}.json"
    review_wrapper = {"review": json.dumps(review, indent=4)}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(review_wrapper, f, indent=4)
    print(f"✓ Saved review to {json_path}")

    # Step 4: Copy PDF to output directory if not already there
    output_pdf = output_dir / f"{base_name}.pdf"
    if pdf_path != output_pdf:
        import shutil
        shutil.copy2(pdf_path, output_pdf)
        print(f"✓ Copied PDF to {output_pdf}")

    print("\n" + "="*60)
    print("Few-shot example generated successfully!")
    print("="*60)
    print(f"Files created:")
    print(f"  - {output_pdf}")
    print(f"  - {txt_path}")
    print(f"  - {json_path}")
    print("\nReview Summary:")
    print(f"  Overall Score: {review.get('Overall', 'N/A')}/10")
    print(f"  Decision: {review.get('Decision', 'N/A')}")
    print(f"  Soundness: {review.get('Soundness', 'N/A')}/4")
    print(f"  Significance: {review.get('Significance', 'N/A')}/4")

    return {
        "pdf": str(output_pdf),
        "txt": str(txt_path),
        "json": str(json_path),
        "review": review,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate few-shot examples for paper review from PDF papers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate from a single HPC paper
  python generate_fewshot_examples.py \\
      --pdf papers/himeno_benchmark.pdf \\
      --output-dir ai_scientist/fewshot_examples \\
      --model gpt-4o-2024-11-20

  # Generate with custom HPC-focused prompt
  python generate_fewshot_examples.py \\
      --pdf papers/mpi_performance.pdf \\
      --output-dir ai_scientist/fewshot_examples \\
      --custom-prompt "Review this HPC paper with focus on parallel scalability, performance analysis, and reproducibility."

  # Generate from multiple papers
  for pdf in papers/*.pdf; do
      python generate_fewshot_examples.py --pdf "$pdf" --output-dir fewshot_hpc
  done
        """,
    )
    parser.add_argument(
        "--pdf",
        type=str,
        required=True,
        help="Path to the PDF paper to generate few-shot example from.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="ai_scientist/fewshot_examples",
        help="Output directory for generated files (default: ai_scientist/fewshot_examples).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-2024-11-20",
        help="Model to use for review generation (default: gpt-4o-2024-11-20).",
    )
    parser.add_argument(
        "--custom-prompt",
        type=str,
        default=None,
        help="Custom instructions for domain-specific review (e.g., 'Focus on HPC scalability and performance').",
    )
    parser.add_argument(
        "--custom-system-prompt",
        type=str,
        default=None,
        help="Custom system prompt to replace default reviewer prompt (for domain experts).",
    )
    parser.add_argument(
        "--num-reflections",
        type=int,
        default=3,
        help="Number of reflection rounds for review quality (default: 3).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only extract text without generating review.",
    )

    args = parser.parse_args()

    if args.dry_run:
        # Only extract text
        pdf_path = Path(args.pdf).resolve()
        print(f"Extracting text from {pdf_path.name}...")
        paper_text = load_paper(str(pdf_path))
        base_name = pdf_path.stem
        txt_path = Path(args.output_dir) / f"{base_name}.txt"
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(paper_text)
        print(f"✓ Saved text to {txt_path}")
        print(f"Text length: {len(paper_text)} characters")
    else:
        generate_fewshot_example(
            pdf_path=args.pdf,
            output_dir=args.output_dir,
            model=args.model,
            custom_system_prompt=args.custom_system_prompt,
            custom_review_prompt=args.custom_prompt,
            num_reflections=args.num_reflections,
        )


if __name__ == "__main__":
    main()
