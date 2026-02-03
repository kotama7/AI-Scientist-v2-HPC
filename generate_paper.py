import argparse
import json
import os
import os.path as osp
import re
import shutil
from pathlib import Path

from ai_scientist.llm import create_client
from ai_scientist.perform_plotting import aggregate_plots
from ai_scientist.perform_writeup import perform_writeup, gather_citations
from ai_scientist.perform_llm_review import (
    perform_review,
    load_paper,
    reviewer_system_prompt_base,
    reviewer_system_prompt_neg,
    reviewer_system_prompt_pos,
)
from ai_scientist.perform_vlm_review import perform_imgs_cap_ref_review
from ai_scientist.utils.token_tracker import token_tracker


def save_token_tracker(exp_dir: Path) -> None:
    with open(exp_dir / "token_tracker.json", "w") as f:
        json.dump(token_tracker.get_summary(), f)
    with open(exp_dir / "token_tracker_interactions.json", "w") as f:
        json.dump(token_tracker.get_interactions(), f)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate plots, writeup, and reviews for an existing experiment folder."
    )
    parser.add_argument(
        "--experiment-dir",
        required=True,
        help="Path to the experiment folder (e.g., experiments/2025-01-01_foo_attempt_0).",
    )
    parser.add_argument(
        "--writeup-type",
        type=str,
        default="normal",
        choices=["normal", "auto"],
        help="Type of writeup to generate (normal=page-limited, auto=no page limit).",
    )
    parser.add_argument(
        "--writeup-page-limit",
        type=int,
        default=8,
        help="Page limit for normal writeups; use 0 to disable. Ignored for auto.",
    )
    parser.add_argument(
        "--writeup-retries",
        type=int,
        default=3,
        help="Number of writeup attempts to try.",
    )
    parser.add_argument(
        "--writeup-reflections",
        type=int,
        default=3,
        help="Number of reflection steps to run during the writeup stage.",
    )
    parser.add_argument(
        "--model-agg-plots",
        type=str,
        default="o3-mini-2025-01-31",
        help="Model to use for plot aggregation.",
    )
    parser.add_argument(
        "--model-agg-plots-ref",
        type=int,
        default=5,
        help="Number of reflections to use for plot aggregation.",
    )
    parser.add_argument(
        "--model-writeup",
        type=str,
        default="o1-preview-2024-09-12",
        help="Model to use for writeup.",
    )
    parser.add_argument(
        "--model-citation",
        type=str,
        default="gpt-4o-2024-11-20",
        help="Model to use for citation gathering.",
    )
    parser.add_argument(
        "--num-cite-rounds",
        type=int,
        default=20,
        help="Number of citation rounds to perform.",
    )
    parser.add_argument(
        "--model-writeup-small",
        type=str,
        default="gpt-4o-2024-05-13",
        help="Smaller model to use for writeup.",
    )
    parser.add_argument(
        "--model-review",
        type=str,
        default="gpt-4o-2024-11-20",
        help="Model to use for review main text and captions.",
    )
    parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="If set, skip the plot aggregation process.",
    )
    parser.add_argument(
        "--skip-writeup",
        action="store_true",
        help="If set, skip the writeup process.",
    )
    parser.add_argument(
        "--skip-review",
        action="store_true",
        help="If set, skip the review process.",
    )
    parser.add_argument(
        "--review-bias",
        type=str,
        choices=["neg", "pos", "neutral"],
        default="neutral",
        help="Review bias mode: 'neg' (strict - reject if unsure), 'pos' (lenient - accept if unsure), 'neutral' (balanced - no bias on uncertainty).",
    )
    parser.add_argument(
        "--clean-experiment-results",
        action="store_true",
        help="If set, remove experiment_results after plot aggregation.",
    )
    return parser.parse_args()


def find_pdf_path_for_review(exp_dir: Path) -> Path | None:
    pdf_files = [f for f in os.listdir(exp_dir) if f.endswith(".pdf")]
    reflection_pdfs = [f for f in pdf_files if "reflection" in f]
    if not reflection_pdfs:
        return None
    final_pdfs = [f for f in reflection_pdfs if "final" in f.lower()]
    if final_pdfs:
        return exp_dir / final_pdfs[0]

    reflection_nums = []
    for fname in reflection_pdfs:
        match = re.search(r"reflection[_.]?(\d+)", fname)
        if match:
            reflection_nums.append((int(match.group(1)), fname))

    if reflection_nums:
        highest_reflection = max(reflection_nums, key=lambda x: x[0])
        return exp_dir / highest_reflection[1]

    return exp_dir / reflection_pdfs[0]


def ensure_experiment_results(exp_dir: Path) -> Path | None:
    direct_path = exp_dir / "experiment_results"
    if direct_path.exists():
        return direct_path

    logs_path = exp_dir / "logs/0-run/experiment_results"
    if logs_path.exists():
        shutil.copytree(logs_path, direct_path, dirs_exist_ok=True)
        return direct_path

    return None


def main() -> None:
    args = parse_arguments()
    repo_root = Path(__file__).resolve().parent
    os.environ["AI_SCIENTIST_ROOT"] = str(repo_root)

    exp_dir = Path(args.experiment_dir).expanduser().resolve()
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    exp_results_dir = ensure_experiment_results(exp_dir)
    if not exp_results_dir:
        print("Warning: experiment_results not found; plotting may fail.")

    if not args.skip_plot:
        aggregate_plots(
            base_folder=str(exp_dir),
            model=args.model_agg_plots,
            n_reflections=args.model_agg_plots_ref,
        )
        if args.clean_experiment_results and exp_results_dir and exp_results_dir.exists():
            shutil.rmtree(exp_results_dir)

    save_token_tracker(exp_dir)

    if not args.skip_writeup:
        writeup_success = False
        for attempt in range(args.writeup_retries):
            print(f"Writeup attempt {attempt + 1} of {args.writeup_retries}")
            page_limit = None
            if args.writeup_type == "normal":
                page_limit = args.writeup_page_limit if args.writeup_page_limit > 0 else None
            citations_text = gather_citations(
                str(exp_dir),
                num_cite_rounds=args.num_cite_rounds,
                small_model=args.model_citation,
            )
            writeup_success = perform_writeup(
                base_folder=str(exp_dir),
                small_model=args.model_writeup_small,
                big_model=args.model_writeup,
                page_limit=page_limit,
                n_writeup_reflections=args.writeup_reflections,
                citations_text=citations_text,
            )
            if writeup_success:
                break

        if not writeup_success:
            print("Writeup process did not complete successfully after all retries.")

    save_token_tracker(exp_dir)

    if not args.skip_review:
        pdf_path = find_pdf_path_for_review(exp_dir)
        if pdf_path and pdf_path.exists():
            print("Paper found at: ", pdf_path)
            paper_content = load_paper(str(pdf_path))
            client, client_model = create_client(args.model_review)

            # Select reviewer system prompt based on bias mode
            if args.review_bias == "neg":
                reviewer_prompt = reviewer_system_prompt_neg
                print("Using negative bias (strict) review mode.")
            elif args.review_bias == "pos":
                reviewer_prompt = reviewer_system_prompt_pos
                print("Using positive bias (lenient) review mode.")
            else:  # neutral
                reviewer_prompt = reviewer_system_prompt_base
                print("Using neutral (balanced) review mode.")

            review_text = perform_review(
                paper_content,
                client_model,
                client,
                reviewer_system_prompt=reviewer_prompt,
            )
            review_img_cap_ref = perform_imgs_cap_ref_review(
                client, client_model, str(pdf_path)
            )
            with open(exp_dir / "review_text.txt", "w") as f:
                f.write(json.dumps(review_text, indent=4))
            with open(exp_dir / "review_img_cap_ref.json", "w") as f:
                json.dump(review_img_cap_ref, f, indent=4)
            print("Paper review completed.")
        else:
            print("No paper PDF found for review; skipping review step.")


if __name__ == "__main__":
    main()
