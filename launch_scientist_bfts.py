import os.path as osp
import json
import argparse
import shutil
import torch
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from ai_scientist.llm import create_client
from ai_scientist.perform_plotting import aggregate_plots
from ai_scientist.perform_writeup import perform_writeup, gather_citations
from ai_scientist.perform_icbinb_writeup import (
    perform_writeup as perform_icbinb_writeup,
    gather_citations as gather_icbinb_citations,
)
from ai_scientist.perform_llm_review import perform_review, load_paper
from ai_scientist.perform_vlm_review import perform_imgs_cap_ref_review
from ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager import (
    perform_experiments_bfts,
)
from ai_scientist.treesearch.bfts_utils import (
    idea_to_markdown,
    edit_bfts_config_file,
)
from ai_scientist.utils.token_tracker import token_tracker


def save_token_tracker(idea_dir):
    with open(osp.join(idea_dir, "token_tracker.json"), "w") as f:
        json.dump(token_tracker.get_summary(), f)
    with open(osp.join(idea_dir, "token_tracker_interactions.json"), "w") as f:
        json.dump(token_tracker.get_interactions(), f)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run AI scientist experiments")
    parser.add_argument(
        "--writeup-type",
        type=str,
        default="icbinb",
        choices=["normal", "icbinb"],
        help="Type of writeup to generate (normal=8 page, icbinb=4 page)",
    )
    parser.add_argument(
        "--load_ideas",
        type=str,
        default="ideas/i_cant_believe_its_not_better.json",
        help="Path to a JSON file containing pregenerated ideas",
    )
    parser.add_argument(
        "--idea_idx",
        type=int,
        default=0,
        help="Index of the idea to run",
    )
    parser.add_argument(
        "--additional-information",
        type=str,
        default=None,
        help="Path to a text file with supplementary information to append to the idea",
    )
    parser.add_argument(
        "--writeup-retries",
        type=int,
        default=3,
        help="Number of writeup attempts to try",
    )
    parser.add_argument(
        "--writeup-reflections",
        type=int,
        default=3,
        help="Number of reflection steps to run during the writeup stage",
    )
    parser.add_argument(
        "--attempt_id",
        type=int,
        default=0,
        help="Attempt ID, used to distinguish same idea in different attempts in parallel runs",
    )
    parser.add_argument(
        "--model_agg_plots",
        type=str,
        default="o3-mini-2025-01-31",
        help="Model to use for plot aggregation",
    )
    parser.add_argument(
        "--model_agg_plots_ref",
        type=int,
        default=5,
        help="Number of reflections to use for plot aggregation",
    )
    parser.add_argument(
        "--model_writeup",
        type=str,
        default="o1-preview-2024-09-12",
        help="Model to use for writeup",
    )
    parser.add_argument(
        "--model_citation",
        type=str,
        default="gpt-4o-2024-11-20",
        help="Model to use for citation gathering",
    )
    parser.add_argument(
        "--num_cite_rounds",
        type=int,
        default=20,
        help="Number of citation rounds to perform",
    )
    parser.add_argument(
        "--model_writeup_small",
        type=str,
        default="gpt-4o-2024-05-13",
        help="Smaller model to use for writeup",
    )
    parser.add_argument(
        "--model_review",
        type=str,
        default="gpt-4o-2024-11-20",
        help="Model to use for review main text and captions",
    )
    parser.add_argument(
        "--skip_writeup",
        action="store_true",
        help="If set, skip the writeup process",
    )
    parser.add_argument(
        "--skip_review",
        action="store_true",
        help="If set, skip the review process",
    )
    parser.add_argument(
        "--phase_mode",
        type=str,
        default="split",
        choices=["split", "single"],
        help="Execution phase mode (split runs download/coding/compile/run phases, single keeps legacy behavior)",
    )
    parser.add_argument(
        "--singularity_image",
        type=str,
        default=None,
        help="Path to Singularity image for worker execution",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of parallel workers to launch (maps worker-i to GPU i).",
    )
    parser.add_argument(
        "--container_overlay",
        type=str,
        default=None,
        help="Writable overlay image/path to enable apt-get inside Singularity",
    )
    parser.add_argument(
        "--disable_writable_tmpfs",
        action="store_true",
        help="Disable --writable-tmpfs when starting container instances.",
    )
    parser.add_argument(
        "--per_worker_sif",
        type=lambda x: str(x).lower() in ("1", "true", "yes"),
        default=True,
        help="Whether to build per-worker Singularity images (default: true).",
    )
    parser.add_argument(
        "--keep_sandbox",
        type=lambda x: str(x).lower() in ("1", "true", "yes"),
        default=False,
        help="Keep worker sandbox directories after building worker SIFs.",
    )
    parser.add_argument(
        "--use_fakeroot",
        type=lambda x: str(x).lower() in ("1", "true", "yes"),
        default=True,
        help="Use --fakeroot for Singularity build/exec when preparing worker images.",
    )
    parser.add_argument(
        "--writable_mode",
        type=str,
        choices=["auto", "tmpfs", "overlay", "none"],
        default="auto",
        help="Writable mode for Phase 1 in Singularity (auto chooses tmpfs, fallback to overlay).",
    )
    parser.add_argument(
        "--phase1_max_steps",
        type=int,
        default=None,
        help="Maximum number of Phase 1 iterative installer steps.",
    )
    parser.add_argument(
        "--resources",
        type=str,
        default=None,
        help="Path to resources JSON/YAML file defining local mounts, GitHub repos, and HuggingFace models/datasets.",
    )
    return parser.parse_args()


def get_available_gpus(gpu_ids=None):
    if gpu_ids is not None:
        return [int(gpu_id) for gpu_id in gpu_ids.split(",")]
    return list(range(torch.cuda.device_count()))


def find_pdf_path_for_review(idea_dir):
    pdf_files = [f for f in os.listdir(idea_dir) if f.endswith(".pdf")]
    reflection_pdfs = [f for f in pdf_files if "reflection" in f]
    if reflection_pdfs:
        # First check if there's a final version
        final_pdfs = [f for f in reflection_pdfs if "final" in f.lower()]
        if final_pdfs:
            # Use the final version if available
            pdf_path = osp.join(idea_dir, final_pdfs[0])
        else:
            # Try to find numbered reflections
            reflection_nums = []
            for f in reflection_pdfs:
                match = re.search(r"reflection[_.]?(\d+)", f)
                if match:
                    reflection_nums.append((int(match.group(1)), f))

            if reflection_nums:
                # Get the file with the highest reflection number
                highest_reflection = max(reflection_nums, key=lambda x: x[0])
                pdf_path = osp.join(idea_dir, highest_reflection[1])
            else:
                # Fall back to the first reflection PDF if no numbers found
                pdf_path = osp.join(idea_dir, reflection_pdfs[0])
    return pdf_path


if __name__ == "__main__":
    args = parse_arguments()
    repo_root = Path(__file__).resolve().parent
    os.environ["AI_SCIENTIST_ROOT"] = str(repo_root)
    print(f"Set AI_SCIENTIST_ROOT to {os.environ['AI_SCIENTIST_ROOT']}")

    # Check available GPUs and adjust parallel processes if necessary
    available_gpus = get_available_gpus()
    print(f"Using GPUs: {available_gpus}")

    with open(args.load_ideas, "r") as f:
        ideas = json.load(f)
        print(f"Loaded {len(ideas)} pregenerated ideas from {args.load_ideas}")

    idea = ideas[args.idea_idx]
    if args.additional_information:
        additional_info_path = Path(args.additional_information)
        if not additional_info_path.is_file():
            raise FileNotFoundError(
                f"Additional information file {additional_info_path} not found"
            )
        additional_information = additional_info_path.read_text(encoding="utf-8")
        idea["Additional Information"] = additional_information
        print(f"Attached supplemental information from {additional_info_path}")

    date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    idea_dir = f"experiments/{date}_{idea['Name']}_attempt_{args.attempt_id}"
    idea_dir_path = (repo_root / idea_dir).resolve()
    idea_dir = str(idea_dir_path)
    print(f"Results will be saved in {idea_dir}")
    idea_dir_path.mkdir(parents=True, exist_ok=True)

    idea_path_md = idea_dir_path / "idea.md"
    idea_to_markdown(idea, str(idea_path_md), load_code=None)

    idea_path_json = idea_dir_path / "idea.json"
    with open(idea_path_json, "w") as f:
        json.dump(ideas[args.idea_idx], f, indent=4)

    config_path = repo_root / "bfts_config.yaml"
    idea_config_path = edit_bfts_config_file(
        str(config_path),
        str(idea_dir_path),
        str(idea_path_md),
        phase_mode=args.phase_mode,
        singularity_image=args.singularity_image,
        num_workers=args.num_workers,
        writable_tmpfs=not args.disable_writable_tmpfs,
        container_overlay=args.container_overlay,
        per_worker_sif=args.per_worker_sif,
        keep_sandbox=args.keep_sandbox,
        use_fakeroot=args.use_fakeroot,
        writable_mode=args.writable_mode,
        phase1_max_steps=args.phase1_max_steps,
        resources_path=args.resources,
    )

    perform_experiments_bfts(idea_config_path)
    experiment_results_dir = idea_dir_path / "logs/0-run/experiment_results"
    if experiment_results_dir.exists():
        shutil.copytree(
            experiment_results_dir,
            idea_dir_path / "experiment_results",
            dirs_exist_ok=True,
        )

    aggregate_plots(base_folder=idea_dir, model=args.model_agg_plots, n_reflections=args.model_agg_plots_ref)

    shutil.rmtree(idea_dir_path / "experiment_results")

    save_token_tracker(idea_dir)

    if not args.skip_writeup:
        writeup_success = False
        for attempt in range(args.writeup_retries):
            print(f"Writeup attempt {attempt+1} of {args.writeup_retries}")
            if args.writeup_type == "normal":
                citations_text = gather_citations(
                    idea_dir,
                    num_cite_rounds=args.num_cite_rounds,
                    small_model=args.model_citation,
                )
                writeup_success = perform_writeup(
                    base_folder=idea_dir,
                    small_model=args.model_writeup_small,
                    big_model=args.model_writeup,
                    page_limit=8,
                    n_writeup_reflections=args.writeup_reflections,
                    citations_text=citations_text,
                )
            else:
                citations_text = gather_icbinb_citations(
                    idea_dir,
                    num_cite_rounds=args.num_cite_rounds,
                    small_model=args.model_citation,
                )
                writeup_success = perform_icbinb_writeup(
                    base_folder=idea_dir,
                    small_model=args.model_writeup_small,
                    big_model=args.model_writeup,
                    page_limit=4,
                    n_writeup_reflections=args.writeup_reflections,
                    citations_text=citations_text,
                )
            if writeup_success:
                break

        if not writeup_success:
            print("Writeup process did not complete successfully after all retries.")

    save_token_tracker(idea_dir)

    if not args.skip_review and not args.skip_writeup:
        # Perform paper review if the paper exists
        pdf_path = find_pdf_path_for_review(idea_dir)
        if os.path.exists(pdf_path):
            print("Paper found at: ", pdf_path)
            paper_content = load_paper(pdf_path)
            client, client_model = create_client(args.model_review)
            review_text = perform_review(paper_content, client_model, client)
            review_img_cap_ref = perform_imgs_cap_ref_review(
                client, client_model, pdf_path
            )
            with open(idea_dir_path / "review_text.txt", "w") as f:
                f.write(json.dumps(review_text, indent=4))
            with open(idea_dir_path / "review_img_cap_ref.json", "w") as f:
                json.dump(review_img_cap_ref, f, indent=4)
            print("Paper review completed.")

    print("Start cleaning up processes")
    # Kill all mp and torch processes associated with this experiment
    import psutil
    import signal

    # Get the current process and all its children
    current_process = psutil.Process()
    children = current_process.children(recursive=True)

    # First try graceful termination
    for child in children:
        try:
            child.send_signal(signal.SIGTERM)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    # Wait briefly for processes to terminate
    gone, alive = psutil.wait_procs(children, timeout=3)

    # If any processes remain, force kill them
    for process in alive:
        try:
            process.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    # Additional cleanup: find any orphaned processes containing specific keywords
    keywords = ["python", "torch", "mp", "bfts", "experiment"]
    for proc in psutil.process_iter(["name", "cmdline"]):
        try:
            # Check both process name and command line arguments
            cmdline = " ".join(proc.cmdline()).lower()
            if any(keyword in cmdline for keyword in keywords):
                proc.send_signal(signal.SIGTERM)
                proc.wait(timeout=3)
                if proc.is_running():
                    proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
            continue

    # Finally, terminate the current process
    # current_process.send_signal(signal.SIGTERM)
    # try:
    #     current_process.wait(timeout=3)
    # except psutil.TimeoutExpired:
    #     current_process.kill()

    # exit the program
    sys.exit(0)
