import os.path as osp
import json
import argparse
import shutil
import torch
import os
import re
import sys
import importlib
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

from contextlib import contextmanager
import yaml

from ai_scientist import prompt_loader
from ai_scientist.llm import create_client, get_response_from_llm
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
from ai_scientist.persona import set_persona_role


def print_time():
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


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
        "--load_code",
        action="store_true",
        help="If set, load a Python file with same name as ideas file but .py extension",
    )
    parser.add_argument(
        "--idea_idx",
        type=int,
        default=0,
        help="Index of the idea to run",
    )
    parser.add_argument(
        "--add_dataset_ref",
        action="store_true",
        help="If set, add a HF dataset reference to the idea",
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
        "--container_runtime",
        type=str,
        default=None,
        choices=["singularity"],
        help="Force Singularity runtime (otherwise auto-detected)",
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
        choices=["auto", "tmpfs", "overlay"],
        default="auto",
        help="Writable mode for Phase 1 in Singularity (auto chooses tmpfs, fallback to overlay).",
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


@contextmanager
def redirect_stdout_stderr_to_file(log_file_path):
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log = open(log_file_path, "a")
    sys.stdout = log
    sys.stderr = log
    try:
        yield
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log.close()


_LANGUAGE_FIELD_KEYS = [
    "Programming Language",
    "programming_language",
    "Preferred Programming Language",
    "Preferred Language",
    "preferred_language",
    "preferred programming language",
    "Language",
    "language",
    "Target Language",
    "target_language",
    "Implementation Language",
    "implementation_language",
    "Code Language",
    "code_language",
]

_EXTENSION_LANGUAGE_HINTS = {
    ".py": "python",
    ".ipynb": "python",
    ".cpp": "c++",
    ".cc": "c++",
    ".cxx": "c++",
    ".hpp": "c++",
}

PROMPT_ADAPTER_SYSTEM_MESSAGE = (
    "You are a meticulous prompt editor. Follow the user instructions exactly and "
    "return only the rewritten prompt text. Do not include explanations."
)

LANGUAGE_DECIDER_SYSTEM_MESSAGE = (
    "You choose the implementation language for an automated research experiment. "
    "Respond with a single token: either `python` or `cpp`."
)


def _infer_language_from_idea(
    idea: dict,
    adapter_settings: Dict[str, Any],
) -> Tuple[str, str, bool]:
    template = prompt_loader.load_prompt(
        "treesearch/parallel_agent/language_adapter/language_decider"
    )
    idea_json = json.dumps(idea, ensure_ascii=False, indent=2)
    user_message = template.format(idea_json=idea_json)
    client, client_model = create_client(adapter_settings["model"])
    temperature = adapter_settings.get("temp", 0.0)
    response, _ = get_response_from_llm(
        user_message,
        client,
        client_model,
        LANGUAGE_DECIDER_SYSTEM_MESSAGE,
        temperature=temperature,
    )
    decision = response.strip().lower()
    if "cpp" in decision or "c++" in decision:
        return "C++", "cpp", True
    return "Python", "python", False


def _detect_target_language(
    idea: dict,
    code_path: Optional[str],
    adapter_settings: Dict[str, Any],
    language_override: Optional[str] = None,
) -> Tuple[str, str, bool]:
    """Determine whether prompts should stay in Python or be adapted to C++.

    Returns (language_label, code_fence, adapt_to_cpp).
    """
    def _coerce_language_override(language_value: Optional[str]) -> Optional[Tuple[str, str, bool]]:
        if language_value is None:
            return None
        normalized = str(language_value).strip().lower()
        if not normalized:
            return None
        if normalized in {"cpp", "c++", "cxx"}:
            return "C++", "cpp", True
        if normalized in {"python", "py"}:
            return "Python", "python", False
        raise ValueError(
            f"Unsupported language override '{language_value}'. Use 'python' or 'cpp'."
        )

    override = _coerce_language_override(language_override)
    if override is not None:
        return override

    if code_path:
        ext = Path(code_path).suffix.lower()
        hint = _EXTENSION_LANGUAGE_HINTS.get(ext)
        if hint == "c++":
            return "C++", "cpp", True
        if hint == "python":
            return "Python", "python", False

    return _infer_language_from_idea(idea, adapter_settings)


def _apply_parallel_agent_language_override(prompt_root: Path, language_key: str) -> bool:
    """Overlay language-specific prompt overrides if they have been prepared."""
    parallel_dir = prompt_root / "treesearch" / "parallel_agent"
    if not parallel_dir.exists():
        return False

    override_dir = parallel_dir / "language_adapter" / f"static_{language_key}"
    if not override_dir.exists():
        return False

    for source_path in override_dir.rglob("*"):
        if not source_path.is_file():
            continue
        relative_path = source_path.relative_to(override_dir)
        destination_path = parallel_dir / relative_path
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)
    return True


def _snapshot_and_prepare_prompts(
    repo_root: Path,
    idea_dir: Path,
    language_label: str,
    code_fence: str,
    adapt_to_cpp: bool,
) -> Path:
    src_prompt_dir = repo_root / "prompt"
    dst_prompt_dir = idea_dir / "prompt"
    shutil.copytree(src_prompt_dir, dst_prompt_dir, dirs_exist_ok=True)
    if adapt_to_cpp:
        if _apply_parallel_agent_language_override(dst_prompt_dir, "cpp"):
            print(
                f"Copied prompts to {dst_prompt_dir} and applied static parallel agent prompts for {language_label}."
            )
        else:
            raise FileNotFoundError(
                f"Static C++ prompt overrides not found under {dst_prompt_dir / 'treesearch/parallel_agent/language_adapter/static_cpp'}"
            )
    else:
        print(f"Copied prompts to {dst_prompt_dir} without language adaptation.")
    return dst_prompt_dir


def _load_base_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f) or {}
    return config


def _load_prompt_adapter_settings(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    adapter_cfg = config.get("prompt_adapter")
    if adapter_cfg is None:
        return None
    if "model" not in adapter_cfg:
        raise ValueError("prompt_adapter section must include a 'model' entry.")
    return adapter_cfg


def _extract_role_description(config: Dict[str, Any]) -> Optional[str]:
    agent_cfg = config.get("agent")
    if isinstance(agent_cfg, dict):
        role = agent_cfg.get("role_description")
        if role is None:
            return None
        role_str = str(role).strip()
        return role_str or None
    return None


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
    idea_dir_path = Path(idea_dir)
    print(f"Results will be saved in {idea_dir}")
    idea_dir_path.mkdir(parents=True, exist_ok=True)

    # Prepare idea metadata paths
    idea_path_md = idea_dir_path / "idea.md"

    # If load_code is True, locate a code file that matches the idea file stem
    code = None
    code_path: Optional[str] = None
    if args.load_code:
        idea_json_path = Path(args.load_ideas)
        candidate_extensions = [".py"]
        candidate_extensions.extend(
            ext for ext in _EXTENSION_LANGUAGE_HINTS.keys() if ext not in candidate_extensions
        )
        for ext in candidate_extensions:
            candidate = idea_json_path.with_suffix(ext)
            if candidate.exists():
                code_path = str(candidate)
                with open(candidate, "r") as f:
                    code = f.read()
                break
        if code_path is None:
            fallback_code_path = idea_json_path.with_suffix(".py")
            print(f"Warning: Code file {fallback_code_path} not found")

    base_config_path = repo_root / "bfts_config.yaml"
    base_config = _load_base_config(base_config_path)
    phase_mode = str(base_config.get("exec", {}).get("phase_mode", "split")).lower()
    prompt_adapter_settings = (
        _load_prompt_adapter_settings(base_config) if phase_mode != "split" else None
    )
    if phase_mode != "split" and prompt_adapter_settings is None:
        raise ValueError(
            "prompt_adapter configuration is required for language inference."
        )
    set_persona_role(_extract_role_description(base_config))
    if phase_mode == "split":
        execution_language = base_config.get("exec", {}).get("language", "python")
        adapt_to_cpp = False
        code_fence = execution_language or "python"
        language_label = (code_fence or "python").capitalize()
    else:
        language_label, code_fence, adapt_to_cpp = _detect_target_language(
            idea,
            code_path,
            prompt_adapter_settings,
            language_override=base_config.get("exec", {}).get("language"),
        )
        execution_language = "cpp" if adapt_to_cpp else "python"
    print(
        f"Configured prompts for target language: {language_label} "
        f"(code fence `{code_fence}`, adapt_to_cpp={adapt_to_cpp})"
    )

    agent_file_name = "runfile.cpp" if adapt_to_cpp else "runfile.py"
    env_packages_template = (
        "treesearch/parallel_agent/language_adapter/environment_packages_cpp"
        if adapt_to_cpp
        else None
    )
    prompt_dir = _snapshot_and_prepare_prompts(
        repo_root,
        idea_dir_path,
        language_label,
        code_fence,
        adapt_to_cpp,
    )
    os.environ["AI_SCIENTIST_PROMPT_DIR"] = str(prompt_dir)
    prompt_loader.PROMPT_DIR = Path(prompt_dir)
    prompt_loader.load_prompt.cache_clear()

    perform_experiments_impl = perform_experiments_bfts

    modules_to_reload = [
        "ai_scientist.treesearch.parallel_agent",
        "ai_scientist.treesearch.agent_manager",
        "ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager",
    ]
    for module_name in modules_to_reload:
        importlib.reload(importlib.import_module(module_name))

    perform_module = importlib.import_module(
        "ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager"
    )
    perform_experiments_impl = perform_module.perform_experiments_bfts

    idea_to_markdown(idea, str(idea_path_md), code_path, code_fence=code_fence)

    dataset_ref_code = None
    if args.add_dataset_ref:
        dataset_ref_path = Path("hf_dataset_reference.py")
        if dataset_ref_path.exists():
            with open(dataset_ref_path, "r") as f:
                dataset_ref_code = f.read()
        else:
            print(f"Warning: Dataset reference file {dataset_ref_path} not found")
            dataset_ref_code = None

    if dataset_ref_code is not None and code is not None:
        added_code = dataset_ref_code + "\n" + code
    elif dataset_ref_code is not None and code is None:
        added_code = dataset_ref_code
    elif dataset_ref_code is None and code is not None:
        added_code = code
    else:
        added_code = None

    print(added_code)

    # Add code to idea json if it was loaded
    if added_code is not None:
        ideas[args.idea_idx]["Code"] = added_code

    # Store raw idea json
    idea_path_json = idea_dir_path / "idea.json"
    with open(idea_path_json, "w") as f:
        json.dump(ideas[args.idea_idx], f, indent=4)

    config_path = "bfts_config.yaml"
    idea_config_path = edit_bfts_config_file(
        config_path,
        idea_dir,
        str(idea_path_json),
        language=execution_language,
        agent_file_name=agent_file_name,
        env_packages_template=env_packages_template,
        phase_mode=args.phase_mode,
        singularity_image=args.singularity_image,
        container_runtime=args.container_runtime,
        num_workers=args.num_workers,
        writable_tmpfs=not args.disable_writable_tmpfs,
        container_overlay=args.container_overlay,
        per_worker_sif=args.per_worker_sif,
        keep_sandbox=args.keep_sandbox,
        use_fakeroot=args.use_fakeroot,
        writable_mode=args.writable_mode,
    )

    perform_experiments_impl(idea_config_path)
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
