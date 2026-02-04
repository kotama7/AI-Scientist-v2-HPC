"""Paper writeup functionality for the output module."""

import json
import os
import os.path as osp
import re
import shutil
import traceback
from typing import Optional

from ai_scientist.llm import (
    get_response_from_llm,
    create_client,
)
from ai_scientist.review.vlm_review import generate_vlm_img_review
from ai_scientist.vlm import create_client as create_vlm_client
from ai_scientist.prompt_loader import load_prompt
from ai_scientist.output.latex_utils import (
    compile_latex,
    extract_latex_snippet,
    detect_pages_before_impact,
)
from ai_scientist.output.citation import gather_citations

# Load prompt templates
WRITEUP_SYSTEM_MESSAGE_TEMPLATE = load_prompt("output/writeup/system_message")
WRITEUP_PROMPT_TEMPLATE = load_prompt("output/writeup/writeup_prompt")
WRITEUP_REFLECTION_PROMPT_TEMPLATE = load_prompt("output/writeup/reflection_prompt")


def load_idea_text(base_folder: str) -> str:
    """Load the idea text from the base folder.

    Args:
        base_folder: Path to the project folder.

    Returns:
        Idea text content.
    """
    idea_text = ""
    research_idea_path = osp.join(base_folder, "research_idea.md")
    if osp.exists(research_idea_path):
        with open(research_idea_path, "r") as f_idea:
            idea_text = f_idea.read()
    else:
        idea_md_path = osp.join(base_folder, "idea.md")
        if osp.exists(idea_md_path):
            with open(idea_md_path, "r") as f_idea:
                idea_text = f_idea.read()
    return idea_text


def load_exp_summaries(base_folder: str) -> dict:
    """Load the experiment summaries from the base folder.

    Args:
        base_folder: Path to the project folder.

    Returns:
        Dictionary of loaded summaries.
    """
    summary_files = [
        ("logs/0-run/baseline_summary.json", "BASELINE_SUMMARY"),
        ("logs/0-run/research_summary.json", "RESEARCH_SUMMARY"),
        ("logs/0-run/ablation_summary.json", "ABLATION_SUMMARY"),
    ]
    loaded_summaries = {}
    for fname, key in summary_files:
        path = osp.join(base_folder, fname)
        if osp.exists(path):
            try:
                with open(path, "r") as f:
                    loaded_summaries[key] = json.load(f)
            except json.JSONDecodeError:
                print(
                    f"Warning: {fname} is not valid JSON. Using empty data for {key}."
                )
                loaded_summaries[key] = {}
        else:
            loaded_summaries[key] = {}
    return loaded_summaries


def load_writeup_memory(base_folder: str) -> str:
    """Load the final writeup memory from the memory folder.

    Args:
        base_folder: Path to the project folder.

    Returns:
        String containing writeup memory content (markdown format), or empty string if not found.
    """
    # Try multiple possible locations for memory files (markdown first for comprehensive content)
    memory_paths = [
        osp.join(base_folder, "0-run/memory/final_memory_for_paper.md"),
        osp.join(base_folder, "logs/0-run/memory/final_memory_for_paper.md"),
        osp.join(base_folder, "memory/final_memory_for_paper.md"),
    ]

    for memory_path in memory_paths:
        if osp.exists(memory_path):
            try:
                with open(memory_path, "r", encoding="utf-8") as f:
                    memory_content = f.read()
                print(f"Loaded writeup memory from: {memory_path}")
                return memory_content
            except Exception as e:
                print(f"Warning: Error loading {memory_path}: {e}")

    print("Warning: final_memory_for_paper.md not found in expected locations")
    return ""


def filter_experiment_summaries(exp_summaries: dict, step_name: str) -> dict:
    """Filter experiment summaries based on the step name.

    Args:
        exp_summaries: Raw experiment summaries dictionary.
        step_name: One of 'citation_gathering', 'writeup', or 'plot_aggregation'.

    Returns:
        Filtered summaries dictionary.

    Raises:
        ValueError: If step_name is invalid.
    """
    if step_name == "citation_gathering":
        node_keys_to_keep = {
            "overall_plan",
            "analysis",
            "metric",
            "vlm_feedback_summary",
            "phase0_plan",
            "phase1_steps_summary",
            "phase3_compile_log_summary",
            "phase4_run_log_summary",
            "phase_artifacts_summary",
        }
    elif step_name == "writeup":
        node_keys_to_keep = {
            "overall_plan",
            "analysis",
            "metric",
            "code",
            "plot_analyses",
            "vlm_feedback_summary",
            "phase0_plan",
            "phase1_steps_summary",
            "phase3_compile_log_summary",
            "phase4_run_log_summary",
            "phase_artifacts_summary",
        }
    elif step_name == "plot_aggregation":
        node_keys_to_keep = {
            "overall_plan",
            "analysis",
            "plot_plan",
            "plot_code",
            "plot_analyses",
            "vlm_feedback_summary",
            "exp_results_npy_files",
        }
    else:
        raise ValueError(f"Invalid step name: {step_name}")

    filtered_summaries = {}
    for stage_name in exp_summaries.keys():
        if stage_name in {"BASELINE_SUMMARY", "RESEARCH_SUMMARY"}:
            filtered_summaries[stage_name] = {}
            for key in exp_summaries[stage_name].keys():
                if key in {"best node"}:
                    filtered_summaries[stage_name][key] = {}
                    for node_key in exp_summaries[stage_name][key].keys():
                        if node_key in node_keys_to_keep:
                            filtered_summaries[stage_name][key][node_key] = (
                                exp_summaries[stage_name][key][node_key]
                            )
        elif stage_name == "ABLATION_SUMMARY" and step_name == "plot_aggregation":
            filtered_summaries[stage_name] = {}
            for ablation_summary in exp_summaries[stage_name]:
                filtered_summaries[stage_name][ablation_summary["ablation_name"]] = {}
                for node_key in ablation_summary.keys():
                    if node_key in node_keys_to_keep:
                        filtered_summaries[stage_name][
                            ablation_summary["ablation_name"]
                        ][node_key] = ablation_summary[node_key]
    return filtered_summaries


def perform_writeup(
    base_folder: str,
    citations_text: str = None,
    no_writing: bool = False,
    num_cite_rounds: int = 20,
    small_model: str = "gpt-4o-2024-05-13",
    big_model: str = "o1-2024-12-17",
    n_writeup_reflections: int = 3,
    page_limit: Optional[int] = 8,
) -> bool:
    """Perform the paper writeup process.

    Args:
        base_folder: Path to the project folder.
        citations_text: Optional pre-gathered citations.
        no_writing: If True, only compile without generating new content.
        num_cite_rounds: Maximum citation gathering rounds.
        small_model: Model for citation gathering.
        big_model: Model for writeup generation.
        n_writeup_reflections: Number of reflection rounds.
        page_limit: Target page limit (None for unlimited).

    Returns:
        True if successful, False otherwise.
    """
    compile_attempt = 0
    base_pdf_file = osp.join(base_folder, f"{osp.basename(base_folder)}")
    latex_folder = osp.join(base_folder, "latex")

    # Cleanup any previous latex folder
    if osp.exists(latex_folder):
        shutil.rmtree(latex_folder)

    try:
        # Load idea text and summaries
        idea_text = load_idea_text(base_folder)
        exp_summaries = load_exp_summaries(base_folder)
        filtered_summaries_for_writeup = filter_experiment_summaries(
            exp_summaries, step_name="writeup"
        )
        combined_summaries_str = json.dumps(filtered_summaries_for_writeup, indent=2)

        # Load writeup memory from final_memory_for_paper.md
        writeup_memory_str = load_writeup_memory(base_folder)

        # Prepare a new fresh latex folder
        if not osp.exists(osp.join(latex_folder, "template.tex")):
            shutil.copytree(
                "ai_scientist/blank_latex", latex_folder, dirs_exist_ok=True
            )

        writeup_file = osp.join(latex_folder, "template.tex")
        with open(writeup_file, "r") as f:
            writeup_text = f.read()

        # Gather plot filenames from figures/ folder
        figures_dir = osp.join(base_folder, "figures")
        plot_names = []
        if osp.exists(figures_dir):
            for fplot in os.listdir(figures_dir):
                if fplot.lower().endswith(".png"):
                    plot_names.append(fplot)

        # Load aggregator script to include in the prompt
        aggregator_path = osp.join(base_folder, "auto_plot_aggregator.py")
        aggregator_code = ""
        if osp.exists(aggregator_path):
            with open(aggregator_path, "r") as fa:
                aggregator_code = fa.read()
        else:
            aggregator_code = "No aggregator script found."

        if no_writing:
            compile_latex(latex_folder, base_pdf_file + ".pdf")
            return osp.exists(base_pdf_file + ".pdf")

        # If no citations provided, try to load from cache first
        if citations_text is None:
            citations_cache_path = osp.join(base_folder, "cached_citations.bib")
            if osp.exists(citations_cache_path):
                try:
                    with open(citations_cache_path, "r") as f:
                        citations_text = f.read()
                    print("Loaded citations from cache")
                except Exception as e:
                    print(f"Error loading cached citations: {e}")
                    citations_text = None

            # If still no citations, gather them
            if not citations_text:
                citations_text = gather_citations(
                    base_folder, num_cite_rounds, small_model
                )
                if citations_text is None:
                    print("Warning: Citation gathering failed")
                    citations_text = ""

        # Insert citations into template.tex
        if citations_text:
            with open(writeup_file, "r") as f:
                content = f.read()
            pattern_end = r"\end{filecontents}"
            content = content.replace(pattern_end, f"\n{citations_text}{pattern_end}")
            with open(writeup_file, "w") as f:
                f.write(content)

        # Generate VLM-based descriptions
        try:
            vlm_client, vlm_model = create_vlm_client(small_model)
            desc_map = {}
            for pf in plot_names:
                ppath = osp.join(figures_dir, pf)
                if not osp.exists(ppath):
                    continue
                img_dict = {
                    "images": [ppath],
                    "caption": "No direct caption",
                }
                review_data = generate_vlm_img_review(img_dict, vlm_model, vlm_client)
                if review_data:
                    desc_map[pf] = review_data.get(
                        "Img_description", "No description found"
                    )
                else:
                    desc_map[pf] = "No description found"

            # Prepare a string listing all figure descriptions
            plot_descriptions_list = []
            for fname in plot_names:
                desc_text = desc_map.get(fname, "No description found")
                plot_descriptions_list.append(f"{fname}: {desc_text}")
            plot_descriptions_str = "\n".join(plot_descriptions_list)
        except Exception:
            print("EXCEPTION in VLM figure description generation:")
            print(traceback.format_exc())
            plot_descriptions_str = "No descriptions available."

        # Construct final prompt for big model
        if page_limit is None:
            page_limit_line = (
                "There is no strict page limit for the main paper. Use the space "
                "needed for clarity and completeness, and avoid unnecessary filler."
            )
        else:
            page_limit_line = (
                "The main paper is limited to "
                f"{page_limit} pages, including all figures and tables, but excluding "
                "references, the impact statement, and optional appendices. In general, "
                "try to use the available space and include all relevant information."
            )
        big_model_system_message = WRITEUP_SYSTEM_MESSAGE_TEMPLATE.format(
            page_limit_line=page_limit_line
        )
        big_client, big_client_model = create_client(big_model)
        with open(writeup_file, "r") as f:
            writeup_text = f.read()

        combined_prompt = WRITEUP_PROMPT_TEMPLATE.format(
            idea_text=idea_text,
            summaries=combined_summaries_str,
            writeup_memory=writeup_memory_str,
            aggregator_code=aggregator_code,
            plot_list=", ".join(plot_names),
            latex_writeup=writeup_text,
            plot_descriptions=plot_descriptions_str,
        )

        response, msg_history = get_response_from_llm(
            prompt=combined_prompt,
            client=big_client,
            model=big_client_model,
            system_message=big_model_system_message,
            print_debug=False,
        )

        latex_code = extract_latex_snippet(response)
        if not latex_code:
            print("No valid LaTeX code block found in initial writeup response.")
            print("Response was:")
            print(response)
            return False
        updated_latex_code = latex_code
        with open(writeup_file, "w") as f:
            f.write(updated_latex_code)

        # Multiple reflection loops on the final LaTeX
        for i in range(n_writeup_reflections):
            with open(writeup_file, "r") as f:
                current_latex = f.read()

            # Check for unused or invalid figure references
            referenced_figs_temp = re.findall(
                r"\\includegraphics(?:\[[^\]]*\])?{([^}]+)}", current_latex
            )
            used_figs = set(os.path.basename(fig) for fig in referenced_figs_temp)
            all_figs = set(plot_names)
            unused_figs = all_figs - used_figs
            invalid_figs = used_figs - all_figs

            # Compile current version before reflection
            compile_latex(latex_folder, base_pdf_file + f"_reflection_{compile_attempt}.pdf")
            compile_attempt += 1
            print(f"Compiled {base_pdf_file}_reflection_{compile_attempt}.pdf")

            # Detect where "Impact Statement" appears
            impact_loc = detect_pages_before_impact(latex_folder)
            if page_limit is None:
                reflection_page_info = (
                    "\nNo page limit is enforced for the main text. "
                    "Focus on clarity, completeness, and correct formatting.\n"
                )
            elif impact_loc is not None:
                page_num, line_num = impact_loc
                reflection_page_info = (
                    f"\nCurrently, 'Impact Statement' begins on page {page_num}, approximately on line {line_num}. "
                    f"The page limit is {page_limit}, which is before the Impact Statement. "
                    f"Papers often look more professional if the main text is near or just under {page_limit} pages in length.\n"
                )
            else:
                reflection_page_info = "\nCould not detect 'Impact Statement' page (compilation or detection failed).\n"

            check_output = os.popen(
                f"chktex {writeup_file} -q -n2 -n24 -n13 -n1"
            ).read()

            reflection_prompt = WRITEUP_REFLECTION_PROMPT_TEMPLATE.format(
                unused_figs=sorted(unused_figs),
                invalid_figs=sorted(invalid_figs),
                reflection_page_info=reflection_page_info,
                check_output=check_output,
            )

            reflection_response, msg_history = get_response_from_llm(
                prompt=reflection_prompt,
                client=big_client,
                model=big_client_model,
                system_message=big_model_system_message,
                msg_history=msg_history,
                print_debug=False,
            )

            if "I am done" in reflection_response:
                print(
                    "LLM indicated it is done with reflections. Exiting reflection loop."
                )
                break

            reflected_latex_code = extract_latex_snippet(reflection_response)
            if reflected_latex_code:
                if reflected_latex_code != current_latex:
                    final_text = reflected_latex_code
                    cleanup_map = {
                        "</end": r"\\end",
                        "</begin": r"\\begin",
                        "'": "'",
                    }
                    for bad_str, repl_str in cleanup_map.items():
                        final_text = final_text.replace(bad_str, repl_str)
                    final_text = re.sub(r"(\d+(?:\.\d+)?)%", r"\1\\%", final_text)

                    with open(writeup_file, "w") as fo:
                        fo.write(final_text)

                    compile_latex(
                        latex_folder, base_pdf_file + f"_reflection_{compile_attempt}.pdf"
                    )
                    compile_attempt += 1
                    print(f"Compiled {base_pdf_file}_reflection_{compile_attempt}.pdf")
                else:
                    print(f"No changes in reflection step {i+1}.")
                    break
            else:
                print(f"No valid LaTeX code block found in reflection step {i+1}.")
                break

        return osp.exists(base_pdf_file + f"_reflection_{compile_attempt-1}.pdf")

    except Exception:
        print("EXCEPTION in perform_writeup:")
        print(traceback.format_exc())
        return False
