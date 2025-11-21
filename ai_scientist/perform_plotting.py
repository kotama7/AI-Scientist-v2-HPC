import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import traceback
from rich import print

from ai_scientist.llm import create_client, get_response_from_llm
from ai_scientist.utils.token_tracker import token_tracker
from ai_scientist.perform_icbinb_writeup import (
    load_idea_text,
    load_exp_summaries,
    filter_experiment_summaries,
)
from ai_scientist.prompt_loader import load_prompt

MAX_FIGURES = 12

AGGREGATOR_SYSTEM_MSG_TEMPLATE = load_prompt("plotting/system_message")
PLOT_AGGREGATOR_PROMPT_TEMPLATE = load_prompt("plotting/aggregator_prompt")
PLOT_REFLECTION_PROMPT_TEMPLATE = load_prompt("plotting/reflection_prompt")



def build_aggregator_prompt(combined_summaries_str, idea_text):
    return PLOT_AGGREGATOR_PROMPT_TEMPLATE.format(
        idea_text=idea_text, combined_summaries_str=combined_summaries_str
    )


def extract_code_snippet(text: str) -> str:
    """
    Look for a Python code block in triple backticks in the LLM response.
    Return only that code. If no code block is found, return the entire text.
    """
    pattern = r"```(?:python)?(.*?)```"
    matches = re.findall(pattern, text, flags=re.DOTALL)
    return matches[0].strip() if matches else text.strip()


def run_aggregator_script(
    aggregator_code, aggregator_script_path, base_folder, script_name
):
    if not aggregator_code.strip():
        print("No aggregator code was provided. Skipping aggregator script run.")
        return ""
    with open(aggregator_script_path, "w") as f:
        f.write(aggregator_code)

    print(
        f"Aggregator script written to '{aggregator_script_path}'. Attempting to run it..."
    )

    aggregator_out = ""
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            cwd=base_folder,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        aggregator_out = result.stdout + "\n" + result.stderr
        print("Aggregator script ran successfully.")
    except subprocess.CalledProcessError as e:
        aggregator_out = (e.stdout or "") + "\n" + (e.stderr or "")
        print("Error: aggregator script returned a non-zero exit code.")
        print(e)
    except Exception as e:
        aggregator_out = str(e)
        print("Error while running aggregator script.")
        print(e)

    return aggregator_out


def aggregate_plots(
    base_folder: str, model: str = "o1-2024-12-17", n_reflections: int = 5
) -> None:
    filename = "auto_plot_aggregator.py"
    aggregator_script_path = os.path.join(base_folder, filename)
    figures_dir = os.path.join(base_folder, "figures")

    # Clean up previous files
    if os.path.exists(aggregator_script_path):
        os.remove(aggregator_script_path)
    if os.path.exists(figures_dir):
        shutil.rmtree(figures_dir)
        print(f"Cleaned up previous figures directory")

    idea_text = load_idea_text(base_folder)
    exp_summaries = load_exp_summaries(base_folder)
    filtered_summaries_for_plot_agg = filter_experiment_summaries(
        exp_summaries, step_name="plot_aggregation"
    )
    # Convert them to one big JSON string for context
    combined_summaries_str = json.dumps(filtered_summaries_for_plot_agg, indent=2)

    # Build aggregator prompt
    aggregator_prompt = build_aggregator_prompt(combined_summaries_str, idea_text)

    # Call LLM
    client, model_name = create_client(model)
    response, msg_history = None, []
    try:
        response, msg_history = get_response_from_llm(
            prompt=aggregator_prompt,
            client=client,
            model=model_name,
            system_message=AGGREGATOR_SYSTEM_MSG_TEMPLATE.format(
                max_figures=MAX_FIGURES
            ),
            print_debug=False,
            msg_history=msg_history,
        )
    except Exception:
        traceback.print_exc()
        print("Failed to get aggregator script from LLM.")
        return

    aggregator_code = extract_code_snippet(response)
    if not aggregator_code.strip():
        print(
            "No Python code block was found in LLM response. Full response:\n", response
        )
        return

    # First run of aggregator script
    aggregator_out = run_aggregator_script(
        aggregator_code, aggregator_script_path, base_folder, filename
    )

    # Multiple reflection loops
    for i in range(n_reflections):
        # Check number of figures
        figure_count = 0
        if os.path.exists(figures_dir):
            figure_count = len(
                [
                    f
                    for f in os.listdir(figures_dir)
                    if os.path.isfile(os.path.join(figures_dir, f))
                ]
            )
        print(f"[{i + 1} / {n_reflections}]: Number of figures: {figure_count}")
        # Reflection prompt with reminder for common checks and early exit
        reflection_prompt = PLOT_REFLECTION_PROMPT_TEMPLATE.format(
            figure_count=figure_count,
            aggregator_out=aggregator_out,
            max_figures=MAX_FIGURES,
        )

        print("[green]Reflection prompt:[/green] ", reflection_prompt)
        try:
            reflection_response, msg_history = get_response_from_llm(
                prompt=reflection_prompt,
                client=client,
                model=model_name,
                system_message=AGGREGATOR_SYSTEM_MSG_TEMPLATE.format(
                    max_figures=MAX_FIGURES
                ),
                print_debug=False,
                msg_history=msg_history,
            )

        except Exception:
            traceback.print_exc()
            print("Failed to get reflection from LLM.")
            return

        # Early-exit check
        if figure_count > 0 and "I am done" in reflection_response:
            print("LLM indicated it is done with reflections. Exiting reflection loop.")
            break

        aggregator_new_code = extract_code_snippet(reflection_response)

        # If new code is provided and differs, run again
        if (
            aggregator_new_code.strip()
            and aggregator_new_code.strip() != aggregator_code.strip()
        ):
            aggregator_code = aggregator_new_code
            aggregator_out = run_aggregator_script(
                aggregator_code, aggregator_script_path, base_folder, filename
            )
        else:
            print(
                f"No new aggregator script was provided or it was identical. Reflection step {i+1} complete."
            )


def main():
    parser = argparse.ArgumentParser(
        description="Generate and execute a final plot aggregation script with LLM assistance."
    )
    parser.add_argument(
        "--folder",
        required=True,
        help="Path to the experiment folder with summary JSON files.",
    )
    parser.add_argument(
        "--model",
        default="o1-2024-12-17",
        help="LLM model to use (default: o1-2024-12-17).",
    )
    parser.add_argument(
        "--reflections",
        type=int,
        default=5,
        help="Number of reflection steps to attempt (default: 5).",
    )
    args = parser.parse_args()
    aggregate_plots(
        base_folder=args.folder, model=args.model, n_reflections=args.reflections
    )


if __name__ == "__main__":
    main()
