import json
import os
import sys
from pathlib import Path

from .journal import Node, Journal

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, parent_dir)
from ai_scientist.llm import get_response_from_llm, extract_json_between_markers
from ai_scientist.treesearch.backend import get_ai_client

from ai_scientist.prompt_loader import load_prompt


REPORT_SUMMARIZER_SYS_MSG = load_prompt(
    "treesearch/log_summarization/report_summarizer_system"
).strip()
OUTPUT_FORMAT_CONTROL = load_prompt(
    "treesearch/log_summarization/output_format_control"
).strip()
REPORT_SUMMARIZER_PROMPT_TEMPLATE = load_prompt(
    "treesearch/log_summarization/report_summarizer_prompt"
)
STAGE_AGGREGATE_PROMPT_TEMPLATE = load_prompt(
    "treesearch/log_summarization/stage_aggregate_prompt"
)
OVERALL_PLAN_SUMMARIZER_TEMPLATE = load_prompt(
    "treesearch/log_summarization/overall_plan_summarizer_prompt"
)


def _read_text(path):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except OSError:
        return ""


def _safe_read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _safe_read_jsonl(path, max_entries=1000):
    entries = []
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
                if len(entries) >= max_entries:
                    break
    except OSError:
        return []
    return entries


def _extract_error_lines(text, max_lines=12):
    if not text:
        return []
    hits = []
    for line in text.splitlines():
        lower = line.lower()
        if (
            "error" in lower
            or "failed" in lower
            or "undefined reference" in lower
            or "not found" in lower
        ):
            hits.append(line.strip())
    return hits[-max_lines:]


def _summarize_phase1_steps(path, max_recent=5):
    entries = _safe_read_jsonl(path)
    if not entries:
        return {}
    recent = []
    for entry in entries[-max_recent:]:
        recent.append(
            {
                "step": entry.get("step"),
                "command": entry.get("command"),
                "exit_code": entry.get("exit_code"),
                "done": entry.get("done"),
                "notes": entry.get("notes"),
                "stdout_summary": entry.get("stdout_summary"),
                "stderr_summary": entry.get("stderr_summary"),
            }
        )
    error_lines = []
    for entry in entries:
        if entry.get("exit_code") not in (None, 0):
            error_lines.extend(_extract_error_lines(entry.get("stderr_summary", "")))
    if error_lines:
        error_lines = error_lines[-12:]
    return {
        "total_steps": len(entries),
        "recent_steps": recent,
        "error_lines": error_lines,
    }


def _normalize_phase_artifacts(raw):
    if not raw:
        return None
    if isinstance(raw, dict) and raw.get("phase_artifacts"):
        raw = raw.get("phase_artifacts")
    if isinstance(raw, list) and raw:
        raw = raw[0]
    if not isinstance(raw, dict):
        return None
    return raw


def _summarize_phase_artifacts(raw):
    phase = _normalize_phase_artifacts(raw)
    if not phase:
        return {}
    download = phase.get("download", {}) if isinstance(phase.get("download", {}), dict) else {}
    coding = phase.get("coding", {}) if isinstance(phase.get("coding", {}), dict) else {}
    compile_section = (
        phase.get("compile", {}) if isinstance(phase.get("compile", {}), dict) else {}
    )
    run_section = phase.get("run", {}) if isinstance(phase.get("run", {}), dict) else {}

    workspace = coding.get("workspace", {}) if isinstance(coding.get("workspace", {}), dict) else {}
    files = workspace.get("files", []) if isinstance(workspace.get("files", []), list) else []
    file_paths = [
        f.get("path") for f in files if isinstance(f, dict) and f.get("path")
    ]

    return {
        "download": {
            "commands": download.get("commands", [])[:10],
            "notes": download.get("notes", ""),
        },
        "coding": {
            "notes": coding.get("notes", ""),
            "workspace": {
                "root": workspace.get("root"),
                "tree": workspace.get("tree", [])[:20],
                "files": file_paths[:20],
            },
        },
        "compile": {
            "build_plan": compile_section.get("build_plan", {}),
            "commands": compile_section.get("commands", []),
            "notes": compile_section.get("notes", ""),
        },
        "run": {
            "commands": run_section.get("commands", []),
            "expected_outputs": run_section.get("expected_outputs", []),
            "notes": run_section.get("notes", ""),
        },
    }


def _summarize_phase_log(path, max_tail_lines=40):
    text = _read_text(path)
    if not text:
        return {}
    lines = text.splitlines()
    return {
        "tail": lines[-max_tail_lines:],
        "error_lines": _extract_error_lines(text),
    }


def get_nodes_infos(nodes):
    node_infos = ""
    for n in nodes:
        node_info = f"Node ID: {n.id}\n"
        node_info += (
            f"Plan: {n.overall_plan}\n"
            if hasattr(n, "overall_plan")
            else "Plan: Not available\n"
        )
        node_info += (
            f"Analysis: {n.analysis}\n"
            if hasattr(n, "analysis")
            else "Analysis: Not available\n"
        )
        node_info += (
            f"Numerical Results: {n.metric}\n"
            if hasattr(n, "metric")
            else "Numerical Results: Not available\n"
        )
        phase_summary = _summarize_phase_artifacts(
            n.phase_artifacts if hasattr(n, "phase_artifacts") else None
        )
        node_info += (
            "Phase Artifacts Summary: "
            + (
                json.dumps(phase_summary, ensure_ascii=True)
                if phase_summary
                else "Not available"
            )
            + "\n"
        )
        node_info += "Plot Analyses:\n"
        if hasattr(n, "plot_analyses") and n.plot_analyses:
            for plot in n.plot_analyses:
                node_info += f"- Plot Path: {plot.get('plot_path', 'Not available')}, Description: {plot.get('analysis', 'Not available')}\n"
        else:
            node_info += "No plot analyses available\n"
        node_infos += node_info + "\n"
    return node_infos


def get_summarizer_prompt(journal, stage_name):
    good_leaf_nodes = [n for n in journal.good_nodes if n.is_leaf]
    if not good_leaf_nodes:
        print("NO GOOD LEAF NODES!!!")
        good_leaf_nodes = [n for n in journal.good_nodes]
    node_infos = get_nodes_infos(good_leaf_nodes)
    prompt_text = REPORT_SUMMARIZER_PROMPT_TEMPLATE.format(
        node_infos=node_infos,
        stage_name=stage_name,
        output_format_control=OUTPUT_FORMAT_CONTROL,
    )
    return REPORT_SUMMARIZER_SYS_MSG, prompt_text


def get_stage_summary(journal, stage_name, model, client):
    sys_msg, prompt = get_summarizer_prompt(journal, stage_name)
    response = get_response_from_llm(prompt, client, model, sys_msg)
    summary_json = extract_json_between_markers(response[0])
    return summary_json


def get_node_log(node):
    node_dict = node.to_dict()
    # Only include keys that are relevant for logging/analysis
    keys_to_include = [
        "overall_plan",
        "analysis",
        "metric",
        "code",
        "plot_code",
        "plot_plan",
        "plot_analyses",
        "plot_paths",
        "vlm_feedback_summary",
        "exp_results_dir",
        "ablation_name",
    ]
    ret = {
        key: node_dict[key]
        for key in keys_to_include
        if key in node_dict and node_dict[key] is not None
    }
    phase_artifacts_summary = _summarize_phase_artifacts(
        node_dict.get("phase_artifacts")
    )
    if phase_artifacts_summary:
        ret["phase_artifacts_summary"] = phase_artifacts_summary
    if "exp_results_dir" in ret:
        original_dir_path = ret["exp_results_dir"]
        # Remove leading path segments before "experiment_results"
        idx = original_dir_path.find("experiment_results")
        short_dir_path = original_dir_path
        if idx != -1:
            short_dir_path = original_dir_path[idx:]

        ret["exp_results_dir"] = short_dir_path

        if os.path.isdir(original_dir_path):
            npy_files = sorted(Path(original_dir_path).rglob("*.npy"))
            # Use absolute paths so plot aggregation can load reliably.
            ret["exp_results_npy_files"] = [str(p.resolve()) for p in npy_files]
            llm_outputs_dir = os.path.join(original_dir_path, "llm_outputs")
            phase0_path = os.path.join(llm_outputs_dir, "phase0_plan.json")
            phase1_steps_path = os.path.join(llm_outputs_dir, "phase1_steps.jsonl")
            phase_artifacts_dir = os.path.join(original_dir_path, "phase_artifacts")
            compile_log_path = os.path.join(phase_artifacts_dir, "compile.log")
            run_log_path = os.path.join(phase_artifacts_dir, "run.log")

            phase0_plan = _safe_read_json(phase0_path) if os.path.isfile(phase0_path) else None
            if phase0_plan:
                ret["phase0_plan"] = phase0_plan.get("plan", phase0_plan)
                ret["phase0_plan_path"] = os.path.join(
                    short_dir_path, "llm_outputs", "phase0_plan.json"
                )

            phase1_summary = (
                _summarize_phase1_steps(phase1_steps_path)
                if os.path.isfile(phase1_steps_path)
                else {}
            )
            if phase1_summary:
                ret["phase1_steps_summary"] = phase1_summary
                ret["phase1_steps_path"] = os.path.join(
                    short_dir_path, "llm_outputs", "phase1_steps.jsonl"
                )

            if os.path.isfile(compile_log_path):
                ret["phase3_compile_log_summary"] = _summarize_phase_log(
                    compile_log_path
                )
                ret["phase3_compile_log_path"] = os.path.join(
                    short_dir_path, "phase_artifacts", "compile.log"
                )
            if os.path.isfile(run_log_path):
                ret["phase4_run_log_summary"] = _summarize_phase_log(run_log_path)
                ret["phase4_run_log_path"] = os.path.join(
                    short_dir_path, "phase_artifacts", "run.log"
                )
        else:
            ret["exp_results_npy_files"] = []
    return ret


def update_summary(
    prev_summary, cur_stage_name, cur_journal, cur_summary, model, client, max_retry=5
):
    prompt = STAGE_AGGREGATE_PROMPT_TEMPLATE.format(
        prev_summary=prev_summary,
        stage_name=cur_stage_name,
        current_summary=cur_summary,
    )
    try:
        response = get_response_from_llm(
            prompt, client, model, "You are an expert machine learning researcher."
        )
        summary_json = extract_json_between_markers(response[0])
        assert summary_json
    except Exception as e:
        if max_retry > 0:
            print(f"Error occurred: {e}. Retrying... ({max_retry} attempts left)")
            return update_summary(
                prev_summary,
                cur_stage_name,
                cur_journal,
                cur_summary,
                model,
                client,
                max_retry - 1,
            )
        else:
            print(f"Failed to update summary after multiple attempts. Error: {e}")
            raise
    return summary_json


def annotate_history(journal, cfg=None):
    for node in journal.nodes:
        if node.parent:
            max_retries = 3
            retry_count = 0
            while retry_count < max_retries:
                try:
                    if cfg.agent.get("summary", None) is not None:
                        model = cfg.agent.summary.model
                    else:
                        model = "gpt-4o-2024-08-06"
                    client = get_ai_client(model)
                    prompt_text = OVERALL_PLAN_SUMMARIZER_TEMPLATE.format(
                        prev_overall_plan=node.parent.overall_plan,
                        current_plan=node.plan,
                    )
                    response = get_response_from_llm(
                        prompt_text,
                        client,
                        model,
                        REPORT_SUMMARIZER_SYS_MSG,
                    )
                    node.overall_plan = extract_json_between_markers(response[0])[
                        "overall_plan"
                    ]
                    break
                except Exception as e:
                    retry_count += 1
                    if retry_count == max_retries:
                        print(f"Failed after {max_retries} attempts. Error: {e}")
                        raise
                    print(
                        f"Error occurred: {e}. Retrying... ({max_retries - retry_count} attempts left)"
                    )
        else:
            node.overall_plan = node.plan


def overall_summarize(journals, cfg=None):
    from concurrent.futures import ThreadPoolExecutor

    def process_stage(idx, stage_tuple):
        stage_name, journal = stage_tuple
        annotate_history(journal, cfg=cfg)
        if idx in [1, 2]:
            best_node = journal.get_best_node(cfg=cfg)
            # get multi-seed results and aggregater node
            child_nodes = best_node.children
            multi_seed_nodes = [
                n for n in child_nodes if n.is_seed_node and not n.is_seed_agg_node
            ]
            agg_node = None
            for n in child_nodes:
                if n.is_seed_node and n.is_seed_agg_node:
                    agg_node = n
                    break
            if agg_node is None:
                # skip agg node
                return {
                    "best node": get_node_log(best_node),
                    "best node with different seeds": [
                        get_node_log(n) for n in multi_seed_nodes
                    ],
                }
            else:
                return {
                    "best node": get_node_log(best_node),
                    "best node with different seeds": [
                        get_node_log(n) for n in multi_seed_nodes
                    ],
                    "aggregated results of nodes with different seeds": get_node_log(
                        agg_node
                    ),
                }
        elif idx == 3:
            good_leaf_nodes = [
                n for n in journal.good_nodes if n.is_leaf and n.ablation_name
            ]
            return [get_node_log(n) for n in good_leaf_nodes]
        elif idx == 0:
            if cfg.agent.get("summary", None) is not None:
                model = cfg.agent.summary.get("model", "")
            else:
                model = "gpt-4o-2024-08-06"
            client = get_ai_client(model)
            summary_json = get_stage_summary(journal, stage_name, model, client)
            return summary_json

    from tqdm import tqdm

    with ThreadPoolExecutor() as executor:
        results = list(
            tqdm(
                executor.map(process_stage, range(len(list(journals))), journals),
                desc="Processing stages",
                total=len(list(journals)),
            )
        )
        draft_summary, baseline_summary, research_summary, ablation_summary = results

    return draft_summary, baseline_summary, research_summary, ablation_summary


if __name__ == "__main__":
    # Test
    example_path = "logs/247-run"

    def load_stage_folders(base_path):
        """
        Load the folders that start with 'stage_' followed by a number.

        Args:
            base_path (str): The base directory path where stage folders are located.

        Returns:
            list: A sorted list of stage folder paths.
        """
        stage_folders = []
        for folder_name in os.listdir(base_path):
            if folder_name.startswith("stage_"):
                stage_folders.append(os.path.join(base_path, folder_name))
        return sorted(stage_folders, key=lambda x: int(x.split("_")[1]))

    def reconstruct_journal(journal_data):
        # Create a mapping of node IDs to Node instances
        id_to_node = {}
        for node_data in journal_data["nodes"]:
            # Remove unused or invalid keys if needed
            if "actionable_insights_from_plots" in node_data:
                del node_data["actionable_insights_from_plots"]
            node = Node.from_dict(node_data)
            id_to_node[node.id] = node

        # Set up parent-child relationships using node2parent
        for node_id, parent_id in journal_data["node2parent"].items():
            child_node = id_to_node[node_id]
            parent_node = id_to_node[parent_id]
            child_node.parent = parent_node
            parent_node.children.add(child_node)

        # Create a Journal and add all nodes
        journal = Journal()
        journal.nodes.extend(id_to_node.values())

        return journal

    # Example usage
    stage_folders = load_stage_folders(example_path)
    journals = []
    for index, folder in enumerate(stage_folders, start=1):
        print(f"Stage {index}: {folder}")
        stage_name = os.path.basename(folder)
        journal_path = os.path.join(folder, "journal.json")
        if os.path.exists(journal_path):
            with open(journal_path, "r") as file:
                journal_data = json.load(file)
                print(f"Loaded journal.json for Stage {index}")
        else:
            print(f"No journal.json found for Stage {index}")
        journal = reconstruct_journal(journal_data)
        journals.append((stage_name, journal))

    # Convert manager journals to list of (stage_name, journal) tuples
    (
        draft_summary,
        baseline_summary,
        research_summary,
        ablation_summary,
    ) = overall_summarize(journals)
    log_dir = "logs/247-run"
    draft_summary_path = log_dir + "/draft_summary.json"
    baseline_summary_path = log_dir + "/baseline_summary.json"
    research_summary_path = log_dir + "/research_summary.json"
    ablation_summary_path = log_dir + "/ablation_summary.json"

    with open(draft_summary_path, "w") as draft_file:
        json.dump(draft_summary, draft_file, indent=2)

    with open(baseline_summary_path, "w") as baseline_file:
        json.dump(baseline_summary, baseline_file, indent=2)

    with open(research_summary_path, "w") as research_file:
        json.dump(research_summary, research_file, indent=2)

    with open(ablation_summary_path, "w") as ablation_file:
        json.dump(ablation_summary, ablation_file, indent=2)

    print(f"Summary reports written to files:")
    print(f"- Draft summary: {draft_summary_path}")
    print(f"- Baseline summary: {baseline_summary_path}")
    print(f"- Research summary: {research_summary_path}")
    print(f"- Ablation summary: {ablation_summary_path}")
