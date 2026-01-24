"""configuration and setup utils"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Hashable, cast, Literal, Optional

import coolname
import rich
from omegaconf import OmegaConf
from rich.syntax import Syntax
import shutup
from rich.logging import RichHandler
import logging

from .file_ops import copytree, preproc_data
from .resource import load_resources, resolve_resources_path, stage_resource_items
from ai_scientist.persona import set_persona_role

shutup.mute_warnings()
logging.basicConfig(
    level="WARNING", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
)
logger = logging.getLogger("ai-scientist")
logger.setLevel(logging.WARNING)


""" these dataclasses are just for type hinting, the actual config is in config.yaml """


@dataclass
class ThinkingConfig:
    type: str
    budget_tokens: Optional[int] = None


@dataclass
class StageConfig:
    model: str
    temp: float
    thinking: Optional[ThinkingConfig] = None
    betas: Optional[str] = None
    max_tokens: Optional[int] = None


@dataclass
class SearchConfig:
    max_debug_depth: int
    debug_prob: float
    num_drafts: int


@dataclass
class DebugConfig:
    stage4: bool = False


@dataclass
class AgentConfig:
    steps: int
    stages: dict[str, int]
    code: StageConfig
    feedback: StageConfig
    vlm_feedback: StageConfig

    search: SearchConfig
    num_workers: int
    type: str
    multi_seed_eval: dict[str, int]

    role_description: str = "AI researcher"

    summary: Optional[StageConfig] = None
    select_node: Optional[StageConfig] = None

@dataclass
class ExecConfig:
    timeout: int = 3600
    agent_file_name: str = "runfile.py"
    format_tb_ipython: bool = True
    language: str = "auto"
    env_packages_template: str | None = None
    phase_mode: str = "split"
    singularity_image: str | None = None
    use_gpu: bool = True
    workspace_mount: str = "/workspace"
    writable_tmpfs: bool = True
    container_overlay: str | None = None
    container_extra_args: list[str] | None = None
    per_worker_sif: bool = True
    keep_sandbox: bool = False
    use_fakeroot: bool = True
    writable_mode: str = "auto"
    phase1_max_steps: int = 12
    resources: str | None = None
    log_prompts: bool = True


@dataclass
class MemoryConfig:
    enabled: bool = False
    db_path: str | None = None
    core_max_chars: int = 2000
    recall_max_events: int = 20
    retrieval_k: int = 8
    use_fts: str = "auto"
    final_memory_enabled: bool = True
    final_memory_filename_md: str = "final_memory_for_paper.md"
    final_memory_filename_json: str = "final_memory_for_paper.json"
    redact_secrets: bool = True
    memory_budget_chars: int = 4000
    root_branch_id: str | None = None
    run_id: str | None = None
    workspace_root: str | None = None
    ai_scientist_root: str | None = None
    phase_mode: str | None = None
    memory_log_dir: str | None = None
    memory_log_enabled: bool = True
    memory_log_max_chars: int = 400
    use_llm_compression: bool = False
    compression_model: str = "gpt-4o-mini"
    max_compression_iterations: int = 3
    datasets_tested_budget_chars: int = 1500
    metrics_extraction_budget_chars: int = 1500
    plotting_code_budget_chars: int = 2000
    plot_selection_budget_chars: int = 1000
    vlm_analysis_budget_chars: int = 1000
    node_summary_budget_chars: int = 2000
    parse_metrics_budget_chars: int = 2000
    archival_snippet_budget_chars: int = 6000
    results_budget_chars: int = 4000
    writeup_recall_limit: int = 10
    writeup_archival_limit: int = 10
    writeup_core_value_max_chars: int = 500
    writeup_recall_text_max_chars: int = 300
    writeup_archival_text_max_chars: int = 400
    # Memory Pressure Management (MemGPT-style)
    auto_consolidate: bool = True
    consolidation_trigger: str = "high"
    recall_consolidation_threshold: float = 1.5
    pressure_thresholds: dict[str, float] = field(default_factory=dict)
    # Multi-turn memory read flow
    max_memory_read_rounds: int = 2


@dataclass
class ExperimentConfig:
    num_syn_datasets: int
    dataset_source: str = "auto"


@dataclass
class Config(Hashable):
    data_dir: Path
    log_dir: Path
    workspace_dir: Path

    copy_data: bool

    exp_name: str

    exec: ExecConfig
    generate_report: bool
    report: StageConfig
    agent: AgentConfig
    experiment: ExperimentConfig

    # Optional fields (must come after required fields)
    preprocess_data: bool = False
    desc_file: Path | None = None
    goal: str | None = None
    eval: str | None = None
    debug: DebugConfig = field(default_factory=DebugConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)


def _auto_detect_language(cfg: Config) -> str:
    """Auto-detect programming language from task description and data directory."""
    language_score = {"python": 0, "cpp": 0, "c": 0}

    # Check task description
    task_desc = ""
    if cfg.desc_file is not None and Path(cfg.desc_file).exists():
        with open(cfg.desc_file, "r", encoding="utf-8") as f:
            task_desc = f.read().lower()
    elif cfg.goal is not None:
        task_desc = cfg.goal.lower()

    # Language keywords in description
    if any(keyword in task_desc for keyword in ["c++", "cpp", "cxx"]):
        language_score["cpp"] += 3
    if any(keyword in task_desc for keyword in ["python", ".py", "numpy", "pandas", "torch", "tensorflow"]):
        language_score["python"] += 3
    if " c " in task_desc or task_desc.startswith("c ") or task_desc.endswith(" c"):
        language_score["c"] += 2

    # Check data directory for code files
    if cfg.data_dir and Path(cfg.data_dir).exists():
        data_path = Path(cfg.data_dir)
        file_counts = {
            "python": len(list(data_path.rglob("*.py"))),
            "cpp": len(list(data_path.rglob("*.cpp"))) + len(list(data_path.rglob("*.cxx"))) + len(list(data_path.rglob("*.cc"))),
            "c": len(list(data_path.rglob("*.c")))
        }

        for lang, count in file_counts.items():
            language_score[lang] += count * 2

    # Select language with highest score
    detected_lang = max(language_score.items(), key=lambda x: x[1])[0]

    # Default to python if all scores are 0
    if language_score[detected_lang] == 0:
        detected_lang = "python"

    return detected_lang


def _get_next_logindex(dir: Path) -> int:
    """Get the next available index for a log directory."""
    max_index = -1
    for p in dir.iterdir():
        try:
            if (current_index := int(p.name.split("-")[0])) > max_index:
                max_index = current_index
        except ValueError:
            pass
    print("max_index: ", max_index)
    return max_index + 1


def _load_cfg(
    path: Path = Path(__file__).parent / "config.yaml", use_cli_args=False
) -> Config:
    cfg = OmegaConf.load(path)
    if use_cli_args:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_cli())
    return cfg


def load_cfg(path: Path = Path(__file__).parent / "config.yaml") -> Config:
    """Load config from .yaml file and CLI args, and set up logging directory."""
    return prep_cfg(_load_cfg(path))


def prep_cfg(cfg: Config):
    if cfg.data_dir is None:
        raise ValueError("`data_dir` must be provided.")

    if cfg.desc_file is None and cfg.goal is None:
        raise ValueError(
            "You must provide either a description of the task goal (`goal=...`) or a path to a plaintext file containing the description (`desc_file=...`)."
        )

    if cfg.data_dir.startswith("example_tasks/"):
        cfg.data_dir = Path(__file__).parent.parent / cfg.data_dir
    cfg.data_dir = Path(cfg.data_dir).resolve()

    if cfg.desc_file is not None:
        cfg.desc_file = Path(cfg.desc_file).resolve()

    top_log_dir = Path(cfg.log_dir).resolve()
    top_log_dir.mkdir(parents=True, exist_ok=True)

    top_workspace_dir = Path(cfg.workspace_dir).resolve()
    top_workspace_dir.mkdir(parents=True, exist_ok=True)

    # generate experiment name and prefix with consecutive index
    ind = max(_get_next_logindex(top_log_dir), _get_next_logindex(top_workspace_dir))
    cfg.exp_name = cfg.exp_name or coolname.generate_slug(3)
    cfg.exp_name = f"{ind}-{cfg.exp_name}"

    cfg.log_dir = (top_log_dir / cfg.exp_name).resolve()
    cfg.workspace_dir = (top_workspace_dir / cfg.exp_name).resolve()

    # validate the config
    cfg_schema: Config = OmegaConf.structured(Config)
    cfg = OmegaConf.merge(cfg_schema, cfg)

    if cfg.agent.type not in ["parallel", "sequential"]:
        raise ValueError("agent.type must be either 'parallel' or 'sequential'")

    set_persona_role(getattr(cfg.agent, "role_description", None))
    cfg.exec.phase_mode = str(getattr(cfg.exec, "phase_mode", "split")).lower()
    if cfg.exec.phase_mode not in {"split", "single"}:
        raise ValueError("exec.phase_mode must be either 'split' or 'single'")

    # Auto-detect language if set to 'auto'
    if cfg.exec.language.lower() == "auto":
        cfg.exec.language = _auto_detect_language(cfg)
        logger.info(f"Auto-detected language: {cfg.exec.language}")

        # Adjust agent_file_name extension based on detected language
        # Note: This only applies to phase_mode="single". In "split" mode, the extension is ignored.
        if cfg.exec.phase_mode == "single":
            agent_file = Path(cfg.exec.agent_file_name)
            stem = agent_file.stem
            if cfg.exec.language == "python" and not agent_file.suffix == ".py":
                cfg.exec.agent_file_name = f"{stem}.py"
            elif cfg.exec.language == "cpp" and agent_file.suffix not in [".cpp", ".cxx", ".cc"]:
                cfg.exec.agent_file_name = f"{stem}.cpp"
            elif cfg.exec.language == "c" and agent_file.suffix != ".c":
                cfg.exec.agent_file_name = f"{stem}.c"

    return cast(Config, cfg)


def print_cfg(cfg: Config) -> None:
    rich.print(Syntax(OmegaConf.to_yaml(cfg), "yaml", theme="paraiso-dark"))


def load_task_desc(cfg: Config):
    """Load task description from markdown file or config str."""

    # either load the task description from a file
    if cfg.desc_file is not None:
        if not (cfg.goal is None and cfg.eval is None):
            logger.warning(
                "Ignoring goal and eval args because task description file is provided."
            )

        with open(cfg.desc_file) as f:
            return f.read()

    # or generate it from the goal and eval args
    if cfg.goal is None:
        raise ValueError(
            "`goal` (and optionally `eval`) must be provided if a task description file is not provided."
        )

    task_desc = {"Task goal": cfg.goal}
    if cfg.eval is not None:
        task_desc["Task evaluation"] = cfg.eval
    print(task_desc)
    return task_desc


def prep_agent_workspace(cfg: Config):
    """Setup the agent's workspace and preprocess data if necessary."""
    (cfg.workspace_dir / "input").mkdir(parents=True, exist_ok=True)
    (cfg.workspace_dir / "working").mkdir(parents=True, exist_ok=True)

    resources_cfg = None
    resources_path = getattr(cfg.exec, "resources", None)
    if resources_path:
        try:
            resources_cfg = load_resources(resolve_resources_path(resources_path))
        except Exception as exc:
            logger.warning("Failed to load resources from %s: %s", resources_path, exc)

    if getattr(cfg, "copy_data", False) and resources_cfg:
        data_dir = Path(cfg.data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        if not any(data_dir.iterdir()):
            try:
                local_resources = resources_cfg.local
                if len(local_resources) == 1:
                    copytree(
                        Path(local_resources[0].host_path),
                        data_dir,
                        use_symlinks=False,
                    )
                elif local_resources:
                    for res in local_resources:
                        dest_name = res.name or Path(res.host_path).name or "resource"
                        dest_dir = data_dir / dest_name
                        dest_dir.mkdir(parents=True, exist_ok=False)
                        copytree(Path(res.host_path), dest_dir, use_symlinks=False)
            except Exception as exc:
                logger.warning(
                    "Failed to copy local resources into data_dir (%s): %s",
                    cfg.data_dir,
                    exc,
                )
        else:
            logger.warning("Data dir %s is not empty; skipping resource copy", data_dir)

    if resources_cfg:
        try:
            staged = stage_resource_items(
                resources_cfg,
                cfg.workspace_dir / "resources",
                classes=("template", "setup", "document"),
            )
            if staged:
                logger.info("Staged %d resource item(s) into workspace", len(staged))
        except Exception as exc:
            logger.warning("Failed to stage resource items into workspace: %s", exc)

    copytree(cfg.data_dir, cfg.workspace_dir / "input", use_symlinks=not cfg.copy_data)
    if cfg.preprocess_data:
        preproc_data(cfg.workspace_dir / "input")
