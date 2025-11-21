from .backend import query
from .journal import Journal
from .utils.config import StageConfig
from ai_scientist.prompt_loader import load_prompt_json


def journal2report(journal: Journal, task_desc: dict, rcfg: StageConfig):
    """
    Generate a report from a journal, the report will be in markdown format.
    """
    report_input = journal.generate_summary(include_code=True)
    system_prompt_dict = load_prompt_json(
        "treesearch/journal2report/system_prompt.json"
    )
    context_prompt = (
        f"Here is the research journal of the agent: <journal>{report_input}<\\journal>, "
        f"and the research idea description is: <research_proposal>{task_desc}<\\research_proposal>."
    )
    return query(
        system_message=system_prompt_dict,
        user_message=context_prompt,
        model=rcfg.model,
        temperature=rcfg.temp,
        max_tokens=4096,
    )
