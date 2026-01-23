"""Citation gathering functionality for the output module."""

import json
import os.path as osp
import re
import shutil
import traceback
import unicodedata

from ai_scientist.llm import (
    get_response_from_llm,
    extract_json_between_markers,
    create_client,
)
from ai_scientist.tools.semantic_scholar import search_for_papers
from ai_scientist.prompt_loader import load_prompt

# Load prompt templates
CITATION_SYSTEM_MSG_TEMPLATE = load_prompt("output/writeup/citation/system_message")
CITATION_FIRST_PROMPT_TEMPLATE = load_prompt("output/writeup/citation/first_prompt")
CITATION_SECOND_PROMPT_TEMPLATE = load_prompt("output/writeup/citation/second_prompt")


def remove_accents_and_clean(s: str) -> str:
    """Remove accents and clean a string for BibTeX keys.

    Args:
        s: Input string with possible accents.

    Returns:
        Cleaned string suitable for BibTeX.
    """
    # Normalize to separate accents
    nfkd_form = unicodedata.normalize("NFKD", s)
    # Remove non-ASCII characters
    ascii_str = nfkd_form.encode("ASCII", "ignore").decode("ascii")
    # Remove anything but letters, digits, underscores, colons, dashes, @, {, }, and commas
    ascii_str = re.sub(r"[^a-zA-Z0-9:_@\{\},-]+", "", ascii_str)
    # Convert to lowercase
    ascii_str = ascii_str.lower()
    return ascii_str


def get_citation_addition(
    client, model, context, current_round, total_rounds, idea_text
):
    """Get a citation addition from the LLM.

    Args:
        client: LLM client instance.
        model: LLM model name.
        context: Tuple of (report, citations).
        current_round: Current citation gathering round.
        total_rounds: Total number of rounds.
        idea_text: Research idea text.

    Returns:
        Tuple of (reference_prompt, done) where reference_prompt is the
        BibTeX entry to add (or None), and done indicates if citation
        gathering should stop.
    """
    report, citations = context
    msg_history = []

    try:
        text, msg_history = get_response_from_llm(
            prompt=CITATION_FIRST_PROMPT_TEMPLATE.format(
                current_round=current_round + 1,
                total_rounds=total_rounds,
                Idea=idea_text,
                report=report,
                citations=citations,
            ),
            client=client,
            model=model,
            system_message=CITATION_SYSTEM_MSG_TEMPLATE.format(total_rounds=total_rounds),
            msg_history=msg_history,
            print_debug=False,
        )
        if "No more citations needed" in text:
            print("No more citations needed.")
            return None, True

        json_output = extract_json_between_markers(text)
        assert json_output is not None, "Failed to extract JSON from LLM output"
        query = json_output["Query"]
        papers = search_for_papers(query)
    except Exception:
        print("EXCEPTION in get_citation_addition (initial search):")
        print(traceback.format_exc())
        return None, False

    if papers is None:
        print("No papers found.")
        return None, False

    paper_strings = []
    for i, paper in enumerate(papers):
        paper_strings.append(
            "{i}: {title}. {authors}. {venue}, {year}.\nAbstract: {abstract}".format(
                i=i,
                title=paper["title"],
                authors=paper["authors"],
                venue=paper["venue"],
                year=paper["year"],
                abstract=paper["abstract"],
            )
        )
    papers_str = "\n\n".join(paper_strings)

    try:
        text, msg_history = get_response_from_llm(
            prompt=CITATION_SECOND_PROMPT_TEMPLATE.format(
                papers=papers_str,
                current_round=current_round + 1,
                total_rounds=total_rounds,
            ),
            client=client,
            model=model,
            system_message=CITATION_SYSTEM_MSG_TEMPLATE.format(total_rounds=total_rounds),
            msg_history=msg_history,
            print_debug=False,
        )
        if "Do not add any" in text:
            print("Do not add any.")
            return None, False

        json_output = extract_json_between_markers(text)
        assert json_output is not None, "Failed to extract JSON from LLM output"
        desc = json_output["Description"]
        selected_papers = str(json_output["Selected"])

        if selected_papers != "[]":
            selected_indices = []
            for x in selected_papers.strip("[]").split(","):
                x_str = x.strip().strip('"').strip("'")
                if x_str:
                    selected_indices.append(int(x_str))
            assert all(
                [0 <= i < len(papers) for i in selected_indices]
            ), "Invalid paper index"
            bibtexs = [papers[i]["citationStyles"]["bibtex"] for i in selected_indices]

            cleaned_bibtexs = []
            for bibtex in bibtexs:
                newline_index = bibtex.find("\n")
                cite_key_line = bibtex[:newline_index]
                cite_key_line = remove_accents_and_clean(cite_key_line)
                cleaned_bibtexs.append(cite_key_line + bibtex[newline_index:])
            bibtexs = cleaned_bibtexs

            bibtex_string = "\n".join(bibtexs)
        else:
            return None, False

    except Exception:
        print("EXCEPTION in get_citation_addition (selecting papers):")
        print(traceback.format_exc())
        return None, False

    references_format = """% {description}
{bibtex}"""

    references_prompt = references_format.format(bibtex=bibtex_string, description=desc)
    return references_prompt, False


def gather_citations(base_folder: str, num_cite_rounds: int = 20, small_model: str = "gpt-4o-2024-05-13") -> str | None:
    """Gather citations for a paper.

    Args:
        base_folder: Path to project folder.
        num_cite_rounds: Maximum number of citation gathering rounds.
        small_model: Model to use for citation collection.

    Returns:
        The gathered citations text, or None if failed.
    """
    # Import here to avoid circular imports
    from ai_scientist.output.writeup import load_idea_text, load_exp_summaries, filter_experiment_summaries

    current_round = 0
    citations_text = ""

    latex_folder = osp.join(base_folder, "latex")

    # Prepare a new fresh latex folder
    if not osp.exists(osp.join(latex_folder, "template.tex")):
        shutil.copytree(
            "ai_scientist/blank_latex", latex_folder, dirs_exist_ok=True
        )

    writeup_file = osp.join(latex_folder, "template.tex")
    with open(writeup_file, "r") as f:
        writeup_text = f.read()

    try:
        # Load idea text and summaries
        idea_text = load_idea_text(base_folder)
        exp_summaries = load_exp_summaries(base_folder)
        filtered_summaries = filter_experiment_summaries(
            exp_summaries, step_name="citation_gathering"
        )
        combined_summaries_str = json.dumps(filtered_summaries, indent=2)

        # Run small model for citation additions
        client, client_model = create_client(small_model)
        for round_idx in range(num_cite_rounds):
            print(f"Citation gathering round {round_idx + 1}/{num_cite_rounds}")
            with open(writeup_file, "r") as f:
                writeup_text = f.read()
            try:
                references_bib = re.search(
                    r"\\begin{filecontents}{references.bib}(.*?)\\end{filecontents}",
                    writeup_text,
                    re.DOTALL,
                )
                if references_bib is None:
                    raise ValueError("No references.bib found in template.tex")
                citations_text = references_bib.group(1)
                context_for_citation = (combined_summaries_str, citations_text)

                addition, done = get_citation_addition(
                    client,
                    client_model,
                    context_for_citation,
                    round_idx,
                    num_cite_rounds,
                    idea_text,
                )
                if done:
                    break

                if addition is not None:
                    # Simple check to avoid duplicating the same title
                    title_match = re.search(r" title = {(.*?)}", addition)
                    if title_match:
                        new_title = title_match.group(1).lower()
                        existing_titles = re.findall(
                            r" title = {(.*?)}", citations_text
                        )
                        existing_titles = [t.lower() for t in existing_titles]
                        if new_title not in existing_titles:
                            pattern_end = r"\end{filecontents}"
                            revised = writeup_text.replace(
                                pattern_end, f"\n{addition}{pattern_end}"
                            )
                            with open(writeup_file, "w") as fo:
                                fo.write(revised)
            except Exception:
                print("EXCEPTION in gather_citations:")
                print(traceback.format_exc())
                continue
        return citations_text if citations_text else None

    except Exception:
        print("EXCEPTION in gather_citations:")
        print(traceback.format_exc())
        return citations_text if citations_text else None
