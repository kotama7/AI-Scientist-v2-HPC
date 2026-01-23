"""Plot aggregation - backward compatibility wrapper.

This module re-exports functionality from the new ai_scientist.output module
and provides the CLI interface for backward compatibility.

New code should import directly from ai_scientist.output:
    from ai_scientist.output import aggregate_plots
"""

import argparse

# Re-export from new modules
from ai_scientist.output import (
    aggregate_plots,
    extract_code_snippet,
    run_aggregator_script,
    build_aggregator_prompt,
)

__all__ = [
    "aggregate_plots",
    "extract_code_snippet",
    "run_aggregator_script",
    "build_aggregator_prompt",
]


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
