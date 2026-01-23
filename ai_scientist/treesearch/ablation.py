"""Ablation and hyperparameter tuning configuration classes.

This module provides data classes for tracking ablation experiments
and hyperparameter tuning configurations.
"""

from ai_scientist.treesearch.journal import Node


class AblationConfig:
    """Track state of ablation experiments.

    Attributes:
        name: Name of the ablation.
        description: Description of what is being ablated.
        code: Code implementing the ablation.
        base_node: Node this ablation is based on.
        attempts: Number of attempts made.
        max_attempts: Maximum number of retry attempts.
        last_error: Last error encountered.
        completed: Whether the ablation completed successfully.
        current_node: Current node being processed.
    """

    def __init__(self, name: str, description: str, code: str, base_node: Node):
        self.name = name
        self.description = description
        self.code = code
        self.base_node = base_node
        self.attempts = 0
        self.max_attempts = 3  # Maximum number of retry attempts
        self.last_error = None
        self.completed = False
        self.current_node = None


class AblationIdea:
    """Ablation idea for experiment design.

    Attributes:
        name: Name of the ablation idea.
        description: Description of the ablation.
    """

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description


class HyperparamTuningIdea:
    """Hyperparameter tuning idea.

    Attributes:
        name: Name of the tuning idea.
        description: Description of the tuning approach.
    """

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
