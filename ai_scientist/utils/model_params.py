OPENAI_COMPLETION_PREFIXES = (
    "gpt-4o",
    "gpt-4.1",
    "gpt-5",
    "o1",
    "o3",
)


def uses_max_completion_tokens(model: str | None) -> bool:
    """
    OpenAI's newer chat models (4o/4.1/5/o1/o3 families) expect `max_completion_tokens`
    instead of `max_tokens`. Older backends (Anthropic, Ollama, DeepSeek, etc.) still
    rely on `max_tokens`, so we switch keys based on the model prefix.
    """
    normalized = (model or "").lower()
    return normalized.startswith(OPENAI_COMPLETION_PREFIXES)


def build_token_params(
    model: str, n_tokens: int | None, *, default_key: str = "max_tokens"
) -> dict[str, int]:
    """
    Return the correct max token kwarg for the given model.

    Args:
        model: Model name string.
        n_tokens: Token budget to request. If None, returns an empty dict.
        default_key: Parameter name for backends that still expect `max_tokens`.
    """
    if n_tokens is None:
        return {}
    key = "max_completion_tokens" if uses_max_completion_tokens(model) else default_key
    return {key: n_tokens}
