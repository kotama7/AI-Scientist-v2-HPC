from functools import wraps
from typing import Dict, Optional, List
import tiktoken
from collections import defaultdict
import asyncio
from datetime import datetime
import logging


class TokenTracker:
    def __init__(self):
        """
        Token counts for prompt, completion, reasoning, and cached.
        Reasoning tokens are included in completion tokens.
        Cached tokens are included in prompt tokens.
        Also tracks prompts, responses, and timestamps.
        We assume we get these from the LLM response, and we don't count
        the tokens by ourselves.
        """
        self.token_counts = defaultdict(
            lambda: {"prompt": 0, "completion": 0, "reasoning": 0, "cached": 0}
        )
        self.interactions = defaultdict(list)

        self.MODEL_PRICES = {
            "gpt-5.2": {
                "prompt": 1.75 / 1000000,  # $1.75 per 1M tokens
                "cached": 0.175 / 1000000,  # $0.175 per 1M tokens
                "completion": 14 / 1000000,  # $14.00 per 1M tokens
            },
            "gpt-4o-2024-11-20": {
                "prompt": 2.5 / 1000000,  # $2.50 per 1M tokens
                "cached": 1.25 / 1000000,  # $1.25 per 1M tokens
                "completion": 10 / 1000000,  # $10.00 per 1M tokens
            },
            "gpt-4o-2024-08-06": {
                "prompt": 2.5 / 1000000,  # $2.50 per 1M tokens
                "cached": 1.25 / 1000000,  # $1.25 per 1M tokens
                "completion": 10 / 1000000,  # $10.00 per 1M tokens
            },
            "gpt-4o-2024-05-13": {  # this ver does not support cached tokens
                "prompt": 5.0 / 1000000,  # $5.00 per 1M tokens
                "completion": 15 / 1000000,  # $15.00 per 1M tokens
            },
            "gpt-4o-mini-2024-07-18": {
                "prompt": 0.15 / 1000000,  # $0.15 per 1M tokens
                "cached": 0.075 / 1000000,  # $0.075 per 1M tokens
                "completion": 0.6 / 1000000,  # $0.60 per 1M tokens
            },
            "o1-2024-12-17": {
                "prompt": 15 / 1000000,  # $15.00 per 1M tokens
                "cached": 7.5 / 1000000,  # $7.50 per 1M tokens
                "completion": 60 / 1000000,  # $60.00 per 1M tokens
            },
            "o1-preview-2024-09-12": {
                "prompt": 15 / 1000000,  # $15.00 per 1M tokens
                "cached": 7.5 / 1000000,  # $7.50 per 1M tokens
                "completion": 60 / 1000000,  # $60.00 per 1M tokens
            },
            "o3-mini-2025-01-31": {
                "prompt": 1.1 / 1000000,  # $1.10 per 1M tokens
                "cached": 0.55 / 1000000,  # $0.55 per 1M tokens
                "completion": 4.4 / 1000000,  # $4.40 per 1M tokens
            },
            # Anthropic Claude models
            "claude-3-5-sonnet-20241022": {
                "prompt": 3.0 / 1000000,  # $3.00 per 1M tokens
                "cached": 0.3 / 1000000,  # $0.30 per 1M tokens
                "completion": 15.0 / 1000000,  # $15.00 per 1M tokens
            },
            "claude-3-5-haiku-20241022": {
                "prompt": 1.0 / 1000000,  # $1.00 per 1M tokens
                "cached": 0.1 / 1000000,  # $0.10 per 1M tokens
                "completion": 5.0 / 1000000,  # $5.00 per 1M tokens
            },
            "claude-3-opus-20240229": {
                "prompt": 15.0 / 1000000,  # $15.00 per 1M tokens
                "cached": 1.5 / 1000000,  # $1.50 per 1M tokens
                "completion": 75.0 / 1000000,  # $75.00 per 1M tokens
            },
            # Ollama (local, no cost)
            "ollama": {
                "prompt": 0.0,
                "cached": 0.0,
                "completion": 0.0,
            },
            # DeepSeek
            "deepseek-coder": {
                "prompt": 0.14 / 1000000,  # $0.14 per 1M tokens
                "completion": 0.28 / 1000000,  # $0.28 per 1M tokens
            },
        }

    def _resolve_price_model(self, model: str) -> Optional[str]:
        """Resolve a model name to a pricing key, falling back to prefix matches."""
        if model in self.MODEL_PRICES:
            return model
        if "-" not in model:
            return None
        parts = model.split("-")
        for i in range(len(parts) - 1, 0, -1):
            candidate = "-".join(parts[:i])
            if candidate in self.MODEL_PRICES:
                return candidate
        return None

    def add_tokens(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        reasoning_tokens: int,
        cached_tokens: int,
    ):
        self.token_counts[model]["prompt"] += prompt_tokens
        self.token_counts[model]["completion"] += completion_tokens
        self.token_counts[model]["reasoning"] += reasoning_tokens
        self.token_counts[model]["cached"] += cached_tokens

    def add_interaction(
        self,
        model: str,
        system_message: str,
        prompt: str,
        response: str,
        timestamp: datetime,
    ):
        """Record a single interaction with the model."""
        self.interactions[model].append(
            {
                "system_message": system_message,
                "prompt": prompt,
                "response": response,
                "timestamp": timestamp,
            }
        )

    def get_interactions(self, model: Optional[str] = None) -> Dict[str, List[Dict]]:
        """Get all interactions, optionally filtered by model."""
        if model:
            return {model: self.interactions[model]}
        return dict(self.interactions)

    def reset(self):
        """Reset all token counts and interactions."""
        self.token_counts = defaultdict(
            lambda: {"prompt": 0, "completion": 0, "reasoning": 0, "cached": 0}
        )
        self.interactions = defaultdict(list)
        # self._encoders = {}

    def calculate_cost(self, model: str) -> float:
        """Calculate the cost for a specific model based on token usage."""
        price_model = self._resolve_price_model(model)
        if price_model is None:
            logging.warning(f"Price information not available for model {model}")
            return 0.0

        prices = self.MODEL_PRICES[price_model]
        tokens = self.token_counts[model]

        # Calculate cost for prompt and completion tokens
        if "cached" in prices:
            prompt_cost = (tokens["prompt"] - tokens["cached"]) * prices["prompt"]
            cached_cost = tokens["cached"] * prices["cached"]
        else:
            prompt_cost = tokens["prompt"] * prices["prompt"]
            cached_cost = 0
        completion_cost = tokens["completion"] * prices["completion"]

        return prompt_cost + cached_cost + completion_cost

    def get_summary(self) -> Dict[str, Dict[str, int]]:
        # return dict(self.token_counts)
        """Get summary of token usage and costs for all models."""
        summary = {}
        for model, tokens in self.token_counts.items():
            summary[model] = {
                "tokens": tokens.copy(),
                "cost (USD)": self.calculate_cost(model),
            }
        return summary


# Global token tracker instance
token_tracker = TokenTracker()


def track_token_usage(func):
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        prompt = kwargs.get("prompt")
        system_message = kwargs.get("system_message")
        if not prompt and not system_message:
            raise ValueError(
                "Either 'prompt' or 'system_message' must be provided for token tracking"
            )

        logging.info("args: ", args)
        logging.info("kwargs: ", kwargs)

        result = await func(*args, **kwargs)
        model = result.model
        timestamp = result.created

        if hasattr(result, "usage") and result.usage.completion_tokens_details is not None:
            token_tracker.add_tokens(
                model,
                result.usage.prompt_tokens,
                result.usage.completion_tokens,
                result.usage.completion_tokens_details.reasoning_tokens,
                (
                    result.usage.prompt_tokens_details.cached_tokens
                    if hasattr(result.usage, "prompt_tokens_details")
                    else 0
                ),
            )
            # Add interaction details
            token_tracker.add_interaction(
                model,
                system_message,
                prompt,
                result.choices[
                    0
                ].message.content,  # Assumes response is in content field
                timestamp,
            )
        return result

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        prompt = kwargs.get("prompt")
        system_message = kwargs.get("system_message")
        if not prompt and not system_message:
            raise ValueError(
                "Either 'prompt' or 'system_message' must be provided for token tracking"
            )
        result = func(*args, **kwargs)
        model = result.model
        timestamp = result.created
        logging.info("args: ", args)
        logging.info("kwargs: ", kwargs)

        if hasattr(result, "usage") and result.usage.completion_tokens_details is not None:
            token_tracker.add_tokens(
                model,
                result.usage.prompt_tokens,
                result.usage.completion_tokens,
                result.usage.completion_tokens_details.reasoning_tokens,
                (
                    result.usage.prompt_tokens_details.cached_tokens
                    if hasattr(result.usage, "prompt_tokens_details")
                    else 0
                ),
            )
            # Add interaction details
            token_tracker.add_interaction(
                model,
                system_message,
                prompt,
                result.choices[
                    0
                ].message.content,  # Assumes response is in content field
                timestamp,
            )
        return result

    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper


def track_openai_response(response, system_message=None, prompt=None):
    """Track tokens from OpenAI API response.

    Args:
        response: OpenAI API response object
        system_message: System message used in the call
        prompt: User prompt or message history
    """
    if not hasattr(response, "usage") or response.usage is None:
        logging.warning(
            f"Response from {getattr(response, 'model', 'unknown')} has no usage info"
        )
        return

    model = response.model
    usage = response.usage

    # Extract tokens with safe defaults
    prompt_tokens = usage.prompt_tokens or 0
    completion_tokens = usage.completion_tokens or 0
    reasoning_tokens = (
        getattr(usage.completion_tokens_details, "reasoning_tokens", 0)
        if hasattr(usage, "completion_tokens_details")
        and usage.completion_tokens_details
        else 0
    )
    cached_tokens = (
        getattr(usage.prompt_tokens_details, "cached_tokens", 0)
        if hasattr(usage, "prompt_tokens_details") and usage.prompt_tokens_details
        else 0
    )

    token_tracker.add_tokens(
        model, prompt_tokens, completion_tokens, reasoning_tokens, cached_tokens
    )

    # Add interaction
    if system_message or prompt:
        content = (
            response.choices[0].message.content if response.choices else ""
        )
        timestamp = getattr(response, "created", datetime.now())
        token_tracker.add_interaction(
            model,
            system_message or "",
            str(prompt) if prompt else "",
            content or "",
            timestamp,
        )


def track_anthropic_response(message, model_name, system_message=None, prompt=None):
    """Track tokens from Anthropic API response.

    Args:
        message: Anthropic API message object
        model_name: Model name string
        system_message: System message used in the call
        prompt: User prompt or message history
    """
    if not hasattr(message, "usage") or message.usage is None:
        logging.warning(f"Message from {model_name} has no usage info")
        return

    usage = message.usage

    # Anthropic uses different field names
    prompt_tokens = usage.input_tokens or 0
    completion_tokens = usage.output_tokens or 0
    reasoning_tokens = 0  # Anthropic doesn't provide this
    cached_tokens = getattr(usage, "cache_read_input_tokens", 0) + getattr(
        usage, "cache_creation_input_tokens", 0
    )

    token_tracker.add_tokens(
        model_name, prompt_tokens, completion_tokens, reasoning_tokens, cached_tokens
    )

    # Add interaction
    if system_message or prompt:
        content = (
            message.content[0].text
            if message.content and len(message.content) > 0
            else ""
        )
        timestamp = datetime.now()
        token_tracker.add_interaction(
            model_name,
            system_message or "",
            str(prompt) if prompt else "",
            content,
            timestamp,
        )
