"""Functions for getting responses from VLMs."""

from typing import Any

import backoff
import openai

from ai_scientist.vlm.constants import AVAILABLE_VLMS, MAX_NUM_TOKENS
from ai_scientist.vlm.utils import encode_image_to_base64
from ai_scientist.utils.token_tracker import track_token_usage
from ai_scientist.utils.model_params import build_token_params


@track_token_usage
def make_llm_call(client, model, temperature, system_message, prompt):
    """Make a low-level LLM API call for VLM.

    Args:
        client: API client instance.
        model: Model identifier string.
        temperature: Sampling temperature.
        system_message: System prompt.
        prompt: Message history/prompt.

    Returns:
        Raw API response.

    Raises:
        ValueError: If the model is not supported.
    """
    if model.startswith("ollama/"):
        return client.chat.completions.create(
            model=model.replace("ollama/", ""),
            messages=[
                {"role": "system", "content": system_message},
                *prompt,
            ],
            temperature=temperature,
            max_tokens=MAX_NUM_TOKENS,
            n=1,
            stop=None,
            seed=0,
        )
    elif "gpt" in model:
        return client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                *prompt,
            ],
            temperature=temperature,
            **build_token_params(model, MAX_NUM_TOKENS),
            n=1,
            stop=None,
            seed=0,
        )
    elif "o1" in model or "o3" in model:
        return client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": system_message},
                *prompt,
            ],
            temperature=1,
            n=1,
            seed=0,
        )
    else:
        raise ValueError(f"Model {model} not supported.")


@track_token_usage
def make_vlm_call(client, model, temperature, system_message, prompt):
    """Make a VLM API call with vision capabilities.

    Args:
        client: API client instance.
        model: Model identifier string.
        temperature: Sampling temperature.
        system_message: System prompt.
        prompt: Message history/prompt with images.

    Returns:
        Raw API response.

    Raises:
        ValueError: If the model is not supported.
    """
    if model.startswith("ollama/"):
        return client.chat.completions.create(
            model=model.replace("ollama/", ""),
            messages=[
                {"role": "system", "content": system_message},
                *prompt,
            ],
            temperature=temperature,
            max_tokens=MAX_NUM_TOKENS,
        )
    elif "gpt" in model:
        return client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                *prompt,
            ],
            temperature=temperature,
            **build_token_params(model, MAX_NUM_TOKENS),
        )
    elif "o1" in model or "o3" in model:
        return client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": system_message},
                *prompt,
            ],
        )
    else:
        raise ValueError(f"Model {model} not supported.")


@backoff.on_exception(
    backoff.expo,
    (
        openai.RateLimitError,
        openai.APITimeoutError,
    ),
)
def get_response_from_vlm(
    msg: str,
    image_paths: str | list[str],
    client: Any,
    model: str,
    system_message: str,
    print_debug: bool = False,
    msg_history: list[dict[str, Any]] | None = None,
    temperature: float = 0.7,
    max_images: int = 25,
) -> tuple[str, list[dict[str, Any]]]:
    """Get response from vision-language model.

    Args:
        msg: Text message to send.
        image_paths: Path(s) to image file(s).
        client: VLM client instance.
        model: Name of model to use.
        system_message: System prompt.
        print_debug: Whether to print debug info.
        msg_history: Previous message history.
        temperature: Sampling temperature.
        max_images: Maximum number of images to include.

    Returns:
        Tuple of (response string, message history).

    Raises:
        ValueError: If the model is not supported.
    """
    if msg_history is None:
        msg_history = []

    if model in AVAILABLE_VLMS:
        # Convert single image path to list for consistent handling
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        # Create content list starting with the text message
        content = [{"type": "text", "text": msg}]

        # Add each image to the content list
        for image_path in image_paths[:max_images]:
            base64_image = encode_image_to_base64(image_path)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "low",
                    },
                }
            )
        # Construct message with all images
        new_msg_history = msg_history + [{"role": "user", "content": content}]

        response = make_vlm_call(
            client,
            model,
            temperature,
            system_message=system_message,
            prompt=new_msg_history,
        )

        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    else:
        raise ValueError(f"Model {model} not supported.")

    if print_debug:
        print()
        print("*" * 20 + " VLM START " + "*" * 20)
        for j, msg in enumerate(new_msg_history):
            print(f'{j}, {msg["role"]}: {msg["content"]}')
        print(content)
        print("*" * 21 + " VLM END " + "*" * 21)
        print()

    return content, new_msg_history


@backoff.on_exception(
    backoff.expo,
    (
        openai.RateLimitError,
        openai.APITimeoutError,
    ),
)
def get_batch_responses_from_vlm(
    msg: str,
    image_paths: str | list[str],
    client: Any,
    model: str,
    system_message: str,
    print_debug: bool = False,
    msg_history: list[dict[str, Any]] | None = None,
    temperature: float = 0.7,
    n_responses: int = 1,
    max_images: int = 200,
) -> tuple[list[str], list[list[dict[str, Any]]]]:
    """Get multiple responses from vision-language model for the same input.

    Args:
        msg: Text message to send.
        image_paths: Path(s) to image file(s).
        client: OpenAI client instance.
        model: Name of model to use.
        system_message: System prompt.
        print_debug: Whether to print debug info.
        msg_history: Previous message history.
        temperature: Sampling temperature.
        n_responses: Number of responses to generate.
        max_images: Maximum number of images to include.

    Returns:
        Tuple of (list of response strings, list of message histories).

    Raises:
        ValueError: If the model is not supported.
    """
    if msg_history is None:
        msg_history = []

    if model in AVAILABLE_VLMS:
        # Convert single image path to list
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        # Create content list with text and images
        content = [{"type": "text", "text": msg}]
        for image_path in image_paths[:max_images]:
            base64_image = encode_image_to_base64(image_path)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "low",
                    },
                }
            )

        # Construct message with all images
        new_msg_history = msg_history + [{"role": "user", "content": content}]

        if model.startswith("ollama/"):
            response = client.chat.completions.create(
                model=model.replace("ollama/", ""),
                messages=[
                    {"role": "system", "content": system_message},
                    *new_msg_history,
                ],
                temperature=temperature,
                max_tokens=MAX_NUM_TOKENS,
                n=n_responses,
                seed=0,
            )
        else:
            # Get multiple responses
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_message},
                    *new_msg_history,
                ],
                temperature=temperature,
                **build_token_params(model, MAX_NUM_TOKENS),
                n=n_responses,
                seed=0,
            )

        # Extract content from all responses
        contents = [r.message.content for r in response.choices]
        new_msg_histories = [
            new_msg_history + [{"role": "assistant", "content": c}] for c in contents
        ]
    else:
        raise ValueError(f"Model {model} not supported.")

    if print_debug:
        # Just print the first response
        print()
        print("*" * 20 + " VLM START " + "*" * 20)
        for j, msg in enumerate(new_msg_histories[0]):
            print(f'{j}, {msg["role"]}: {msg["content"]}')
        print(contents[0])
        print("*" * 21 + " VLM END " + "*" * 21)
        print()

    return contents, new_msg_histories
