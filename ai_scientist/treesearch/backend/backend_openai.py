import json
import logging
import time

from .utils import FunctionSpec, OutputType, opt_messages_to_list, backoff_create
from funcy import notnone, once, select_values
import openai
from rich import print
from ai_scientist.utils.token_tracker import track_openai_response

logger = logging.getLogger("ai-scientist")


OPENAI_TIMEOUT_EXCEPTIONS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)

def get_ai_client(model: str, max_retries=2) -> openai.OpenAI:
    if model.startswith("ollama/"):
        client = openai.OpenAI(
            base_url="http://localhost:11434/v1", 
            max_retries=max_retries
        )
    else:
        client = openai.OpenAI(max_retries=max_retries)
    return client


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    client = get_ai_client(model_kwargs.get("model"), max_retries=0)
    filtered_kwargs: dict = select_values(notnone, model_kwargs)  # type: ignore

    base_messages = opt_messages_to_list(system_message, user_message)

    if func_spec is not None:
        filtered_kwargs["tools"] = [func_spec.as_openai_tool_dict]
        # force the model to use the function
        filtered_kwargs["tool_choice"] = func_spec.openai_tool_choice_dict

    if filtered_kwargs.get("model", "").startswith("ollama/"):
       filtered_kwargs["model"] = filtered_kwargs["model"].replace("ollama/", "")

    retry_messages: list[dict[str, str]] = []
    # Give the model a couple of extra chances to return a tool call when it
    # accidentally responds without one.
    max_func_retries = 5 if func_spec else 1
    total_req_time = 0.0
    total_in_tokens = 0
    total_out_tokens = 0
    output: OutputType | None = None
    last_completion = None

    for attempt in range(max_func_retries):
        messages = base_messages + retry_messages
        t0 = time.time()
        completion = backoff_create(
            client.chat.completions.create,
            OPENAI_TIMEOUT_EXCEPTIONS,
            messages=messages,
            **filtered_kwargs,
        )
        last_completion = completion
        req_time = time.time() - t0
        total_req_time += req_time

        # Track token usage
        track_openai_response(completion, system_message, user_message)

        choice = completion.choices[0]
        usage = completion.usage
        if usage:
            total_in_tokens += usage.prompt_tokens or 0
            total_out_tokens += usage.completion_tokens or 0

        if func_spec is None:
            output = choice.message.content
            # DEBUG: Log raw response for non-function calls
            logger.info("[OpenAI DEBUG] Raw content (first 500 chars): %s", repr(output[:500]) if output else "EMPTY/NONE")
            logger.info("[OpenAI DEBUG] Full message object: %s", choice.message)
            # Check for reasoning content in newer models (like gpt-5.2, o1, o3)
            if hasattr(choice.message, 'reasoning') and choice.message.reasoning:
                logger.info("[OpenAI DEBUG] Reasoning content found: %s", repr(str(choice.message.reasoning)[:500]))
            if output is None or output == "":
                # Try to get output from other possible attributes
                logger.warning("[OpenAI DEBUG] content is empty/None, checking other attributes...")
                for attr in ['reasoning', 'text', 'response']:
                    if hasattr(choice.message, attr) and getattr(choice.message, attr):
                        logger.info("[OpenAI DEBUG] Found non-empty attribute '%s': %s", attr, repr(str(getattr(choice.message, attr))[:200]))
            break

        if choice.message.tool_calls:
            assert (
                choice.message.tool_calls[0].function.name == func_spec.name
            ), "Function name mismatch"
            try:
                print(f"[cyan]Raw func call response: {choice}[/cyan]")
                output = json.loads(choice.message.tool_calls[0].function.arguments)
            except json.JSONDecodeError as e:
                logger.error(
                    f"Error decoding the function arguments: {choice.message.tool_calls[0].function.arguments}"
                )
                raise e
            break

        error_details_parts = []
        if getattr(choice.message, "reasoning", None):
            reasoning_text = str(choice.message.reasoning).strip()
            if reasoning_text:
                error_details_parts.append(f"Reasoning: {reasoning_text}")
        if choice.message.content:
            content_text = str(choice.message.content).strip()
            if content_text:
                error_details_parts.append(f"Content: {content_text}")
        if not error_details_parts:
            error_details_parts.append(str(choice.message))
        error_text = "\n".join(error_details_parts)
        error_message = (
            f"The assistant response did not include the required tool call "
            f"'{func_spec.name}'. Details:\n{error_text}\n"
            "Please call the tool with valid JSON arguments matching the provided schema."
        )
        logger.warning(error_message)

        if attempt == max_func_retries - 1:
            raise AssertionError(
                f"function_call is empty, it is not a function call: {choice.message}"
            )

        retry_messages.append({"role": "user", "content": error_message})

    assert last_completion is not None
    assert output is not None

    info = {
        "system_fingerprint": last_completion.system_fingerprint,
        "model": last_completion.model,
        "created": last_completion.created,
    }

    return output, total_req_time, total_in_tokens, total_out_tokens, info
