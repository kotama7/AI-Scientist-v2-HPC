"""Constants for VLM module."""

MAX_NUM_TOKENS = 4096

AVAILABLE_VLMS = [
    "gpt-4o-2024-05-13",
    "gpt-4o-2024-08-06",
    "gpt-4o-2024-11-20",
    "gpt-4o-mini-2024-07-18",
    "gpt-4o",
    "gpt-5.2",
    "o3-mini",
    # Ollama models
    # llama4
    "ollama/llama4:16x17b",
    # mistral
    "ollama/mistral-small3.2:24b",
    # qwen
    "ollama/qwen2.5vl:32b",
    "ollama/z-uo/qwen2.5vl_tools:32b",
]
