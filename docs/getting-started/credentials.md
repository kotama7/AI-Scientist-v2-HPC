# Credentials

Set only the environment variables required for the models you plan to use.
Example exports:

```bash
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."        # Claude models
export GEMINI_API_KEY="..."           # Gemini models (OpenAI-compatible endpoint)
export OPENROUTER_API_KEY="..."       # Llama 3.1 via OpenRouter
export DEEPSEEK_API_KEY="..."         # deepseek-coder-v2-0724
export HUGGINGFACE_API_KEY="..."      # deepcoder-14b via Hugging Face API
export OLLAMA_API_KEY="..."           # Optional; local Ollama endpoint
export S2_API_KEY="..."               # Optional; Semantic Scholar
```

## Credential flow

<!-- TODO: Generate credentials_flow.png
IMAGE_PROMPT:
Create a 16:9 flow diagram: left cluster of key icons labeled OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, OPENROUTER_API_KEY, DEEPSEEK_API_KEY, HUGGINGFACE_API_KEY, OLLAMA_API_KEY, S2_API_KEY. Arrows converge into a central box "launch_scientist_bfts.py". From there, arrows go to right-side boxes labeled "LLM providers" and optional branches to "Semantic Scholar" and "Hugging Face". Use flat vector style, blue/teal/gray palette, white background, sans-serif labels. Title: "Credential flow". -->

## Where to set secrets

- Shell environment before launching jobs.
- Slurm or batch job scripts.
- Cluster secret management tooling (preferred when available).

Avoid committing secrets to the repo.

## How keys are used

- LLM model selection lives in `bfts_config.yaml` under `agent` and `report`.
- `S2_API_KEY` is only required when the ideation flow uses Semantic Scholar.
- `HUGGINGFACE_API_KEY` is needed for private models or gated downloads.
- `OLLAMA_API_KEY` is only required when using a local Ollama endpoint that
  expects authentication.

## Provider mapping (quick reference)

| Environment variable | Typical usage |
| --- | --- |
| `OPENAI_API_KEY` | OpenAI models in `agent.*` or `report.model` |
| `ANTHROPIC_API_KEY` | Claude models |
| `GEMINI_API_KEY` | Gemini models via OpenAI-compatible endpoint |
| `OPENROUTER_API_KEY` | OpenRouter-hosted models |
| `DEEPSEEK_API_KEY` | DeepSeek models |
| `HUGGINGFACE_API_KEY` | HF Inference API or gated downloads |
| `OLLAMA_API_KEY` | Local Ollama endpoints requiring auth |
| `S2_API_KEY` | Semantic Scholar (ideation enrichment) |

## Validation checks

- `python -c "import os; print('OPENAI_API_KEY' in os.environ)"`
- Run a small ideation or plotting step to confirm model access.

## Container environment

In split mode, ensure required variables are available inside the container.
Most clusters do this by passing the environment through Singularity; if not,
set them explicitly in your job script or container settings.
