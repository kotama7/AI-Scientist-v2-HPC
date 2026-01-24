# Requirements

This fork targets HPC environments with a split-phase execution path that runs
inside Singularity. Requirements are split between the host (control plane) and
the worker container image.

## Host vs container matrix

| Capability | Host (control plane) | Container (worker) | Notes |
| --- | --- | --- | --- |
| Python 3.10+ | Required | Required | Host for launcher; container for Phase 1-4 |
| Torch | Required | Optional | Host uses torch for GPU detection |
| Singularity | Required (split mode) | N/A | Only needed to launch containers |
| CUDA runtime | Optional | Optional | Required only for GPU workloads |
| Build tools | Optional | Required | Needed for Phase 1/compile in split mode |
| LaTeX toolchain | Optional | Optional | Required only when writeups enabled |
| `pdftotext` | Optional | Optional | Required only when review enabled |

## Host requirements

- Linux host (the launcher uses `psutil` for cleanup).
- Python 3.10+ for the control plane (3.11 is used in examples).
- Singularity CLI available as `singularity` (Apptainer must be aliased or
  symlinked).
- Torch installed on the host (imported by the launcher for GPU detection).
- GPU + CUDA recommended (default config maps workers to GPU IDs).
- LaTeX toolchain for writeups: `pdflatex`, `bibtex`, `chktex`.
- `pdftotext` (from poppler) for PDF checks and reviews.

## When you can skip items

- If you skip writeups (`--skip_writeup`), you can omit the LaTeX toolchain.
- If you skip PDF review (`--skip_review`), you can omit `pdftotext`.
- If you only run single mode, Singularity is not required.

## Optional host requirements

- `pyyaml` if you want to use YAML resource files.

## Python packages (host)

The host requirements live in `requirements.txt`. Major groups include:

- LLM APIs: `openai`, `anthropic`, `backoff`
- Data + ML: `numpy`, `transformers`, `datasets`, `tiktoken`
- Visualization + PDF handling: `matplotlib`, `pypdf`, `pymupdf4llm`
- Utilities: `rich`, `tqdm`, `coolname`, `omegaconf`, `jsonschema`

## Container image requirements (split mode)

The base SIF image referenced by `exec.singularity_image` should include:

- Python 3.10+
- CUDA toolkit (if you plan to use GPUs)
- Build tools (`gcc`, `make`, `cmake`)
- Git
- Any extra libraries you want available during Phase 1 installs
- `huggingface_hub` if you plan to fetch Hugging Face resources in Phase 1

## Why Torch is required on the host

The launcher imports `torch` before starting containers to detect GPU
availability and map workers to GPU IDs. Even in split mode, this detection
happens on the host.

## HPC notes

- If your cluster uses environment modules, load them before running the
  launcher (CUDA, Singularity, LaTeX, and so on).
- Make sure the `singularity` command resolves even if you are using Apptainer.
- For restricted networks, pre-build images and cache dependencies inside the
  base SIF.
