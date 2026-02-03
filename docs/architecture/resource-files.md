# Resource Files

You can supply a JSON/YAML resource file with `--resources` to describe
additional data, templates, or code to inject into the LLM context and/or mount
into the container.

See `data_resources.json` for a concrete example used in this fork.

## When to use resources

- Mount datasets from shared storage into the container.
- Stage template code or baseline implementations for reference.
- Provide documentation or setup notes that should be injected into prompts.

## File structure

A resource file can include four top-level lists: `local`, `github`,
`huggingface`, and `items`.

```json
{
  "local": [
    {
      "name": "input_data",
      "host_path": "/shared/datasets/my_data",
      "mount_path": "/workspace/input/data",
      "read_only": true
    }
  ],
  "github": [
    {
      "name": "cnpy",
      "repo": "https://github.com/rogersce/cnpy.git",
      "ref": "v1.0.0",
      "dest": "/workspace/third_party/cnpy",
      "as": "library"
    }
  ],
  "huggingface": [
    {
      "name": "my_model",
      "type": "model",
      "repo_id": "org/model-name",
      "revision": "abc123def456...",
      "dest": "/workspace/input/my_model"
    }
  ],
  "items": [
    {
      "name": "baseline_template",
      "class": "template",
      "source": "local",
      "resource": "input_data",
      "path": "baseline",
      "include_tree": true,
      "include_files": ["main.c", "Makefile"],
      "notes": "Use as a reference implementation."
    },
    {
      "name": "readme_doc",
      "class": "document",
      "source": "local",
      "resource": "input_data",
      "path": "README.md",
      "include_content": true
    }
  ]
}
```

## Concrete example from this repo

`data_resources.json` mounts local data and stages templates for the Himeno
benchmark. The template items include specific files only (no full tree dump),
which keeps prompt size controlled.

```json
{
  "local": [
    {
      "name": "ai_scientist_data",
      "host_path": "/home/users/.../AI-Scientist-v2-HPC/data",
      "mount_path": "/workspace/input/data",
      "read_only": true
    }
  ],
  "items": [
    {
      "name": "himenobmt_baseline",
      "class": "template",
      "source": "local",
      "resource": "ai_scientist_data",
      "path": "himenobmt",
      "include_tree": true,
      "max_chars": 0,
      "max_total_chars": 0,
      "include_files": ["himenoBMT.c", "Makefile"],
      "notes": "Baseline Himeno BMT C implementation."
    },
    {
      "name": "himenobmt_pthreads",
      "class": "template",
      "source": "local",
      "resource": "ai_scientist_data",
      "path": "himenobmt_c_thr",
      "include_tree": true,
      "max_chars": 0,
      "max_total_chars": 0,
      "include_files": ["himenoBMT_t.c", "barrier.h", "barrier.c", "Makefile"],
      "notes": "Pthreads version with a custom barrier."
    },
    {
      "name": "himenobmt_mpi",
      "class": "template",
      "source": "local",
      "resource": "ai_scientist_data",
      "path": "himenobmt_mpi",
      "include_tree": true,
      "max_chars": 0,
      "max_total_chars": 0,
      "include_files": ["himenoBMT_m.c", "Makefile"],
      "notes": "MPI version for distributed-memory runs."
    },
    {
      "name": "data_manifest",
      "class": "document",
      "source": "local",
      "resource": "ai_scientist_data",
      "path": "manifest.json",
      "include_content": true,
      "notes": "Manifest describing available local resources."
    }
  ]
}
```

## Minimal example (dataset mount only)

```json
{
  "local": [
    {
      "name": "dataset",
      "host_path": "/shared/datasets/my_dataset",
      "mount_path": "/workspace/input/dataset",
      "read_only": true
    }
  ]
}
```

## Path rules

- `mount_path`/`dest` must be under `/workspace`.
- `host_path` must exist for local resources (relative paths are resolved
  relative to the resource file).
- Resource file paths are resolved relative to `AI_SCIENTIST_ROOT` when set by
  the launcher.
- `items.path` is relative to the resource root and must not contain `..`.

## Mounts vs. items

- `local` entries bind-mount host directories into the container (split mode).
- `github` and `huggingface` entries describe content to fetch during Phase 1.
  They are not auto-fetched by the host.
- `items` classify files/dirs and control what gets injected into prompts.
  The `class` must be one of `template`, `library`, `dataset`, `model`, `setup`,
  or `document`.

## Staging and prompt injection

- Local items for `template`, `setup`, and `document` are staged into
  `resources/<class>/<name>` before execution.
- By default, templates include summarized file content and are injected into
  Phase 0/1/2 prompts; setup content is injected in Phase 0/1; document content
  is injected in Phase 0/1/2/3/4.
- Libraries/datasets/models include metadata only unless overridden via
  `include_*` or `max_*` fields.

## Debugging resources

- Confirm the launcher is run from the repo root so `AI_SCIENTIST_ROOT` is set.
- Inspect `experiments/<run>/memory/resource_snapshot.json` only if a snapshot
  was explicitly created (it is not auto-generated).
- Check for staged content under `experiments/<run>/resources/` in split mode.

## Fetch behavior

- GitHub/HF resources are fetched inside the container during Phase 1 via
  `git clone` or `huggingface_hub` calls.
- The `dest` path should match where the code expects to find the resource.

## YAML support

YAML resource files are supported but require `pyyaml` on the host.
