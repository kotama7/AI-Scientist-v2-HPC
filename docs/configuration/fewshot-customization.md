# Few-shot Example Customization

This document explains how to customize few-shot examples for paper reviews.

## Overview

Few-shot examples show the LLM the expected review quality and style. The default examples target AI/ML papers, but you can replace them with HPC, physics, biology, or other domain-specific examples.

## Few-shot Example Structure

Each example consists of three files:

```
ai_scientist/fewshot_examples/
├── example_name.pdf      # Paper PDF
├── example_name.txt      # Text extracted from the PDF (optional)
└── example_name.json     # Review output
```

### JSON Format

```json
{
    "review": "{\"Summary\": \"...\", \"Strengths\": [...], \"Weaknesses\": [...], \"Originality\": 4, ...}"
}
```

## Method 1: Use the Auto-generation Script (Recommended)

### Basic Usage

```bash
# Generate from a single PDF
python generate_fewshot_examples.py \
    --pdf papers/your_hpc_paper.pdf \
    --output-dir ai_scientist/fewshot_examples \
    --model gpt-4o-2024-11-20
```

### Customize for HPC Papers

```bash
python generate_fewshot_examples.py \
    --pdf papers/himeno_benchmark.pdf \
    --output-dir ai_scientist/fewshot_examples \
    --model gpt-4o-2024-11-20 \
    --custom-prompt "Review this HPC paper focusing on: 1) Parallel scalability (strong/weak scaling), 2) Performance analysis and bottlenecks, 3) Reproducibility of benchmark conditions, 4) Comparison with baseline implementations." \
    --num-reflections 5
```

### Batch Generation from Multiple PDFs

```bash
# Generate from a directory of HPC papers
for pdf in papers/hpc/*.pdf; do
    python generate_fewshot_examples.py \
        --pdf "$pdf" \
        --output-dir ai_scientist/fewshot_examples_hpc \
        --custom-prompt "Focus on HPC scalability, performance, and reproducibility"
done
```

### Text Extraction Only (No Review)

```bash
# Skip review generation and only extract text
python generate_fewshot_examples.py \
    --pdf papers/example.pdf \
    --output-dir ai_scientist/fewshot_examples \
    --dry-run
```

## Method 2: Create Manually

### Step 1: Extract PDF Text

```python
from ai_scientist.review import load_paper

pdf_path = "papers/your_paper.pdf"
text = load_paper(pdf_path)

with open("ai_scientist/fewshot_examples/your_paper.txt", "w") as f:
    f.write(text)
```

### Step 2: Generate a Review

```python
from ai_scientist.llm import create_client
from ai_scientist.review import perform_review
import json

client, model = create_client("gpt-4o-2024-11-20")
review = perform_review(
    text=text,
    model=model,
    client=client,
    num_reflections=3,
    num_fs_examples=0  # Do not use existing examples
)

# Save as JSON
review_wrapper = {"review": json.dumps(review, indent=4)}
with open("ai_scientist/fewshot_examples/your_paper.json", "w") as f:
    json.dump(review_wrapper, f, indent=4)
```

### Step 3: Copy the PDF

```bash
cp papers/your_paper.pdf ai_scientist/fewshot_examples/
```

## Method 3: Update the Few-shot Example List

To use your generated few-shot examples, update the list in code.

### Check the Existing List

[ai_scientist/review/llm_review.py:40-50](ai_scientist/review/llm_review.py#L40-L50)

```python
fewshot_papers = [
    os.path.join(parent_dir, "fewshot_examples/132_automated_relational.pdf"),
    os.path.join(parent_dir, "fewshot_examples/attention.pdf"),
    os.path.join(parent_dir, "fewshot_examples/2_carpe_diem.pdf"),
]

fewshot_reviews = [
    os.path.join(parent_dir, "fewshot_examples/132_automated_relational.json"),
    os.path.join(parent_dir, "fewshot_examples/attention.json"),
    os.path.join(parent_dir, "fewshot_examples/2_carpe_diem.json"),
]
```

### Replace with HPC Examples

```python
# Switch to HPC paper examples
fewshot_papers = [
    os.path.join(parent_dir, "fewshot_examples/himeno_benchmark.pdf"),
    os.path.join(parent_dir, "fewshot_examples/mpi_performance.pdf"),
    os.path.join(parent_dir, "fewshot_examples/gpu_acceleration.pdf"),
]

fewshot_reviews = [
    os.path.join(parent_dir, "fewshot_examples/himeno_benchmark.json"),
    os.path.join(parent_dir, "fewshot_examples/mpi_performance.json"),
    os.path.join(parent_dir, "fewshot_examples/gpu_acceleration.json"),
]
```

### Specify Dynamically at Runtime (Extended)

A more flexible approach:

```python
# Edit ai_scientist/review/llm_review.py
import os

def get_fewshot_paths(domain="ai"):
    """Get few-shot example paths by domain.

    Args:
        domain: "ai", "hpc", "physics", "biology", etc.

    Returns:
        Tuple of (fewshot_papers, fewshot_reviews)
    """
    parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    domain_examples = {
        "ai": [
            "132_automated_relational",
            "attention",
            "2_carpe_diem",
        ],
        "hpc": [
            "himeno_benchmark",
            "mpi_scalability",
            "gpu_acceleration",
        ],
    }

    examples = domain_examples.get(domain, domain_examples["ai"])

    fewshot_papers = [
        os.path.join(parent_dir, f"fewshot_examples/{ex}.pdf")
        for ex in examples
    ]
    fewshot_reviews = [
        os.path.join(parent_dir, f"fewshot_examples/{ex}.json")
        for ex in examples
    ]

    return fewshot_papers, fewshot_reviews

# Default is AI
fewshot_papers, fewshot_reviews = get_fewshot_paths(
    domain=os.environ.get("REVIEW_DOMAIN", "ai")
)
```

Usage:

```bash
# Review with the HPC domain
REVIEW_DOMAIN=hpc python generate_paper.py --experiment-dir experiments/xxx
```

## Best Practices for HPC Few-shot Examples

### 1. Choose Appropriate Papers

Good examples:
- Reviewable conference/journal papers (SC, ISC, TPDS, etc.)
- Clear performance evaluation and scalability analysis
- Reproducible benchmark conditions
- Diverse topics (MPI, GPU, hybrid parallelism, etc.)

Avoid:
- Preprints or non-peer-reviewed papers (review quality is uncertain)
- Papers tied to extremely specific environments or hardware
- Theory-only papers without experiments

### 2. Check Review Quality

After auto-generation, verify:
- **Strengths/Weaknesses**: includes HPC-specific aspects (scalability, performance analysis)
- **Significance**: evaluates the validity of benchmark results
- **Soundness**: considers reproducibility of experimental conditions

If anything is missing, manually edit the JSON to improve quality.

### 3. Ensure Diversity

Cover different angles with three examples:
1. **Parallel algorithms**: MPI, OpenMP, hybrid
2. **Accelerators**: GPU, FPGA optimization
3. **Benchmarking and performance analysis**: real-world applications
