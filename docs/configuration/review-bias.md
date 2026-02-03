# Review Bias Configuration

Updated: 2026-02-02

## Overview

AI Scientist supports configurable review bias modes to control how the automated paper review system evaluates generated papers. The review bias determines how the reviewer LLM handles uncertainty when assessing paper quality.

## Available Bias Modes

### Neutral (Default)

**Flag**: `--review-bias neutral`

**Behavior**: Balanced review with no explicit bias on uncertainty.

**System Prompt**: Base reviewer prompt without additional bias instructions.

**Use Case**: Standard fair review for production use.

### Negative (Strict)

**Flag**: `--review-bias neg`

**Behavior**: Strict evaluation that rejects papers when uncertain.

**System Prompt**: Base prompt + "If a paper is bad or you are unsure, give it bad scores and reject it."

**Use Case**:
- Pre-submission quality gate
- Conservative filtering for high-stakes venues
- Identifying potential weaknesses before submission

### Positive (Lenient)

**Flag**: `--review-bias pos`

**Behavior**: Lenient evaluation that accepts papers when uncertain.

**System Prompt**: Base prompt + "If a paper is good or you are unsure, give it good scores and accept it."

**Use Case**:
- Early-stage idea exploration
- Encouraging experimental research directions
- Identifying promising work that needs refinement

## Implementation

### In generate_paper.py

```bash
python generate_paper.py \
    --experiment-dir experiments/2026-01-30_16-22-06_my_experiment \
    --review-bias neutral
```

### In launch_scientist_bfts.py

```bash
python launch_scientist_bfts.py \
    --load_ideas ideas/my_ideas.json \
    --idea_idx 0 \
    --review_bias neutral
```

## Technical Details

### System Prompt Components

1. **Base Prompt** (`reviewer_system_prompt_base`):
   ```
   You are an {persona} who is reviewing a paper that was submitted to a
   prestigious research venue. Be critical and cautious in your decision.
   ```

2. **Negative Bias Suffix** (`reviewer_system_prompt_neg`):
   ```
   If a paper is bad or you are unsure, give it bad scores and reject it.
   ```

3. **Positive Bias Suffix** (`reviewer_system_prompt_pos`):
   ```
   If a paper is good or you are unsure, give it good scores and accept it.
   ```

### Code Mapping

The bias mode is mapped to system prompts in both scripts:

```python
if args.review_bias == "neg":
    reviewer_prompt = reviewer_system_prompt_neg
    print("Using negative bias (strict) review mode.")
elif args.review_bias == "pos":
    reviewer_prompt = reviewer_system_prompt_pos
    print("Using positive bias (lenient) review mode.")
else:  # neutral
    reviewer_prompt = reviewer_system_prompt_base
    print("Using neutral (balanced) review mode.")

review_text = perform_review(
    paper_content,
    client_model,
    client,
    reviewer_system_prompt=reviewer_prompt,
)
```

## Historical Context

**Prior Behavior (Before 2026-02-02)**:
- Default was `reviewer_system_prompt_neg` (strict mode)
- No command-line option to change bias
- Results in systematically lower scores due to "reject if unsure" instruction

**Current Behavior**:
- Default is `neutral` mode (balanced)
- Three modes available via `--review-bias` flag
- Explicit logging of which mode is active

## Comparison Example

For the same paper, different bias modes produce different evaluations:

| Metric | NEG (Strict) | Neutral | POS (Lenient) |
|--------|--------------|---------|---------------|
| Overall Score | 4/10 | 6/10 | 7/10 |
| Decision | Reject | Borderline Accept | Accept |
| Soundness | 2/4 | 3/4 | 3/4 |
| Significance | 2/4 | 2/4 | 3/4 |

**Note**: These are representative examples. Actual scores depend on paper quality and reviewer model.

## Best Practices

### When to Use Each Mode

1. **Neutral (Default)**:
   - Standard workflow
   - Fair comparison across experiments
   - Production paper generation

2. **Negative/Strict**:
   - Final quality check before submission
   - When false positives are more costly than false negatives
   - Validating controversial claims

3. **Positive/Lenient**:
   - Exploring novel/risky ideas
   - When false negatives are more costly than false positives
   - Encouraging creativity in early research stages

### Interpreting Results

- **With Negative Bias**: Low scores indicate genuine weaknesses that need addressing
- **With Neutral Bias**: Scores represent balanced assessment
- **With Positive Bias**: High scores indicate promising directions despite uncertainties

### Multi-Mode Evaluation

For important papers, consider running multiple reviews with different biases:

```bash
# Generate three reviews with different biases
for bias in neg neutral pos; do
    python generate_paper.py \
        --experiment-dir experiments/my_experiment \
        --review-bias $bias \
        --skip-plot --skip-writeup \
        --model-review gpt-4o-2024-11-20

    mv experiments/my_experiment/review_text.txt \
       experiments/my_experiment/review_${bias}.txt
done
```

Then compare the consensus across all three modes.

## Related Configuration

- Review model: `--model-review` (default: `gpt-4o-2024-11-20`)
- Skip review: `--skip-review` flag
- Review output: `review_text.txt` (text review) and `review_img_cap_ref.json` (figure review)

## See Also

- [CLI Entry Points](cli-entry-points.md) - Full command-line reference
- [Few-Shot Customization](fewshot-customization.md) - Customizing review examples
- [Verification Report](../development/verification-report.md) - Implementation consistency
