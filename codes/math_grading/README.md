# Math answer grading

The math test runner reports three independent accuracy values:

1. Dataset-specific simple matching: numeric normalization for AIME and
   conservative symbolic-string normalization for MATH-500.
2. The OpenAI PRM800K `grade_answer` implementation vendored under
   `openai_prm800k/`.
3. Hugging Face Math-Verify, pinned in `requirements.txt`.

Install the external dependencies in the same environment used for inference:

```bash
python -m pip install -r codes/math_grading/requirements.txt
```

Run MATH-500 inference:

```bash
python -u codes/run_math_test_inference.py \
  --code math500_qwen_base \
  --dataset data/processed/math500/test.jsonl \
  --model Qwen/Qwen3-4B \
  --device cuda:0
```

Each prediction stores `simple_correct`, `openai_correct`, and
`math_verify_correct`. The summary reports each accuracy separately; the
grader results are never combined with an OR rule.
