# Unified first-stage experiment tracking

This directory tracks the canonical first-stage results used for the extended
experiments:

- OpenBookQA in non-reasoning mode
- HotpotQA in reasoning mode
- MATH-500 in reasoning mode
- Qwen3-4B and Gemma3-4B target models
- RPO, EvoPrompt-DE, and ETGPO first-stage optimizers

`detailed_first_stage_results.txt` contains raw and stable initial, snapshot,
and final scores, together with CODE values, run directories, prompt paths, and
prompt-audit status.

`first_stage_summary_and_prompts.txt` contains the compact initial/final stable
score table followed by the complete initial and retained prompt texts.

Regenerate both reports after new first-stage runs finish:

```bash
python -u codes/report_first_stage_results.py
```

The canonical run mapping is intentionally explicit in the reporting script so
that older, aborted, or superseded directories are not selected accidentally.
Validation sizes may differ across experiments. Stable-score gains must only be
interpreted relative to the initial score from the same run.
