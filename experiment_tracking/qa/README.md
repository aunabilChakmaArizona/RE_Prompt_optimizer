# QA experiment tracking

This directory tracks the OpenBookQA prompt-optimization experiments only. Math experiments must be recorded separately.

## Current report

`first_stage_lambda1_status.txt` records the completed first-stage runs using validation standard-deviation penalty lambda = 1. It includes:

- whether each run completed;
- whether its selected prompt genuinely improved over the initial prompt;
- raw and stable validation scores;
- repository-relative run and prompt paths;
- the 20 intended prompt slots for second-stage optimization; and
- compact iteration histories for RPO and EvoPrompt-DE.

## Status meanings

- `READY`: the run is valid, the saved prompt differs from the initial prompt, and its lambda-1 stable validation score is higher.
- `NOT READY`: the run completed but did not produce a genuinely improved prompt. It should be retuned before it is presented as a successful first-stage optimization or used as an improved first-stage source.
- `PENDING RERUN`: an older artifact exists, but the implementation has changed and the run must be repeated before use.

Run completion and optimization success are deliberately tracked separately. A completed run that retains the initial prompt is not counted as a first-stage improvement.

## Scoring and reporting rules

- Raw score is accuracy over the 900-example validation set.
- Stable score is mean accuracy over three fixed 300-example validation folds minus one population standard deviation.
- Improvement is determined by the stable score, but the prompt text must also differ from the initial prompt.
- Scores in this folder are validation scores, not test scores.
- Test evaluation remains disabled during prompt selection. Paper test results must come from the separate five-run final-test evaluation.
- If iteration 5 and iteration 10 contain identical prompt text, they remain separate experimental slots but are marked as duplicates.

## Required refresh

The two non-reasoning ETGPO runs currently require rerunning. They were produced before non-reasoning ETGPO was changed to generate one post-hoc error explanation per wrong answer before taxonomy construction. A refreshed run should contain:

- `failure_feedbacks.json`;
- `taxonomy.json` with `analysis_source` equal to `posthoc_target_model_feedback`;
- `final_prompt.txt`; and
- `summary.json` with `taxonomy_analysis_source` equal to `posthoc_target_model_feedback`.

After either run is repeated, update `first_stage_lambda1_status.txt` from the new `summary.json`, `taxonomy.json`, and saved prompt files before launching its second-stage jobs.

The four current ETGPO rerun commands are the only active commands in:

- `codes/run_qa_first_stage_qwen.sh`; and
- `codes/run_qa_first_stage_gemma.sh`.

They use new run identities ending in `_lambda1_fixed`, so the historical ETGPO outputs remain available for comparison.
