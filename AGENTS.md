Notes on this project
* Simple but effective code
* Modularized and readable code
* Share and confirm the plan with me first before implementing
* Ask me if you are confused (don't decide yourself)
* define each "def" methods in simple description in the next line
* When adapting a published baseline, preserve its original prompts and wording as closely as possible; make only necessary task-specific substitutions
* Keep meta-prompt wording and structure consistent across tasks; change only unavoidable task descriptions, labels, and answer formats

Experiment tracking
* Canonical first-stage results for OpenBookQA non-reasoning, HotpotQA reasoning, and MATH-500 reasoning are tracked in `experiment_tracking/first_stage/detailed_first_stage_results.txt`.
* The compact stable-score table and complete initial/retained prompts are tracked in `experiment_tracking/first_stage/first_stage_summary_and_prompts.txt`.
* Regenerate both reports after a first-stage run finishes with `python -u codes/report_first_stage_results.py`.
* See `experiment_tracking/first_stage/README.md` for the report scope and interpretation rules.
* The QA/Math second-stage implementation audit, resolved issues, and remaining follow-ups are tracked in `experiment_tracking/second_stage/qa_math_second_stage_code_audit.txt`.
* OpenBookQA GradPO tuning results and selected per-model configurations are tracked in `experiment_tracking/second_stage/openbookqa_gradpo_tuning_results.txt`.
* Before final QA/Math second-stage experiments, revisit task-specific protected words/tokens. Relation extraction protected structural template words and literal labels; QA/Math do not yet have an equivalent policy. See the audit above before deciding or implementing one.
* Second-stage refiners share exact source-prompt validation results through `outputs/shared_source_validation_cache`. Do not substitute historical first-stage aggregate scores unless every validation and decoding setting matches the cache identity.

Performance TODO
* For the first-stage optimizers (RPO, EvoPrompt, and ETGPO), batch all candidate-prompt/example pairs together when scoring candidates on a training or validation set, instead of evaluating each candidate prompt separately, wherever the scoring method permits it. The current vLLM-backed shared evaluation path already supports this; retain the design in future changes.
