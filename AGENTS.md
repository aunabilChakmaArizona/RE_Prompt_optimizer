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
