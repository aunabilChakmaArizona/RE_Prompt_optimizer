# Central experiment and research-note index

This is the central index for project-created experiment notes, reports, and
processing documentation. Generated run directories and third-party baseline
files are intentionally not indexed individually.

## First-stage experiments

- `first_stage/detailed_first_stage_results.txt`: canonical raw/stable scores,
  iteration snapshots, CODE values, directories, and prompt-audit status.
- `first_stage/first_stage_summary_and_prompts.txt`: compact stable-score table
  followed by the complete initial and retained prompts.
- `first_stage/math_qwen_rpo_analysis.txt`: iteration-by-iteration diagnosis of
  the Qwen Math RPO run that retained its initial prompt.
- `first_stage/README.md`: scope and interpretation rules for the canonical
  first-stage reports.
- `qa/first_stage_lambda1_status.txt`: historical OpenBookQA-only tracking.
- `qa/README.md`: interpretation of the historical QA report.

Regenerate the canonical cross-dataset first-stage reports with:

```bash
python -u codes/report_first_stage_results.py
```

## Second-stage experiments

- `second_stage/qa_math_second_stage_code_audit.txt`: QA/Math LPO, GreaTer, and
  GradPO flow review; resolved implementation issues; intentional differences
  from relation extraction; and the pending protected-word decision.
- `second_stage/gradient_sampling_configuration.txt`: task/model-specific
  random-pool sizes, the fixed 100-correct plus 100-incorrect gradient subset,
  and controlled-comparison requirements.
- `second_stage/openbookqa_gradpo_tuning_results.txt`: Qwen GradPO tuning table,
  selected OpenBookQA configuration, run path, and staged Gemma tuning status.
- `../gradpo_gen_vs_random_all_20_results.txt`: complete GradPO-Gen versus
  random-span result table used for the rebuttal.
- `../random_mode_results.txt`: raw random-span experiment results.
- `../changed_prompt_stats.txt`: edit-distance and changed-prompt statistics.
- `../gradpo_span_selection_content_stats.txt`: selected-span counts and content
  analysis for GradPO.
- `../gradpo_changed_span_audit.csv`: detailed GradPO selected-span audit.
- `../lpo_greater_changed_source_audit.csv`: source/edit audit for LPO and
  GreaTer.
- `../prompt_edit_distance_details.csv`: prompt-pair edit details underlying the
  aggregate statistics.

Associated report scripts:

- `../write_changed_prompt_stats.py`
- `../write_gradpo_random_paper_table.py`
- `../write_gradpo_span_selection_content_stats.py`
- `../gradpo_span_audit.py`
- `../lpo_greater_source_audit.py`
- `../prompt_edit_distance_stats.py`
- `../prompt_pair_token_stats.py`

## Prompt and paper working files

- `../all_the_prompts.txt`: collected optimized-prompt source used by the prompt
  comparison analyses.
- `../all_the_prompts_tacred_tcolorboxes.txt`: TACRED prompt material formatted
  for paper boxes.
- `../meta_prompts_tcolorboxes.txt`: meta-prompts formatted for the paper.
- `../method_section.txt`: working method-section text.
- `../review.txt`: reviewer comments and rebuttal working notes.
- `../15315_Two_Stage_Prompt_Optimiz.pdf`: current paper PDF used as the primary
  experimental reference.

## Hyperparameter and diagnostic reports

- `../lambda_tuning.txt`: stable-score/perplexity-weight tuning notes.
- `../codes/lambda_beam_score_report.txt`
- `../codes/er_beam_score_report.txt`
- `../codes/beam3_first_iteration_best_prompt_report.txt`
- `../codes/q5_first_iteration_best_prompt_report.txt`
- `../codes/region_tune_best_prompt_report.txt`
- `../codes/textual_vs_gradient_tradeoffs.txt`
- `../codes/population_validation_scores_report.txt`
- `../codes/population_validation_scores_score_based_report.txt`
- `../codes/population_validation_scores_score_based_report_80.txt`
- `../codes/population_validation_scores_score_based_lineage_report_80.txt`
- `../codes/population_cumulative_f1_score_based_report.txt` and its window-5
  comparison/bootstrap variants.

## Implementation and dataset processing documentation

- `../codes/prompt_optimization/README.md`: QA/Math optimization design,
  hyperparameters, runners, commands, and saved artifacts.
- `../codes/math_grading/README.md`: Math grading implementation.
- Dataset provenance and processing decisions are recorded in each dataset's
  `SOURCE.md` under `data/`, including OpenBookQA, HotpotQA, Math-500, FOLIO,
  WebQuestions, CommonsenseQA, and the AIME collections.

## Maintenance rule

Whenever a new persistent note or aggregate report is created, add it to this
index. Do not add individual logs, generated prompts inside run directories, or
third-party repository data files; those are discoverable through their parent
run or upstream README.
