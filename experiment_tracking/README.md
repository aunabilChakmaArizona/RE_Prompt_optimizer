# Central experiment and research-note index

This is the central index for project-created experiment notes, reports, and
processing documentation. Generated run directories and third-party baseline
files are intentionally not indexed individually.

DeltaAI agents should also read `../DELTAAI_AGENTS.md` for the required reading
order, runtime precautions, and the current Gemma/vLLM incident status.

## Current completion status

- `experiment_status.txt`: one central COMPLETE/INCOMPLETE table for
  OpenBookQA, MATH-500, and HotpotQA, split by model and by all three
  first-stage and six second-stage methods, with pointers to detailed results.
  Update it whenever a run is completed, replaced, or selected for final
  evaluation.

## Dataset preparation

- ANLI prompt optimization uses `../data/processed/anli/validation_promptopt.jsonl`:
  3,000 examples divided into three deterministic 1,000-example folds. Every
  fold mixes R1/R2/R3 and is label-balanced, with the single extra label and
  round example rotated across folds. Preparation and validation are implemented
  in `../codes/prepare_anli.py`; detailed counts are recorded in
  `../data/processed/anli/dataset_info.json`.

## First-stage experiments

- `first_stage/detailed_first_stage_results.txt`: canonical raw/stable scores,
  iteration snapshots, CODE values, directories, and prompt-audit status.
- `first_stage/first_stage_summary_and_prompts.txt`: compact stable-score table
  followed by the complete initial and retained prompts.
- `first_stage/math_qwen_rpo_analysis.txt`: iteration-by-iteration diagnosis of
  the Qwen Math RPO run that retained its initial prompt, plus the implemented
  incorrect-only/token-limit feedback update, completed retry, and verified
  per-model validation sizes.
- `first_stage/math_qwen_rpo_source_adjustments.txt`: selected Qwen Math RPO
  sources, random-selection seed, actual iterations, retry scores, and exact
  prompt paths; original iteration snapshots are preserved.
- `first_stage/math_evoprompt_source_adjustments.txt`: explicit Math EvoPrompt
  experimental source-slot assignments, actual iterations, scores, and prompt
  files for both models; original snapshot files are preserved.
- `first_stage/README.md`: scope and interpretation rules for the canonical
  first-stage reports.
- `qa/first_stage_lambda1_status.txt`: historical OpenBookQA-only tracking.
- `qa/README.md`: interpretation of the historical QA report.

Regenerate the canonical cross-dataset first-stage reports with:

```bash
python -u codes/report_first_stage_results.py
```

## Second-stage experiments

- `second_stage/vllm_objective_scoring.txt`: optional vLLM candidate loss/fluency,
  unchanged objective, beam-score caching, regression checks, and a read-only
  cached-response GPU benchmark; small Qwen measurements are available, while
  Gemma, full-beam timing, and near-tie score reproducibility remain open.
- `second_stage/length_sorted_objective_scoring_benchmark.txt`: exact-token-length
  archived sorting diagnostic (temporary code removed), two-pass HF/vLLM
  timings, score-mapping checks, and numerical variation. Sorting benefited
  only HF in this test; it is not enabled in production scoring.
- `second_stage/baseline_tuning_followups.txt`: TODO for limited LPO/GreaTer
  sensitivity checks and disclosure of unequal OpenBookQA tuning effort.
- `second_stage/math_gradpo_gen_tuning_plan.txt`: completed Qwen Math
  GradPO-Gen S/T/H matrix and two seed checks (C=7, B=5, target limit 4096),
  selected shared setting, flagged malformed backup prompt, cache rules, and
  launch commands. The later eight-run grid remains the script default on GPU 1.
  Commands: `../codes/run_math_gradpo_tuning_qwen.sh`.
- `second_stage/math_gradpo_gemma_slurm_guidelines.txt`: approved Gemma Math
  eight-attempt S/T/H matrix (pool 600, G 200, C 7, B 5, target limit 4096),
  H100/H200 deployment, DeltaAI ARM/GH200 caveat, GitHub transfer, Slurm
  submission/resume, and result paths. Portable, verbatim RPO5/RPO10 sources
  are in `second_stage/math_gemma_sources/` (original snapshots unchanged).
  Commands: `../codes/run_math_gradpo_tuning_gemma.sh`;
  Slurm: `../codes/run_math_gradpo_tuning_gemma.slurm`.
- `second_stage/deltaai_gemma_vllm_failure_handoff.txt`: audit of the completed
  but invalid DeltaAI Gemma MATH-500 eight-run matrix, including the corrupted
  RPO5 candidate generation, collapsed/cached RPO10 source evaluation, exact
  runtime configuration, cache risks, and the ordered A/B diagnostics required
  before rerunning the matrix.
- `second_stage/shared_training_pool_cache.txt`: shared LPO/GreaTer/GradPO raw
  training-response cache, compatible larger-pool reuse, matching rules, flags,
  and the LPO greedy training-feedback decoding change.
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
  analysis for GradPO using the consolidated four-category taxonomy.
- `../gradpo_selected_spans_four_categories.txt`: complete 75-occurrence
  selected-span list grouped into the four paper categories, with duplicates,
  source runs, selected-region positions, and per-category frequency summaries.
- `../gradpo_selected_spans_four_categories_simple.txt`: minimal four-category
  list containing only group headings, counts, and one selected span per line.
- `../temp_category.tex`: compact four-category LaTeX table for the paper.
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

## Final-test experiments

- `final_test/openbookqa_five_seed_test_protocol.txt`: five separate batched
  test evaluations per model using seeds 42, 1, 100, 1000, 10000; one model load,
  16 output tokens and memory ratio 0.8. The latest-RPO Qwen and Gemma
  matrices each have 15 rows. Scripts:
  `../codes/run_openbookqa_final_test_qwen_5runs.sh` and
  `../codes/run_openbookqa_final_test_gemma_5runs.sh`. Results are saved separately
  under `../outputs/qa_final_test/non_reasoning/selected_prompts/<CODE>/seed_<SEED>/`.
  The parent summary tracks completion. The completed mean/sample-std accuracy
  and source-relative mean gains are recorded in
  `final_test/openbookqa_latest_rpo_five_seed_results.txt`.
- `final_test/openbookqa_qwen_latest_rpo_final_prompts.tsv`: frozen 15-row
  Qwen manifest containing its latest RPO-5/RPO-10 and six refiners per source.

## Professor-facing result bundles

- `demos/math500_qwen_gradpo/results.txt`: compact tables for all eight
  completed Qwen MATH-500 GradPO-Gen validation runs, per-source best settings,
  and the common setting that improved both RPO sources. The adjacent
  `demos/math500_qwen_gradpo/prompts/` folder contains simply named copies
  of the initial, RPO-5, RPO-10, and two best GradPO prompts. These are tuning
  results, not final MATH-500 test scores; no Gemma second-stage results are
  included because no completed local Gemma summaries currently exist.
- `demos/openbookqa_qwen_5runs/results.txt`: Qwen OpenBookQA final-test table
  across seeds 42, 1, 100, 1000, and 10000. It reports mean accuracy and sample
  standard deviation plus within-seed paired gain mean/std for all 36 manifest
  rows. Seven unchanged second-stage rows are intentionally left blank.
- `final_test/openbookqa_qwen_matched_retest.txt`: fresh Qwen-only solo/batch
  retest protocol, using `../codes/run_openbookqa_qwen_matched_test.sh`, GPU 3,
  ratio 0.8, 16 output tokens, native context 40960 and timestamped CODEs.
  After completion, `../codes/report_openbookqa_test_rerun.py` automatically
  saves all selected old/new scores and per-method averages to
  `final_test/openbookqa_qwen_matched16_comparison_<RUN_TAG>.txt`.
  Historical reports and runs are preserved.
- `final_test/openbookqa_gemma_matched16_test_results.txt`: completed fresh
  Gemma solo and batched tests (tag `20260917T174040Z`), including all selected
  first-/second-stage scores, source-relative test gains, validation/test method
  averages and transfer counts. Both use 16 output tokens and images allowed.
  Initial scores are 64.20% solo and 64.40% batched; the former 0% anomaly
  did not recur. The historical 10-token tables remain separate and unchanged.
- `final_test/openbookqa_test_scores_only.txt`: completed Qwen/Gemma
  OpenBookQA non-reasoning test accuracies for the initial prompt, all five
  first-stage sources, and all 30 second-stage rows per model. Also includes
  exact selected-prompt validation accuracy/stable scores, source-relative
  validation and test gains, validation sizes, and per-method transfer counts.
  Additional comparison tables group all five sources together per second-stage
  method, separately for Qwen and Gemma, without changing any scores.
  Per-method arithmetic-average tables include all five sources (unchanged
  refinements included), using unrounded saved validation/test scores and gains.
  Stable scores apply only to validation, not the single test split.
  These tables retain the historical 10-token batch protocol; the fresh
  matched16 retest is tracked separately. Gemma's recorded initial
  score is anomalous (all 500 outputs missing tags); investigate before
  using initial-to-first-stage gains in the paper.
- `final_test/openbookqa_final_test_protocol.txt`: OpenBookQA non-reasoning
  final-test protocol, one decoding run using the existing
  `../codes/run_qa_final_test_evaluation.py --runs 1`, GPU 3 by default,
  selected-source provenance, and output paths. Run the model scripts
  sequentially on this GPU. Scripts:
  `../codes/run_openbookqa_final_test_qwen.sh` and
  `../codes/run_openbookqa_final_test_gemma.sh`.
- `final_test/openbookqa_selected_prompts.tsv`: frozen 72-row first-/second-stage
  prompt manifest with exact CODEs, paths, parent links, and SHA-256 hashes.
- `final_test/openbookqa_gemma_latest_rpo_final_prompts.tsv`: frozen 15-row
  Gemma manifest containing the initial prompt, latest RPO-5/RPO-10 prompts,
  and all six final second-stage variants for each source.
- `../codes/run_openbookqa_initial_test_gemma.sh`: standalone initial-prompt
  check for Gemma's anomalous 0/500 batched baseline. Uses the existing
  evaluator, one decoding run, GPU 3, memory ratio 0.8, output limit 16 tokens,
  and a separate retry
  CODE. Results:
  `../outputs/qa_final_test/non_reasoning/final_test/openbookqa_non_reasoning_gemma_initial_test_once_retry/`.
  This does not replace the earlier recorded baseline or scores-only table.
- `final_test/openbookqa_gemma_matched_retest.txt`: verified Gemma prompt
  selections and fresh solo-versus-batch diagnostic protocol. Run
  `../codes/run_openbookqa_gemma_matched_test.sh` for a solo initial check followed
  by all 36 selected rows, with 16 output tokens, images allowed, native context,
  GPU 3 and memory ratio 0.8. Timestamped CODEs preserve previous results;
  historical tables are not automatically replaced.
- Current per-prompt final-test scores and predictions are saved in
  `../outputs/qa_final_test/non_reasoning/final_test/<CODE>/`, with 36 unique
  CODEs per model. The earlier batch evaluator/report helper are not invoked;
  the combined helper does not yet read this per-prompt format.
- Alternative batch script: `../codes/run_openbookqa_final_test_batched.sh`
  loads each model once and evaluates all 36 selected rows (not 32), with one
  decoding run on GPU 3. All three final-test scripts default to GPU memory
  utilization 0.8 after the initial 0.9 startup memory check failed.
  By default it runs Qwen then Gemma sequentially;
  pass `qwen` or `gemma` for one model only. The existing two single-prompt
  scripts remain unchanged. Batched outputs use
  `../outputs/qa_final_test/non_reasoning/selected_prompts/<CODE>/`; after both
  models finish, this script automatically writes the compatible combined TXT
  to `final_test/openbookqa_final_test_results.txt`.

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

- `deltaai_setup_step_by_step.txt`: complete beginner-friendly DeltaAI SSH/Duo,
  allocation/storage discovery, one-time Bash startup settings, HOME-based
  Conda and project-based Git/data/results, pinned vLLM 0.11/PyTorch 2.8
  CUDA 12.9 wheel installation,
  compute-node checks, Gemma access, Slurm submission/resume, troubleshooting,
  and a planned Git-friendly TXT results bundle (exporter not yet implemented).
- `vllm_v2_environment_setup.txt`: storage2 conda location configuration,
  vLLM v2 installation commands, verification, and driver caveats; optional
  candidate scoring is implemented, with small Qwen GPU comparisons documented
  in the second-stage scoring notes and broader checks still pending.
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
