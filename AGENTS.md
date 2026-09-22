Notes on this project
* Simple but effective code
* Modularized and readable code
* Share and confirm the plan with me first before implementing
* Ask me if you are confused (don't decide yourself)
* define each "def" methods in simple description in the next line
* When adapting a published baseline, preserve its original prompts and wording as closely as possible; make only necessary task-specific substitutions
* Keep meta-prompt wording and structure consistent across tasks; change only unavoidable task descriptions, labels, and answer formats

Runtime environment
* Use the Conda environment `re_prompt_optimization_vllm_v2` for vLLM inference and the current QA/Math experiment code.
* Activate it with `conda activate re_prompt_optimization_vllm_v2`.
* On NCSA DeltaAI, the site module's `LD_LIBRARY_PATH` can make PyTorch load an incompatible site NCCL. After activating the Conda environment on a compute node or in a Slurm job, run `unset LD_LIBRARY_PATH` before importing PyTorch/vLLM. Keep this job-scoped; do not put it in `~/.bashrc`.
* With Gemma 3 and vLLM 0.11.0, never pass `--vllm-disable-images` or set `limit_mm_per_prompt={"image": 0}`. This configuration produced empty/corrupted generations on DeltaAI. Keep image support enabled even for text-only tasks. The vLLM loader rejects this unsafe combination.

Experiment tracking
* The central COMPLETE/INCOMPLETE status for OpenBookQA, MATH-500, and HotpotQA is tracked in `experiment_tracking/experiment_status.txt`, with one table split by model and by all three first-stage and six second-stage methods, followed by result-file pointers. Update it whenever a method finishes or a selected run changes.
* Canonical first-stage results for OpenBookQA non-reasoning, HotpotQA reasoning, and MATH-500 reasoning are tracked in `experiment_tracking/first_stage/detailed_first_stage_results.txt`.
* The compact stable-score table and complete initial/retained prompts are tracked in `experiment_tracking/first_stage/first_stage_summary_and_prompts.txt`.
* Regenerate both reports after a first-stage run finishes with `python -u codes/report_first_stage_results.py`.
* See `experiment_tracking/first_stage/README.md` for the report scope and interpretation rules.
* The QA/Math second-stage implementation audit, resolved issues, and remaining follow-ups are tracked in `experiment_tracking/second_stage/qa_math_second_stage_code_audit.txt`.
* OpenBookQA GradPO tuning results and selected per-model configurations are tracked in `experiment_tracking/second_stage/openbookqa_gradpo_tuning_results.txt`.
* Before final QA/Math second-stage experiments, revisit task-specific protected words/tokens. Relation extraction protected structural template words and literal labels; QA/Math do not yet have an equivalent policy. See the audit above before deciding or implementing one.
* Second-stage refiners share exact source-prompt validation results through `outputs/shared_source_validation_cache`. Do not substitute historical first-stage aggregate scores unless every validation and decoding setting matches the cache identity.
* Before final comparative claims, run a small LPO/GreaTer sensitivity check; these methods have tunable edit/feedback/proposal/fluency settings, not just batch sizes. GradPO received more extensive OpenBookQA tuning. See `experiment_tracking/second_stage/baseline_tuning_followups.txt`.
* The approved Qwen Math GradPO-Gen eight-attempt fast tuning plan is tracked in `experiment_tracking/second_stage/math_gradpo_gen_tuning_plan.txt`; commands are in `codes/run_math_gradpo_tuning_qwen.sh`. The default runs all eight attempts with C=7, B=5, and target output limit 4096; optional phase overrides remain available.
* LPO and all gradient refiners share greedy training-source responses through `outputs/shared_gradient_cache`, including compatible larger-pool reuse. Matching rules, prefix selection, flags, and the LPO training-decoding change are tracked in `experiment_tracking/second_stage/shared_training_pool_cache.txt`.
* Optional vLLM 0.11.0 forward-only candidate task loss and per-search beam-score caching are tracked in `experiment_tracking/second_stage/vllm_objective_scoring.txt`. HF retains gradients and scores candidate-prompt fluency because Gemma produced a non-finite vLLM fluency logprob; use `--backend dual --objective-scoring-backend vllm`. Run the documented real-model comparison before claiming GPU numerical parity or speed.

Performance TODO
* For the first-stage optimizers (RPO, EvoPrompt, and ETGPO), batch all candidate-prompt/example pairs together when scoring candidates on a training or validation set, instead of evaluating each candidate prompt separately, wherever the scoring method permits it. The current vLLM-backed shared evaluation path already supports this; retain the design in future changes.
