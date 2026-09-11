# QA and math prompt optimization

This package implements the QA and MATH expansion of the two-stage prompt-optimization experiments. It does not modify the existing relation-extraction runners.

## Experiment matrix

For each QA mode (`reasoning`, `non_reasoning`) and target model (Qwen3-4B, Gemma3-4B), stage one produces five starting prompts:

- RPO iteration 5 and iteration 10
- EvoPrompt-DE iteration 5 and iteration 10
- ETGPO iteration 1

Each starting prompt is refined by six stage-two methods:

- LPO
- GreaTer
- GreaTer-TG
- GradPO-Gen
- GradPO-Prob
- GradPO-Gen-Random

This gives 12 first-stage jobs, 20 first-stage prompt snapshots, and 120 second-stage attempts.

## Fixed QA protocol

Only the leading instruction is editable. The answer-format instruction, question, and choices are always static. OpenBookQA's `fact` field is not shown to the target model or optimizer model.

Reasoning mode enables model thinking and defaults to 4,096 generated tokens. Non-reasoning mode disables thinking and defaults to 10 generated tokens. Both modes use the fixed Qwen3 or Gemma3 sampling settings already used by the project.

## Fixed MATH protocol

MATH uses the same first-stage implementations through `--qa-task math500` and supports reasoning mode only. The editable instruction is followed by the same fixed answer instruction used by `run_math_test_inference.py`; the optimizer never edits or sees that answer-format instruction. Target generations default to 8,192 tokens.

Answers are extracted from the last `<answer>...</answer>` block, with the last balanced `\boxed{...}` expression as fallback. Prompt selection uses the vendored OpenAI PRM800K grader as its `correct` signal. Simple normalization and `math-verify==0.9.0` are run and saved as diagnostics. The 1,500-example validation set has three fixed folds of 500, and prompt selection uses the same mean-minus-standard-deviation stable score with lambda 1. Passing `--validation-fold-size 300` deterministically keeps the first 300 records from each existing fold (900 total) without reshuffling or creating another split. Test evaluation remains disabled during optimization.

Long MATH reasoning traces shown to RPO or ETGPO are compacted to at most 2,000 optimizer-model tokens by retaining their beginning and end. This keeps optimizer meta-prompts within the model context while preserving the setup and final derivation.

## Inference backends

The shared runners default to `--backend transformers`. Use `--backend vllm --gpu-memory-utilization 0.90` to run generated responses with vLLM continuous batching. The vLLM path uses the same chat templates, thinking flags, decoding parameters, response-list interface, and token-usage format as the Transformers path. `--target-batch-size` and `--optimizer-batch-size` remain accepted for command compatibility, but vLLM schedules all submitted prompts dynamically.

RPO, EvoPrompt-DE, ETGPO, and LPO support vLLM. GreaTer, GreaTer-TG, and all GradPO variants must use Transformers because they require backward gradients and direct model logits. The command generator automatically uses Transformers for these gradient-based methods even when `--backend vllm` is requested for the experiment matrix.

Install the separate pinned environment from `requirements_vllm.txt`. Offline vLLM uses one visible GPU per process, so the target and optimizer device arguments must match in a vLLM run.

Validation uses the fixed 1,500-example split as three folds of 500 examples. Prompt selection uses `mean fold accuracy - lambda * population standard deviation`, with `--validation-std-penalty 1.0` by default. Setting lambda to zero gives ordinary accuracy because the three folds have equal size. A second-stage candidate is retained only if its stable validation score strictly exceeds its first-stage source prompt; otherwise, the first-stage prompt is retained. Raw accuracy, each fold accuracy, the fold mean and standard deviation, and the stable score are all saved.

Optimizer-model generations default to `--optimizer-max-new-tokens 10000`, matching the relation-extraction experiment setting.

Generated evaluations are reseeded deterministically by mode, split, and record subset, so every candidate sees the same sampling stream and repeated prompt evaluations are reproducible.

Console logs show one representative prompt/output for each important model-call phase. They also report evaluation sizes, iteration or beam-region progress, current and best regular/stable scores, phase time, and total elapsed run time. Complete prompts, outputs, predictions, and traces remain available in the saved JSON and JSONL artifacts.

EvoPrompt-DE starts from the source instruction plus four fixed provisional seeds in `qa_evoprompt_seeds.py`; replace the clearly labeled placeholders with the final curated seeds before the full experiment. Its original DE meta-prompt and sampled generation are unchanged, but an exact child duplicate is resampled up to `--duplicate-retries` times to prevent population collapse. ETGPO analyzes every sampled failure, groups them by reusable reasoning or decision errors rather than question topics, selects frequent categories to the requested coverage, and passes one identical short-guidance meta-prompt to the optimizer independently `--num-candidates` times in both QA modes. For non-reasoning QA, the target model first generates one post-hoc explanation of the most likely cause of each incorrect answer; the optimizer model then constructs the taxonomy from those explanations. These explanations are probabilistic feedback, not observed chains of thought, and are saved in `failure_feedbacks.json`.

For reasoning tasks, GreaTer and GradPO first generate a deterministic solution trace from the current source prompt. If the response contains an `<answer>...</answer>` block, its last predicted answer is replaced with the gold answer; the fixed GreaTer-style extractor is appended only when no complete answer block exists. Loss is applied only to the exact gold-answer tokens. LaTeX answers retain their original case, and answer-token loss is averaged within each problem before averaging across problems, so longer symbolic answers do not receive extra weight. The same source-prompt traces are reused during local candidate scoring; full validation regenerates each candidate's answers.

## Gradient sampling protocol

The task/model pool sizes and fixed 200-example correctness-balanced gradient
subset are recorded in
`experiment_tracking/second_stage/gradient_sampling_configuration.txt`.
The initial pools are 800/600 for OpenBookQA, 700/500 for HotpotQA, and
1,200/600 for Math, listed as Qwen/Gemma. After deterministic source-prompt
inference, gradient methods target 100 correct and 100 incorrect results. If a
bucket contains fewer than 100, the run logs a warning and uses the largest
available equal-sized subset. OpenBookQA uses these two outcome buckets rather
than eight separate answer-label/outcome buckets. GradPO-Gen and
GradPO-Gen-Random must reuse the same sampled record IDs for a controlled
comparison.

LPO does not calculate gradients and samples its training-feedback pool
uniformly at random without answer-label balancing.

`GradPO-Gen-Random` matches the rebuttal control: it uses the same target-model candidate generation and beam search as GradPO-Gen, but randomly samples from the common gradient-derived editable-region pool instead of taking the highest-gradient regions.

## Default hyperparameters

| Method | Main defaults |
| --- | --- |
| RPO | 10 iterations, snapshots at 5/10, feedback sample 100, 3 near-balanced examples with the odd slot assigned to an incorrect answer, population 10, parent temperature 1.0 |
| EvoPrompt-DE | 10 iterations, snapshots at 5/10, fixed population 5, train fitness sample 1,000, up to 3 exact-duplicate retries |
| ETGPO | 1 iteration, train errors 1,000, non-reasoning feedback limit 10,000 tokens, taxonomy batch 6, coverage 0.7, minimum 2 problems/category, at most 5 categories, 5 independent guidance generations |
| LPO | 1 iteration, train sample 512, 3 incorrect feedback examples, at most 5 locations, at most 3 words/location, 5 rewrites |
| GreaTer / TG | 1 token, default initial pool 3,000, fixed balanced gradient subset 200, gradient batch 4, proposal examples 50, top-k 25, minimum proposals 10, gradient top-mu 10, dev top-z 5, fluency weight 0.2 |
| GradPO | 1 iteration, default initial pool 3,000, fixed balanced gradient subset 200, 5 candidates/span, beam 5 with target-model synthesis, candidate-generation limit 10,000 tokens, beam-synthesis limit 10,000 tokens, expansion ratio 0.6, fluency weight 0.5 |
| GradPO Qwen | 5 spans, at most 2 target-model tokens/span |
| GradPO Gemma | 3 spans, at most 3 target-model tokens/span |

## Runners

```text
codes/run_qa_promptopt_rpo.py
codes/run_qa_promptopt_evoprompt.py
codes/run_qa_promptopt_etgpo.py
codes/run_qa_promptopt_lpo.py
codes/run_qa_promptopt_greater.py --variant greater|greater_tg
codes/run_qa_promptopt_gradpo.py --variant gen|prob|gen_random
```

Generate all commands after adjusting device arguments:

```bash
python -u codes/generate_qa_promptopt_commands.py \
  --phase first_stage \
  --backend vllm \
  --gpu-memory-utilization 0.90 \
  --output-file codes/qa_promptopt_first_stage_commands.sh

python -u codes/generate_qa_promptopt_commands.py \
  --phase second_stage \
  --backend vllm \
  --gpu-memory-utilization 0.90 \
  --output-file codes/qa_promptopt_second_stage_commands.sh
```

Run stage two only after all stage-one prompt files exist. The generator uses Qwen3-14B and Gemma3-12B as the corresponding reasoning-based optimizer models.

## Outputs

Every optimization run saves its config, initial and final prompts, candidate metrics, optimizer traces, validation predictions, and a summary. Optimization runners never load or evaluate the test split. Gradient-based stage-two runs also save their gradient, candidate, selected-region, and beam traces so the optimization process is reproducible. Post-hoc edit analysis is currently disabled.

The historical OpenBookQA-only status is retained in
`experiment_tracking/qa/first_stage_lambda1_status.txt`. The canonical
cross-dataset first-stage results, directories, source-prompt scores, and full
prompt texts are now generated under `experiment_tracking/first_stage/` with:

```bash
python -u codes/report_first_stage_results.py
```

Rerun this command whenever a tracked first-stage experiment finishes. The
reporter uses explicit canonical CODE values so stale, aborted, and superseded
runs are not included. Do not treat a score change obtained with unchanged
prompt text as a genuine prompt improvement.

The central index for persistent reports, analyses, paper notes, and processing
documentation is `experiment_tracking/README.md`. Add each new persistent note
or aggregate report there; do not index individual run artifacts or third-party
baseline files.

After all prompt choices are finalized, evaluate any saved first- or second-stage prompt over five fixed test runs:

```bash
python -u codes/run_qa_final_test_evaluation.py \
  --code openbookqa_reasoning_qwen_rpo10_final \
  --qa-mode reasoning \
  --model Qwen/Qwen3-4B \
  --backend vllm \
  --device cuda:0 \
  --prompt-file outputs/qa_prompt_optimization/reasoning/rpo/openbookqa_reasoning_qwen_rpo/prompt_iteration_10.txt
```

The final-test runner defaults to five runs with consecutive base seeds 42–46 and reports mean accuracy with population standard deviation. Use the same seeds for every prompt being compared.

Generate the 60 OpenBookQA non-reasoning second-stage attempts (30 per model) with:

```bash
python -u codes/generate_openbookqa_second_stage_commands.py
```

This writes `codes/run_openbookqa_second_stage_qwen.sh` and
`codes/run_openbookqa_second_stage_gemma.sh`. Each script uses RPO-5, RPO-10,
EvoPrompt-5, EvoPrompt-10, and the corrected ETGPO-1 prompt as its five sources,
then runs LPO, GreaTer, GreaTer-TG, GradPO-Gen, GradPO-Prob, and
GradPO-Gen-Random. Selection uses the full 1,500-example validation split and
lambda 1; test evaluation is not run.

```bash
nohup bash codes/run_openbookqa_second_stage_qwen.sh \
  > codes/nohup_outs/openbookqa_second_stage_qwen.log 2>&1 &

nohup bash codes/run_openbookqa_second_stage_gemma.sh \
  > codes/nohup_outs/openbookqa_second_stage_gemma.log 2>&1 &
```

Run the three first-stage MATH optimizers for both target-model families with:

```bash
nohup bash codes/run_math_first_stage_qwen.sh > codes/nohup_outs/math_first_stage_qwen.log 2>&1 &
nohup bash codes/run_math_first_stage_gemma.sh > codes/nohup_outs/math_first_stage_gemma.log 2>&1 &
```

The scripts use the prepared `5,999/1,500/500` train/validation/test files, Qwen3-14B or Gemma3-12B as the corresponding optimizer, vLLM, lambda 1, and `outputs/math_prompt_optimization`. The Qwen commands use deterministic 3 x 300 validation-fold prefixes; the current Gemma commands use all 3 x 500 records. They do not evaluate the 500-problem test set.

After all five source prompts exist for each model, generate the 60 MATH second-stage attempts (30 per model):

```bash
python -u codes/generate_math_second_stage_commands.py
```

This writes `codes/run_math_second_stage_qwen.sh` and `codes/run_math_second_stage_gemma.sh`. Each script checks for RPO-5, RPO-10, EvoPrompt-5, EvoPrompt-10, and ETGPO-1 before starting. LPO uses vLLM; GreaTer, GreaTer-TG, GradPO-Gen, GradPO-Prob, and GradPO-Gen-Random use Transformers because they require gradients or direct logits. Run the generated scripts only after the lightweight gradient smoke checks and the first-stage source checks pass.

Before the full matrix, select an unused GPU in `codes/run_math_second_stage_smoke.sh` and run:

```bash
bash codes/run_math_second_stage_smoke.sh
```

The smoke script uses only two training examples and one example from each validation fold. It exercises reasoning-conditioned answer gradients, GreaTer ranking, GradPO synthesis, memory allocation, and required artifact creation. Its scores are diagnostics only and must not be reported as experimental results.

After experiments finish, write the aggregate text report with:

```bash
python -u codes/report_qa_prompt_optimization.py \
  --output-root outputs/qa_prompt_optimization \
  --report-file qa_prompt_optimization_stats.txt
```
