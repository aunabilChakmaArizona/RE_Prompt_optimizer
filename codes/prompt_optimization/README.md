# OpenBookQA prompt optimization

This package implements the QA-only expansion of the two-stage prompt-optimization experiments. It does not modify the existing relation-extraction runners.

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

Reasoning mode enables model thinking and defaults to 4,096 generated tokens. Non-reasoning mode disables thinking and defaults to 16 generated tokens. Both modes use the fixed Qwen3 or Gemma3 sampling settings already used by the project.

## Inference backends

The shared runners default to `--backend transformers`. Use `--backend vllm --gpu-memory-utilization 0.90` to run generated responses with vLLM continuous batching. The vLLM path uses the same chat templates, thinking flags, decoding parameters, response-list interface, and token-usage format as the Transformers path. `--target-batch-size` and `--optimizer-batch-size` remain accepted for command compatibility, but vLLM schedules all submitted prompts dynamically.

RPO, EvoPrompt-DE, ETGPO, and LPO support vLLM. GreaTer, GreaTer-TG, and all GradPO variants must use Transformers because they require backward gradients and direct model logits. The command generator automatically uses Transformers for these gradient-based methods even when `--backend vllm` is requested for the experiment matrix.

Install the separate pinned environment from `requirements_vllm.txt`. Offline vLLM uses one visible GPU per process, so the target and optimizer device arguments must match in a vLLM run.

Validation uses the fixed 900-example split as three folds of 300 examples. Prompt selection uses `mean fold accuracy - lambda * population standard deviation`, with `--validation-std-penalty 2.0` by default. Setting lambda to zero gives ordinary accuracy because the three folds have equal size. A second-stage candidate is retained only if its stable validation score strictly exceeds its first-stage source prompt; otherwise, the first-stage prompt is retained. Raw accuracy, each fold accuracy, the fold mean and standard deviation, and the stable score are all saved.

Optimizer-model generations default to `--optimizer-max-new-tokens 10000`, matching the relation-extraction experiment setting.

Generated evaluations are reseeded deterministically by mode, split, and record subset, so every candidate sees the same sampling stream and repeated prompt evaluations are reproducible.

EvoPrompt-DE starts from the source instruction plus four fixed provisional seeds in `qa_evoprompt_seeds.py`; replace the clearly labeled placeholders with the final curated seeds before the full experiment. ETGPO analyzes every sampled failure, selects frequent categories to the requested coverage, and passes one identical guidance meta-prompt to the optimizer independently `--num-candidates` times in both QA modes.

For GreaTer and GradPO, gradients are computed from teacher-forced `<answer>X</answer>` responses, but loss is applied only to the inner gold option-label token. This prevents fixed answer tags from dominating the instruction gradient.

`GradPO-Gen-Random` matches the rebuttal control: it uses the same target-model candidate generation and beam search as GradPO-Gen, but randomly samples from the common gradient-derived editable-region pool instead of taking the highest-gradient regions.

## Default hyperparameters

| Method | Main defaults |
| --- | --- |
| RPO | 10 iterations, snapshots at 5/10, feedback sample 100, separate feedback for 3 mixed examples, population 10, parent temperature 1.0 |
| EvoPrompt-DE | 10 iterations, snapshots at 5/10, fixed population 5, train fitness sample 1,000 |
| ETGPO | 1 iteration, train errors 1,000, batch 6, coverage 0.7, minimum 2 problems/category, at most 5 categories, 5 independent guidance generations |
| LPO | 1 iteration, train sample 512, 3 incorrect feedback examples, at most 5 locations, at most 3 words/location, 5 rewrites |
| GreaTer / TG | 1 token, train sample 3,000, gradient batch 4, proposal examples 50, top-k 25, minimum proposals 10, gradient top-mu 10, dev top-z 5, fluency weight 0.2 |
| GradPO | 1 iteration, train sample 3,000, 5 candidates/span, beam 5 with target-model synthesis, candidate-generation limit 10,000 tokens, beam-synthesis limit 10,000 tokens, expansion ratio 0.6, fluency weight 0.5 |
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

After experiments finish, write the aggregate text report with:

```bash
python -u codes/report_qa_prompt_optimization.py \
  --output-root outputs/qa_prompt_optimization \
  --report-file qa_prompt_optimization_stats.txt
```
