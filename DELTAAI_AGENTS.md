# Instructions for DeltaAI agents

This file is dedicated to Codex/AI agents working on this project on NCSA
DeltaAI. It supplements, but does not replace, the root `AGENTS.md`.

## Required reading order

1. Read `AGENTS.md` for the project-wide coding, experiment, environment, and
   tracking rules.
2. Read `experiment_tracking/second_stage/deltaai_gemma_vllm_failure_handoff.txt`
   before diagnosing or rerunning Gemma MATH-500 GradPO on DeltaAI. It contains
   the complete incident evidence, invalid-run table, cache paths, hypotheses,
   diagnostic order, and acceptance criteria.
3. Read `experiment_tracking/second_stage/math_gradpo_gemma_slurm_guidelines.txt`
   for the approved Gemma tuning matrix and DeltaAI launch procedure.
4. Use `experiment_tracking/README.md` as the central index for the rest of the
   project's experiment notes and result files.

## DeltaAI environment

- Conda environment: `re_prompt_optimization_vllm_v2`
- Slurm file: `codes/run_math_gradpo_tuning_gemma.slurm`
- Run script: `codes/run_math_gradpo_tuning_gemma.sh`
- Model: `google/gemma-3-4b-it`
- Activate the environment inside the compute job, then run
  `unset LD_LIBRARY_PATH` before importing PyTorch or vLLM. Do not put this
  unset command in `~/.bashrc`.
- Let Slurm manage GPU visibility; do not set `CUDA_VISIBLE_DEVICES` in the
  Slurm job.

## Current incident status

The archived DeltaAI eight-run Gemma MATH-500 GradPO matrix completed at the
process level but is invalid because vLLM produced corrupted Gemma text. Do not
report its scores and do not launch the full matrix again yet.

Important evidence in the handoff includes:

- RPO5 replacement-candidate generation produced garbage, so all four searches
  were no-ops.
- RPO10 source validation collapsed to 1/900 correct, with most outputs empty
  or reaching 4,096 tokens.
- The corrupt RPO10 source result was cached and reused.
- The source prompt files themselves match the intended files byte-for-byte.

Follow the handoff's small A/B diagnostics first. Preserve existing artifacts;
do not delete caches or run outputs. Use fresh debug cache roots during
diagnosis. After a runtime configuration is verified, refresh both the source
validation and gradient caches before a real rerun.

## Relevant paths

- Incident handoff:
  `experiment_tracking/second_stage/deltaai_gemma_vllm_failure_handoff.txt`
- DeltaAI setup notes:
  `experiment_tracking/deltaai_setup_step_by_step.txt`
- Gemma Slurm guidance:
  `experiment_tracking/second_stage/math_gradpo_gemma_slurm_guidelines.txt`
- Portable Gemma sources:
  `experiment_tracking/second_stage/math_gemma_sources/rpo5.txt`
  `experiment_tracking/second_stage/math_gemma_sources/rpo10.txt`
- Source-validation cache implementation:
  `codes/prompt_optimization/source_validation_cache.py`
- Training/gradient cache implementation:
  `codes/prompt_optimization/gradient_cache.py`
- vLLM loader:
  `codes/agents/agent_vllm_models.py`
- vLLM prompting:
  `codes/agents/agent_vllm_prompting.py`
- GradPO flow:
  `codes/prompt_optimization/second_stage.py`
- vLLM objective scoring:
  `codes/prompt_optimization/vllm_scoring.py`

Record diagnostic commands, exact package/model revisions, raw token evidence,
and conclusions in the incident handoff or an adjacent tracked report. Clearly
separate confirmed evidence from hypotheses.
