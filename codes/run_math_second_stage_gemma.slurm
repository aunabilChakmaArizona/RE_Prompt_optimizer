#!/usr/bin/env bash
#SBATCH --job-name=math-gemma-rest
#SBATCH --account=bihn-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=codes/nohup_outs/math_second_stage_gemma_%j.log
#SBATCH --error=codes/nohup_outs/math_second_stage_gemma_%j.log

# Submit from the repository root with:
#   sbatch codes/run_math_second_stage_gemma.slurm
# Gemma 3 image support intentionally remains enabled for vLLM 0.11.0.

set -euo pipefail
cd -- "${SLURM_SUBMIT_DIR:?Submit this file with sbatch from the repository root}"

if [[ -n "${CONDA_MODULE:-}" ]]; then
  module load "$CONDA_MODULE"
fi
if [[ -n "${CONDA_SH:-}" ]]; then
  source "$CONDA_SH"
elif command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
else
  printf 'Set CONDA_SH to conda.sh or CONDA_MODULE to the site module.\n' >&2
  exit 2
fi

conda activate "${CONDA_ENV:-re_prompt_optimization_vllm_v2}"
unset LD_LIBRARY_PATH
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

if [[ -z "${HF_TOKEN:-}" ]]; then
  printf 'HF_TOKEN is unset; Gemma model access requires Hugging Face authentication.\n' >&2
  exit 2
fi

python -c 'import importlib.metadata as m, platform, torch; print("Architecture:", platform.machine()); print("Packages:", {name: m.version(name) for name in ("vllm", "torch", "transformers", "math-verify")}); assert m.version("vllm") == "0.11.0"; assert torch.cuda.is_available()'

# Slurm owns CUDA_VISIBLE_DEVICES. One task runs all 14 attempts sequentially.
unset MATH_GEMMA_SECOND_STAGE_GPU
srun --ntasks=1 bash codes/run_math_second_stage_gemma.sh
