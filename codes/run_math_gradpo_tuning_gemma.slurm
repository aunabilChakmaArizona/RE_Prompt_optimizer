#!/usr/bin/env bash
#SBATCH --job-name=math-gemma-gradpo
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=codes/nohup_outs/math_gradpo_gemma_%j.log
#SBATCH --error=codes/nohup_outs/math_gradpo_gemma_%j.log

# Submit from the repository root; pass your allocation with sbatch --account.
# DeltaAI: --partition=ghx4 (ARM GH200/H100, needs an ARM-compatible environment).
# Delta H200: override using sbatch --partition=gpuH200x8.
# Do NOT set CUDA_VISIBLE_DEVICES or CUDA_DEVICE_ORDER here: Slurm owns the GPU.
set -euo pipefail
cd -- "${SLURM_SUBMIT_DIR:?Submit this file using sbatch from the repository root}"

if [[ -n "${CONDA_MODULE:-}" ]]; then
  module load "$CONDA_MODULE"
fi
if [[ -n "${CONDA_SH:-}" ]]; then
  source "$CONDA_SH"
elif command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
else
  printf 'Set CONDA_SH to your conda.sh path, or CONDA_MODULE to your site module.\n' >&2
  exit 2
fi
conda activate "${CONDA_ENV:-re_prompt_optimization_vllm_v2}"
unset LD_LIBRARY_PATH
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

# Check the selected environment without downloading/loading model weights.
python -c 'import importlib.metadata as m, platform, torch; print("Architecture:", platform.machine()); print("Packages:", {name: m.version(name) for name in ("vllm", "torch", "transformers", "math-verify")}); assert m.version("vllm") == "0.11.0", "Candidate logprob scoring requires vLLM 0.11.0"; assert torch.cuda.is_available(), "CUDA PyTorch or allocated GPU unavailable"'

# One Slurm task runs all eight attempts sequentially on its one assigned GPU.
srun --ntasks=1 bash codes/run_math_gradpo_tuning_gemma.sh
