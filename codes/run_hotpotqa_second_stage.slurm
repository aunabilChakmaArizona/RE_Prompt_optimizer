#!/usr/bin/env bash
#SBATCH --job-name=hotpot-stage2
#SBATCH --account=bihn-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --array=0-1
#SBATCH --output=codes/nohup_outs/hotpotqa_second_stage_%A_%a.log
#SBATCH --error=codes/nohup_outs/hotpotqa_second_stage_%A_%a.log

# Submit both model jobs from the repository root. By default each model runs
# only its ready RPO-5, RPO-10, and ETGPO-1 sources (18 attempts per model):
#   sbatch codes/run_hotpotqa_second_stage.slurm
# Array task 0 runs Qwen; task 1 runs Gemma. Each task receives one GPU.

set -euo pipefail
cd -- "${SLURM_SUBMIT_DIR:?Submit this file from the repository root}"

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

python -c 'import importlib.metadata as m, platform, torch; print("Architecture:", platform.machine()); print("Packages:", {name: m.version(name) for name in ("vllm", "torch", "transformers")}); assert m.version("vllm") == "0.11.0"; assert torch.cuda.is_available()'

case "${SLURM_ARRAY_TASK_ID:?Missing Slurm array index}" in
  0)
    printf '[hotpot-slurm] starting Qwen second-stage matrix\n'
    srun --ntasks=1 bash codes/run_hotpotqa_second_stage_qwen.sh
    ;;
  1)
    if [[ -z "${HF_TOKEN:-}" ]]; then
      printf 'HF_TOKEN is unset; Gemma access requires Hugging Face authentication.\n' >&2
      exit 2
    fi
    printf '[hotpot-slurm] starting Gemma second-stage matrix with image support enabled\n'
    srun --ntasks=1 bash codes/run_hotpotqa_second_stage_gemma.sh
    ;;
  *)
    printf 'Unexpected SLURM_ARRAY_TASK_ID=%s\n' "$SLURM_ARRAY_TASK_ID" >&2
    exit 2
    ;;
esac
