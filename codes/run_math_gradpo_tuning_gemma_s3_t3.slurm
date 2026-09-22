#!/usr/bin/env bash
#SBATCH --job-name=math-gemma-small
#SBATCH --account=bihn-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=codes/nohup_outs/math_gradpo_gemma_small_%j.log
#SBATCH --error=codes/nohup_outs/math_gradpo_gemma_small_%j.log

# Submit from the repository root with:
#   sbatch codes/run_math_gradpo_tuning_gemma_s3_t3.slurm
# Do not set CUDA_VISIBLE_DEVICES: Slurm owns the assigned GPU.
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

# Verify the environment without downloading model weights.
python -c 'import importlib.metadata as m, platform, torch; print("Architecture:", platform.machine()); print("Packages:", {name: m.version(name) for name in ("vllm", "torch", "transformers", "math-verify")}); assert m.version("vllm") == "0.11.0", "Candidate logprob scoring requires vLLM 0.11.0"; assert torch.cuda.is_available(), "CUDA PyTorch or allocated GPU unavailable"'

# Verify image-enabled Gemma generation before starting the expensive matrix.
SMOKE_CODE="deltaai_gemma_small_edits_images_enabled_${SLURM_JOB_ID}"
srun --ntasks=1 python -u codes/run_math_test_inference.py \
  --code "$SMOKE_CODE" \
  --model google/gemma-3-4b-it \
  --dataset data/processed/math500/validation.jsonl \
  --backend vllm \
  --device cuda:0 \
  --gpu_memory_utilization 0.50 \
  --max_new_tokens 4096 \
  --start 0 \
  --end 10 \
  --prompt "$(cat experiment_tracking/second_stage/math_gemma_sources/rpo10.txt)" \
  --output_dir outputs/deltaai_diagnostics \
  --overwrite

python - "$SMOKE_CODE" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path("outputs/deltaai_diagnostics") / sys.argv[1] / "summary.json"
summary = json.loads(summary_path.read_text(encoding="utf-8"))
statistics = summary["statistics"]
missing = int(statistics["missing_answer_tags"])
average_tokens = float(statistics["token_usage"]["averages"]["output_tokens"])
print(
    f"[deltaai-smoke] missing_answer_tags={missing}/10 "
    f"average_output_tokens={average_tokens:.2f}"
)
if missing == 10 or average_tokens >= 3891.2:
    raise SystemExit(
        "Gemma/vLLM smoke test failed; full ten-run matrix was not started."
    )
PY

# Run all ten attempts sequentially on the assigned GPU.
srun --ntasks=1 bash codes/run_math_gradpo_tuning_gemma_s3_t3.sh
