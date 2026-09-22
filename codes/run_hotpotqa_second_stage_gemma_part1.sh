#!/usr/bin/env bash
set -euo pipefail

# Gemma HotpotQA ready sources, attempts 1-9 of 18:
# all six RPO-5 refiners, then RPO-10 GradPO-Gen/Prob/Gen-Random.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export HOTPOT_GEMMA_SOURCE_SET=ready
export HOTPOT_GEMMA_START=1
export HOTPOT_GEMMA_END=9
exec bash "$SCRIPT_DIR/run_hotpotqa_second_stage_gemma.sh"
