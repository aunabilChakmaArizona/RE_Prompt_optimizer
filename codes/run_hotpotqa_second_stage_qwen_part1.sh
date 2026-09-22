#!/usr/bin/env bash
set -euo pipefail

# Qwen HotpotQA ready sources, attempts 1-9 of 18:
# all six RPO-5 refiners, then RPO-10 GradPO-Gen/Prob/Gen-Random.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export HOTPOT_QWEN_SOURCE_SET=ready
export HOTPOT_QWEN_START=1
export HOTPOT_QWEN_END=9
exec bash "$SCRIPT_DIR/run_hotpotqa_second_stage_qwen.sh"
