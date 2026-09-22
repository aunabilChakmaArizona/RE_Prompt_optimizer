#!/usr/bin/env bash
set -euo pipefail

# Qwen HotpotQA ready sources, attempts 10-18 of 18:
# RPO-10 LPO/GreaTer/GreaTer-TG, then all six ETGPO-1 refiners.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export HOTPOT_QWEN_SOURCE_SET=ready
export HOTPOT_QWEN_START=10
export HOTPOT_QWEN_END=18
exec bash "$SCRIPT_DIR/run_hotpotqa_second_stage_qwen.sh"
