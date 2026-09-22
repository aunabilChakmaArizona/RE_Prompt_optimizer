#!/usr/bin/env bash
set -euo pipefail

# Gemma HotpotQA ready sources, attempts 10-18 of 18:
# RPO-10 LPO/GreaTer/GreaTer-TG, then all six ETGPO-1 refiners.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export HOTPOT_GEMMA_SOURCE_SET=ready
export HOTPOT_GEMMA_START=10
export HOTPOT_GEMMA_END=18
exec bash "$SCRIPT_DIR/run_hotpotqa_second_stage_gemma.sh"
