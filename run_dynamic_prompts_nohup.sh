#!/usr/bin/env bash
set -euo pipefail
mkdir -p outputs/logs outputs/opt_prompt

# Generated from all_the_prompts.txt
# Each CODE block is passed as the instruction split; answer/input splits come from agents.agent_prompts.
# Commands run with nohup in the background.
# All commands still use cuda:0.
# Prompt blocks: 124
# Empty-status non-empty prompt blocks: 0
# Empty-status empty prompt blocks skipped: 34
# Status running/done blocks skipped: 90
# Other non-empty status blocks skipped: 0
# Exact duplicate non-empty blocks skipped: 0

