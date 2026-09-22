HOTPOTQA SECOND-STAGE SOURCE PROMPTS
====================================

These are portable, exact copies of the retained first-stage instruction
prompts used by the HotpotQA second-stage shell scripts. The fixed answer
instruction is not included because the QA renderer appends it separately.

Qwen sources
------------
- qwen/rpo5.txt
- qwen/rpo10.txt
- qwen/evoprompt5_10_unchanged.txt
  The original iteration-5, iteration-10, final, and initial files are
  byte-identical. Its six physical second-stage runs represent both logical
  EvoPrompt rows and must be marked as unchanged first-stage sources.
- qwen/etgpo1.txt

Gemma sources
-------------
- gemma/rpo5.txt
- gemma/rpo10.txt
- gemma/evoprompt5.txt
- gemma/evoprompt10.txt
- gemma/etgpo1.txt

Original run directories and first-stage scores remain documented in:
- experiment_tracking/first_stage/detailed_first_stage_results.txt
- experiment_tracking/first_stage/first_stage_summary_and_prompts.txt
