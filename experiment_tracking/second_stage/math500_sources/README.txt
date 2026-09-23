Portable MATH-500 second-stage source prompts
=============================================

These prompt copies are stored outside `outputs/` so Git-based runs on DeltaAI
and other machines do not depend on ignored first-stage output directories.
They must remain byte-for-byte consistent with the selected first-stage prompt
artifacts documented under `experiment_tracking/first_stage/`.

Qwen sources:
- qwen/evoprompt5.txt: experimental EvoPrompt-5, actual iteration 1.
- qwen/evoprompt10.txt: experimental EvoPrompt-10, actual iteration 2.
- qwen/etgpo1.txt: retained ETGPO iteration-1 prompt.
