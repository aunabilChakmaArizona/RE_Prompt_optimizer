# OpenBookQA dataset

## Recommended copy

The original AllenAI release is stored in:

`original/OpenBookQA-V1-Sep2018/`

It contains the original JSONL/TSV splits and both knowledge collections. The
archive was downloaded from the URL used by AllenAI's official preparation
script:

<https://s3-us-west-2.amazonaws.com/ai2-website/data/OpenBookQA-V1-Sep2018.zip>

- Archive SHA-256: `82368cf05df2e3b309c17d162e10b888b4d768fad6e171e0a041954c8553be46`
- Main splits: 4,957 train, 500 dev, and 500 test questions
- Unique questions across splits: 5,957
- `Data/Main/openbook.txt`: 1,326 core science facts
- `Data/Additional/crowdsourced-facts.txt`: 5,166 additional facts

All JSONL files were parsed and validated. Every main-split record has four
choices and an answer key matching one of its choice labels.

## Hugging Face copy

For compatibility with the Hugging Face representation, both `main` and
`additional` configurations are also stored as Parquet files. The `additional`
configuration contains the same questions plus the associated `fact1`, human
score, clarity, and anonymized annotator metadata.

- Dataset: <https://huggingface.co/datasets/allenai/openbookqa>
- Pinned revision: `388097ea7776314e93a529163e0fea805b8a6454`
- Retrieved: 2026-08-28

| Configuration | Split | Records | SHA-256 |
|---|---|---:|---|
| main | train | 4,957 | `98148f8a54e62eb862346a75192d5fb824d6cbb68f2f59aecd793d39ecb5cd8b` |
| main | validation | 500 | `35370b9cfee8c1ff325ccc74adc434d12c47ca0ac3244aa87f3fa77069285206` |
| main | test | 500 | `cd5483e366daa230c1c87bbdc512d8b7229f14f6dd04d19fc8b1a3855aaaa8a3` |
| additional | train | 4,957 | `d16d719e87efb86ed0a2ac4c8cdf380f7bfb94b602088393674c0a64ce9ed3d3` |
| additional | validation | 500 | `92e5e68e4da7bec7d130d925385abf377c2d82b89a16de502b4e1b9cf3f50a26` |
| additional | test | 500 | `33b318ea8e2354484868bc601c1b30a58149e9deb93162ff422bb8de980c7105` |

## License note

The Hugging Face dataset card currently marks the dataset license as unknown.
AllenAI's official OpenBookQA code repository is Apache-2.0; its license is
preserved here as `LICENSE.official-repo`, pinned to repository revision
`b51971646e9371a61508d9953fc706645e194a71`. Confirm dataset licensing with
AllenAI before redistribution if needed.

## Prompt-optimization split

After removing five exact training duplicates, processing moves 400 training
questions into validation with seed 42 and equal selection across answer labels.
This creates 4,552 training, 900 validation, and 500 test questions. Validation
contains the official 500 questions plus the 400-question training supplement and
is divided into three fixed folds of 300, balanced by source and answer label.

For prompt-optimization experiments, use train examples for gradient/error
signals, validation for stable prompt selection, and test only for final
reporting. The standard metric is multiple-choice accuracy. The `additional`
configuration can support separate closed-book and fact-provided prompt settings.
