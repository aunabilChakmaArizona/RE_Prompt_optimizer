# MATH data sources

## Training and validation source

- Dataset: `DigitalLearningGmbH/MATH-lighteval`
- Upstream dataset: Hendrycks et al. MATH
- Revision: `0530c78699ea5e8eb5530600900e1f328b48acad`
- Source file: `data/train-00000-of-00001.parquet`
- Local raw file: `data/math500/original/math_train.parquet`
- SHA-256: `eca6e667f4305dd5e5ba09b4fd55e7f3174a0fbe361cdfd4c44758b593a76933`
- URL: <https://huggingface.co/datasets/DigitalLearningGmbH/MATH-lighteval/resolve/0530c78699ea5e8eb5530600900e1f328b48acad/data/train-00000-of-00001.parquet>

The raw source contains 7,500 official MATH training problems. Processing removes
one exact duplicate question and creates a deterministic 6,999/500
train/validation split. The split uses seed 42 and is stratified by subject and
difficulty level. Four source solutions have nonstandard final boxes: two use
unbraced boxes, and two empty boxes correspond to an answer of zero. These cases
are handled explicitly by `codes/prepare_math500_training.py`.

## Test source

- Dataset: `HuggingFaceH4/MATH-500`
- Revision: `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be`
- Local raw file: `data/math500/original/test.jsonl`
- SHA-256: `35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132`
- URL: <https://huggingface.co/datasets/HuggingFaceH4/MATH-500/resolve/6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be/test.jsonl>

MATH-500 remains an untouched 500-problem test set. Normalized questions are
checked to ensure that neither the processed training nor validation split
overlaps with MATH-500.
