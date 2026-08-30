# AIME 2025 dataset

## Source and provenance

- Upstream dataset: <https://huggingface.co/datasets/opencompass/AIME2025>
- Pinned revision: `a6ad95f611d72cf628a80b58bd0432ef6638f958`
- Retrieved: 2026-08-28
- Upstream license metadata: MIT
- Upstream configurations: `AIME2025-I` and `AIME2025-II`, both marked as
  test splits

The two upstream JSONL files and dataset card are preserved unchanged as
`aime2025-I.jsonl`, `aime2025-II.jsonl`, and `README.upstream.md`.

## Files and checksums

| File | Records | SHA-256 |
|---|---:|---|
| `aime2025-I.jsonl` | 15 | `b91b3c96f05d9635d2a0692b124ebe023c1ff59cb19c074275e6c4b349d0659e` |
| `aime2025-II.jsonl` | 15 | `16a2dcfbbf9db1b11f8a69a3ba5e4cac73e3641b19a37e2307e9c12240bbed5e` |
| `README.upstream.md` | n/a | `43ac9ef26311be77671372031a242d031858ba836a6d79f323a1bac748e012ac` |
| `aime2025.jsonl` (normalized) | 30 | `9bcf5a2c5e60d667694a33789ca7cb7c407372cc650263caf804114c3b5de2f9` |

## Normalized file

Run `python3 build_normalized.py` to create `aime2025.jsonl`. It combines the
two exams, adds stable IDs and exam metadata, validates all 30 records, and
normalizes answers to digit strings suitable for exact-match evaluation.

The upstream answer for AIME II problem 5 is `336^\circ`, because the question
asks for an arc measure in degrees. The normalized file stores the required
integer answer as `336`. This value is also reported by an independent AIME
2025 dataset: <https://huggingface.co/datasets/pe-nlp/DAPO-AIME-2025>.

## Evaluation note

Treat all 30 AIME 2025 problems as a final test set. Do not use them to select
or optimize prompts when reporting AIME 2025 performance. Use earlier,
non-overlapping math problems for optimization and validation to avoid test
leakage.
