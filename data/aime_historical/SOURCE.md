# Historical AIME datasets

Retrieved on 2026-08-28. Every Hugging Face source is pinned to the exact
revision listed below. Upstream README files are stored beside the data.

## Downloaded collections

| Local directory | Upstream dataset | Revision | Upstream rows | Contents |
|---|---|---|---:|---|
| `sources/pandores_aime_1983_2025` | `Pandores/aime-1983-2025` | `292f1a1b4f041918afced4435127c7eda48a2597` | 1,035 | Complete 1983–2025 problem collection with solutions; 1,005 rows are before 2025 |
| `sources/ai_mo_aime_2022_2024` | `AI-MO/aimo-validation-aime` | `13f9e12f613e720c2a2b2f345dd04b998a29494d` | 90 | Thirty problems per year for 2022–2024, with solutions |
| `sources/huggingfaceh4_aime_2024` | `HuggingFaceH4/aime_2024` | `2fe88a2f1091d5048c0f36abc874fb997b3dd99a` | 30 | AIME I and II 2024, with solutions |
| `sources/maxwell_jia_aime_2024` | `Maxwell-Jia/AIME_2024` | `8d88b2876a82a080e2f172cc9b25d0d9d2cb4792` | 30 | Alternative AIME I and II 2024 representation, with solutions |
| `sources/philschmid_aime_1983_2024` | `philschmid/AIME_1983_2024` | `34e18402a7c28461ce46c2a6dd4969c4f6130e9c` | 933 | Incomplete historical CSV; it has only 14 rows for 2024 |
| `sources/sxiong_aime_trajectory` | `sxiong/AIME-trajectory` | `7fc7a89aaae52d114a7a1819088e048a7336cc02` | 1,438 trajectories | 1,258 verified-correct train trajectories over 875 unique 1983–2023 problems; 180 trajectories over 30 unique 2024 problems |

These row counts must not be added together: the collections substantially
overlap. For a unique problem corpus, use the Pandores collection and filter
by year. The other copies are retained for solution/format comparison and for
the generated reasoning trajectories.

## Validation

- All four Parquet files have valid `PAR1` headers and footers and match the
  pinned upstream checksums.
- The philschmid CSV parses as 933 rows with 933 unique IDs.
- The trajectory JSONL files parse successfully. Train has 1,258 rows over 875
  unique problems, all marked correct. Test has 180 rows over 30 unique 2024
  problems, of which six trajectories are marked correct.
- SHA-256 values are recorded in `CHECKSUMS.sha256`.

## Recommended experimental split

When using AIME 2025 as the final benchmark:

| Experimental split | Years | Unique problems |
|---|---|---:|
| Optimization/train | 1983–2021 | 915 |
| Prompt selection/validation | 2022–2024 | 90 |
| Final test | 2025 | 30 |

The source repositories often call their only storage split `train`; that is
not an official experimental assignment. Apply the year-based split above and
never use the 2025 rows from the historical Pandores file for optimization.

## Rights note

Licenses and metadata vary across these community datasets. AIME problem
statements are owned by the Mathematical Association of America. Review each
upstream README and the intended research/educational use before redistribution.
