---
dataset_info:
  features:
  - name: year
    dtype: int64
  - name: index
    dtype: int64
  - name: part
    dtype: string
  - name: problem
    dtype: string
  - name: solutions
    list: string
  - name: answer
    dtype: int64
  - name: all_answers
    list: int64
  - name: note
    dtype: string
  splits:
  - name: train
    num_bytes: 4967742
    num_examples: 1035
  download_size: 4755884
  dataset_size: 4967742
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
task_categories:
- question-answering
language:
- en
tags:
- math
- aime
pretty_name: AIME
size_categories:
- 1K<n<10K
---

# AIME Datasets from 1983 to 2025
This dataset contains the AIME datasets from 1983 to 2025.

## Features Description
|Feature|Description|Example|
|:---|:---|:----|
|`year`|The year this problem was released. From 1983 to 2025.|2022|
|`index`|The index of the problem for a year and part. From 1 to 15.|12|
|`part`|The dataset part if this dataset has multiple parts. Can be `AIME`, `AIME I`, `AIME II` or None. Datasets have multiple parts since the year 2000.|`AIME I`|
|`problem`|The problem description in Latex.|Let $x$ ...|
|`solutions`|A list of human-made solutions for this problem.| Let $x$ ...|
|`answer`|The answer to the problem.|46|
|`all_answers`|All the answers. Problems usually have a single solution, but some questions can accept secondary solutions due to ambiguity.|[46, 102]|
|`note`|Optional note concerning the problem or solutions.|Note that some of these solutions assume that $R$ ...|

## Example
```python
from datasets import load_dataset

dataset = load_dataset("Pandores/aime-1983-2025")

print(dataset["train"][0])
```

## Data Source
This content is derived from the **AIME Problems and Solutions** page on the [Art of Problem Solving (AoPS) Wiki](https://artofproblemsolving.com/wiki/index.php/AIME_Problems_and_Solutions).

**NOTE**: Please check the AoPS Wiki's official terms for the specific version of the license that applies.