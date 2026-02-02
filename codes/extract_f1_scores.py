#!/usr/bin/env python3
"""Extract lines containing f1= from nohup_*.out and summarize improvements."""

import glob
import os
import re
import statistics
DIR = "/storage2/home/aunabilchakma/codes/RE_Prompt_optimizer/codes"
PATTERN = "nohup_*.out"
K = 10  # set -1 for all


def extract_metric(line: str, name: str):
    m = re.search(
        rf"{name}\s*=\s*([0-9]*\.?[0-9]+)(?:\s*(?:±|\+/-|\+-)\s*([0-9]*\.?[0-9]+))?",
        line,
        re.IGNORECASE,
    )
    if not m:
        return None, None
    mean = float(m.group(1))
    std = float(m.group(2)) if m.group(2) is not None else 0.0
    return mean, std


def format_mean_std(mean, std):
    if mean is None:
        return "n/a"
    return f"{mean:.1f}±{std:.2f}"


def iter_entries(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if "f1=" not in line.lower():
                continue
            f1_mean, f1_std = extract_metric(line, "f1")
            if f1_mean is None:
                continue
            p_mean, p_std = extract_metric(line, "precision")
            r_mean, r_std = extract_metric(line, "recall")
            yield {
                "line": line.rstrip("\n"),
                "f1_mean": f1_mean,
                "f1_std": f1_std,
                "precision_mean": p_mean,
                "precision_std": p_std,
                "recall_mean": r_mean,
                "recall_std": r_std,
            }


def summarize_f1(f1s):
    if not f1s:
        return 0, None, None
    base = f1s[0]
    better = [v for v in f1s[1:] if v > base]
    if not better:
        return 0, None, None
    return len(better), statistics.mean(better), statistics.pstdev(better)


files = sorted(glob.glob(os.path.join(DIR, PATTERN)))
if not files:
    print(f"No files matched: {os.path.join(DIR, PATTERN)}")

for path in files:
    entries = list(iter_entries(path))
    if K is not None and K >= 0:
        entries = entries[: 1 + K]

    f1s = [e["f1_mean"] for e in entries]

    print("=" * 80)
    print(os.path.basename(path))
    if not f1s:
        print("No f1= lines found.")
        continue

    base = f1s[0]
    k_entries = entries[1:]
    k_values = [e["f1_mean"] for e in k_entries]
    best = max(k_entries, key=lambda e: e["f1_mean"]) if k_entries else None

    print(f"Base f1 (first): {base}")
    if best is None:
        print("Best precision/recall/f1: n/a")
    else:
        best_p = format_mean_std(best["precision_mean"], best["precision_std"])
        best_r = format_mean_std(best["recall_mean"], best["recall_std"])
        best_f = format_mean_std(best["f1_mean"], best["f1_std"])
        print(f"Best precision/recall/f1: {best_p}\t{best_r}\t{best_f}")

    count_better, mean_better, std_better = summarize_f1(f1s)
    print(f"Better-than-base count: {count_better}")
    if mean_better is None:
        print("Better-than-base mean/std: n/a")
    else:
        print(f"Better-than-base mean: {mean_better}")
        print(f"Better-than-base std (population): {std_better}")
        print(f"Gain over base (better only): {mean_better - base:.2f}")

    mean_k = statistics.mean(k_values) if k_values else None
    if mean_k is None:
        print("Gain over base (all K): n/a")
    else:
        print(f"Gain over base (all K): {mean_k - base:.2f}")

    better_gain = (mean_better - base) if mean_better is not None else None
    all_k_gain = (mean_k - base) if mean_k is not None else None
    better_gain_str = f"{better_gain:.2f}" if better_gain is not None else "n/a"
    all_k_gain_str = f"{all_k_gain:.2f}" if all_k_gain is not None else "n/a"
    print(f"{count_better}\t{better_gain_str}\t{all_k_gain_str}")
