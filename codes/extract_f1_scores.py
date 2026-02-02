#!/usr/bin/env python3
"""Extract lines containing f1= from nohup_*.out and summarize improvements."""
from __future__ import annotations

import argparse
import glob
import os
import re
import statistics
from typing import Iterable, List, Tuple

F1_RE = re.compile(r"f1\s*=\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)
PM_RE = re.compile(r"(?:±|\\+/-|\\+-)")


def extract_metric(line: str, name: str) -> Tuple[float | None, float | None]:
    pattern = rf"{name}\s*=\s*([0-9]*\.?[0-9]+)(?:\s*(?:±|\+/-|\+-)\s*([0-9]*\.?[0-9]+))?"
    m = re.search(pattern, line, re.IGNORECASE)
    if not m:
        return None, None
    try:
        mean = float(m.group(1))
    except ValueError:
        return None, None
    std = None
    if m.group(2) is not None:
        try:
            std = float(m.group(2))
        except ValueError:
            std = None
    return mean, std


def iter_f1_lines(path: str) -> Iterable[dict]:
    """Yield metrics dicts for each line containing f1=."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if "f1=" not in line.lower():
                continue
            f1_mean, f1_std = extract_metric(line, "f1")
            if f1_mean is None:
                continue
            precision_mean, precision_std = extract_metric(line, "precision")
            recall_mean, recall_std = extract_metric(line, "recall")
            yield {
                "line": line.rstrip("\n"),
                "f1_mean": f1_mean,
                "f1_std": f1_std,
                "precision_mean": precision_mean,
                "precision_std": precision_std,
                "recall_mean": recall_mean,
                "recall_std": recall_std,
            }


def summarize_f1(f1s: List[float]) -> Tuple[int, float | None, float | None]:
    """Return (count_better, mean_better, std_better) vs first f1."""
    if not f1s:
        return 0, None, None
    base = f1s[0]
    better = [v for v in f1s[1:] if v > base]
    if not better:
        return 0, None, None
    mean = statistics.mean(better)
    # Use population std to avoid ambiguity for small n.
    std = statistics.pstdev(better)
    return len(better), mean, std


def trim_to_k_after_base(f1s: List[float], k: int | None) -> List[float]:
    """Keep base + next k entries (if k is None or < 0, keep all)."""
    if k is None or k < 0:
        return f1s
    if not f1s:
        return f1s
    return f1s[: 1 + k]


def format_mean_std(mean: float | None, std: float | None) -> str:
    if mean is None:
        return "n/a"
    if std is None:
        std = 0.0
    return f"{mean:.1f}±{std:.2f}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract f1= lines from nohup_*.out and summarize improvements."
    )
    parser.add_argument(
        "--dir",
        default=os.path.dirname(__file__),
        help="Directory to search (default: script directory).",
    )
    parser.add_argument(
        "--pattern",
        default="nohup_*.out",
        help="Glob pattern to match files (default: nohup_*.out).",
    )
    parser.add_argument(
        "--no-lines",
        action="store_true",
        help="Skip printing matched lines; only show summary.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=-1,
        help="Only analyze the first K f1s after the base (default: all).",
    )
    args = parser.parse_args()

    pattern = os.path.join(args.dir, args.pattern)
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No files matched: {pattern}")
        return 1

    for path in files:
        entries = list(iter_f1_lines(path))

        if args.k is not None and args.k >= 0:
            entries = entries[: 1 + args.k]
        f1s = [e["f1_mean"] for e in entries]
        if not args.no_lines:
            lines = [e["line"] for e in entries]

        print("=" * 80)
        print(os.path.basename(path))
        # if not args.no_lines:
        #     for line in lines:
        #         print(line)
        if not f1s:
            print("No f1= lines found.")
            continue

        count_better, mean_better, std_better = summarize_f1(f1s)
        base = f1s[0]
        k_entries = entries[1:]
        best_entry = max(k_entries, key=lambda e: e["f1_mean"]) if k_entries else None
        print(f"Base f1 (first): {base}")
        if best_entry is None:
            print("Best f1 among K: n/a")
        else:
            best_f1 = format_mean_std(best_entry["f1_mean"], best_entry["f1_std"])
            best_precision = format_mean_std(
                best_entry["precision_mean"], best_entry["precision_std"]
            )
            best_recall = format_mean_std(
                best_entry["recall_mean"], best_entry["recall_std"]
            )
            print(
                f"Best precision/recall/f1: {best_precision}\t{best_recall}\t{best_f1}"
            )
        print(f"Better-than-base count: {count_better}")
        if mean_better is None:
            print("Better-than-base mean/std: n/a")
        else:
            print(f"Better-than-base mean: {mean_better}")
            print(f"Better-than-base std (population): {std_better}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
