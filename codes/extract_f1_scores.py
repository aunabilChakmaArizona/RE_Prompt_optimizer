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


def iter_f1_lines(path: str) -> Iterable[Tuple[str, float]]:
    """Yield (line, f1) for each line containing f1=."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if "f1=" not in line.lower():
                continue
            m = F1_RE.search(line)
            if not m:
                continue
            try:
                f1 = float(m.group(1))
            except ValueError:
                continue
            yield line.rstrip("\n"), f1


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
        f1s: List[float] = []
        lines: List[str] = []
        for line, f1 in iter_f1_lines(path):
            f1s.append(f1)
            lines.append(line)

        f1s = trim_to_k_after_base(f1s, args.k)
        if not args.no_lines:
            lines = lines[: len(f1s)]

        print("=" * 80)
        print(os.path.basename(path))
        if not args.no_lines:
            for line in lines:
                print(line)
        if not f1s:
            print("No f1= lines found.")
            continue

        count_better, mean_better, std_better = summarize_f1(f1s)
        base = f1s[0]
        print(f"Base f1 (first): {base}")
        print(f"Better-than-base count: {count_better}")
        if mean_better is None:
            print("Better-than-base mean/std: n/a")
        else:
            print(f"Better-than-base mean: {mean_better}")
            print(f"Better-than-base std (population): {std_better}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
