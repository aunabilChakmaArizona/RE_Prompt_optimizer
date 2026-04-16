#!/usr/bin/env python3
"""Print precision, recall, and F1 from pasted agent_evaluate log lines."""

import re


# Paste the evaluate_fn log lines here.
LOG_TEXT = """
[agent_evaluate] evaluate_fn: done in 3625.22s, precision=31.68±3.90, recall=34.41±4.08, f1=32.99±3.99
[agent_evaluate] evaluate_fn: done in 4976.74s, precision=33.55±2.63, recall=27.58±0.74, f1=30.23±1.22
[agent_evaluate] evaluate_fn: done in 4910.16s, precision=25.97±2.83, recall=43.00±4.95, f1=32.38±3.60
[agent_evaluate] evaluate_fn: done in 4943.51s, precision=28.86±4.72, recall=33.02±3.93, f1=30.79±4.39
[agent_evaluate] evaluate_fn: done in 4735.97s, precision=26.65±4.59, recall=36.89±5.30, f1=30.94±4.96
[agent_evaluate] evaluate_fn: done in 4665.73s, precision=30.18±3.78, recall=35.15±3.57, f1=32.47±3.70
[agent_evaluate] evaluate_fn: done in 4372.14s, precision=23.26±2.88, recall=44.53±4.88, f1=30.53±3.56
[agent_evaluate] evaluate_fn: done in 3202.49s, precision=20.23±4.31, recall=47.23±8.23, f1=28.31±5.70
[agent_evaluate] evaluate_fn: done in 3322.92s, precision=36.76±4.40, recall=30.84±2.87, f1=33.53±3.52
[agent_evaluate] evaluate_fn: done in 3229.77s, precision=33.24±3.87, recall=32.25±3.91, f1=32.74±3.89
[agent_evaluate] evaluate_fn: done in 3988.09s, precision=27.05±2.85, recall=40.07±4.00, f1=32.29±3.33
"""


METRIC_RE = re.compile(
    r"precision\s*=\s*(?P<precision>[0-9]*\.?[0-9]+)"
    r"(?:\s*(?:±|\+/-|\+-)\s*[0-9]*\.?[0-9]+)?"
    r".*?"
    r"recall\s*=\s*(?P<recall>[0-9]*\.?[0-9]+)"
    r"(?:\s*(?:±|\+/-|\+-)\s*[0-9]*\.?[0-9]+)?"
    r".*?"
    r"f1\s*=\s*(?P<f1>[0-9]*\.?[0-9]+)"
    r"(?:\s*(?:±|\+/-|\+-)\s*[0-9]*\.?[0-9]+)?",
    re.IGNORECASE,
)


def main():
    for line in LOG_TEXT.splitlines():
        match = METRIC_RE.search(line)
        if not match:
            continue

        precision = float(match.group("precision"))
        recall = float(match.group("recall"))
        f1 = float(match.group("f1"))
        print(f"{precision:.1f}\t{recall:.1f}\t{f1:.1f}\t")


if __name__ == "__main__":
    main()
