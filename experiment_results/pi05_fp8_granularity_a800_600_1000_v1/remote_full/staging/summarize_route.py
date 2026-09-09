#!/usr/bin/env python3
"""Summarize a frozen FP8 per-tensor route as JSON and CSV."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("route", type=Path)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.route.read_text())
    records = payload["scores"]
    counts = Counter()
    byte_counts = defaultdict(int)
    rows = []
    for name, record in sorted(records.items()):
        chosen = record["chosen"]
        size = int(record["bytes"])
        counts[chosen] += 1
        byte_counts[chosen] += size
        rows.append({"name": name, **record})
    summary = {
        "calibration_steps": payload.get("calibration_steps"),
        "tensor_counts": dict(counts),
        "selected_bytes": dict(byte_counts),
        "total_tensors": sum(counts.values()),
        "total_bytes": sum(byte_counts.values()),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["name", "shape", "dtype", "bytes", "row", "column",
                      "block", "chosen", "runner_up_gap"]
        with args.csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
