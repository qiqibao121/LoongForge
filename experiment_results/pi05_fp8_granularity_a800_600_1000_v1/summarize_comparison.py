#!/usr/bin/env python3
from __future__ import annotations
import csv, json, math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sources = {
    "dense": ROOT.parent / "fp8_granularity_long_a800_v1" / "dense_metrics.jsonl",
    "gradient_row": ROOT.parent / "fp8_granularity_long_a800_v1" / "row_metrics.jsonl",
    "gradient_block": ROOT.parent / "fp8_granularity_long_a800_v1" / "block_metrics.jsonl",
    "per_tensor": ROOT / "remote_output" / "per_tensor" / "metrics.jsonl",
}

def read(path: Path):
    return {int((r := json.loads(line))["step"]): r for line in path.read_text().splitlines() if line.strip()}

data = {name: read(path) for name, path in sources.items()}
steps = sorted(set.intersection(*(set(v) for v in data.values())))
rows = []
for name, records in data.items():
    paired = [(s, data["dense"][s]["action_loss"], records[s]["action_loss"]) for s in steps]
    errs = [abs(x - y) for _, x, y in paired]
    mape = [e / max(abs(x), 1e-12) for _, x, y in paired for e in [abs(x - y)]]
    tail = [records[s]["action_loss"] for s in steps[-20:]]
    rows.append({
        "variant": name,
        "steps": len(steps),
        "first_step": steps[0],
        "last_step": steps[-1],
        "final_action_loss": records[steps[-1]]["action_loss"],
        "mae_vs_dense": sum(errs) / len(errs),
        "mape_vs_dense_percent": 100 * sum(mape) / len(mape),
        "max_abs_error_vs_dense": max(errs),
        "max_abs_error_step": steps[errs.index(max(errs))],
        "last20_mean_action_loss": sum(tail) / len(tail),
        "mean_step_time": sum(records[s].get("step_time", math.nan) for s in steps) / len(steps),
        "mean_samples_per_sec": sum(records[s].get("samples_per_sec", math.nan) for s in steps) / len(steps),
        "nan_iterations": sum(records[s].get("nan_iterations", 0) for s in steps),
        "skipped_iterations": sum(records[s].get("skipped_iterations", 0) for s in steps),
    })

out = {"alignment": {"first_step": steps[0], "last_step": steps[-1], "steps": len(steps)}, "results": rows}
(ROOT / "comparison_summary.json").write_text(json.dumps(out, indent=2) + "\n")
with (ROOT / "comparison_summary.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
print(json.dumps(out, indent=2))
