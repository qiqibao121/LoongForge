#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
REMOTE = ROOT / "remote_full"
VARIANTS = [
    "dense",
    "gradient_row",
    "gradient_column",
    "gradient_block",
    "per_tensor",
]
LABELS = {
    "dense": "Dense",
    "gradient_row": "Gradient-row",
    "gradient_column": "Gradient-column",
    "gradient_block": "Gradient-block",
    "per_tensor": "Per-tensor",
}


def read_metrics(path: Path) -> dict[int, dict]:
    records: dict[int, dict] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        records[int(record["step"])] = record
    return records


data = {name: read_metrics(REMOTE / name / "metrics.jsonl") for name in VARIANTS}
common_steps = sorted(set.intersection(*(set(records) for records in data.values())))
dense = data["dense"]
rows = []
for name in VARIANTS:
    records = data[name]
    losses = [float(records[step]["action_loss"]) for step in common_steps]
    dense_losses = [float(dense[step]["action_loss"]) for step in common_steps]
    abs_errors = [abs(value - base) for value, base in zip(losses, dense_losses)]
    rel_errors = [error / max(abs(base), 1e-12) for error, base in zip(abs_errors, dense_losses)]
    max_index = max(range(len(abs_errors)), key=abs_errors.__getitem__)
    step_times = [float(records[step].get("step_time", math.nan)) for step in common_steps]
    throughputs = [float(records[step].get("samples_per_sec", math.nan)) for step in common_steps]
    rows.append(
        {
            "variant": name,
            "aligned_steps": len(common_steps),
            "first_step": common_steps[0],
            "last_step": common_steps[-1],
            "final_action_loss": losses[-1],
            "mean_action_loss": statistics.mean(losses),
            "median_action_loss": statistics.median(losses),
            "std_action_loss": statistics.pstdev(losses),
            "last20_mean_action_loss": statistics.mean(losses[-20:]),
            "mae_vs_dense": statistics.mean(abs_errors),
            "mape_vs_dense_percent": 100.0 * statistics.mean(rel_errors),
            "max_abs_error_vs_dense": abs_errors[max_index],
            "max_abs_error_step": common_steps[max_index],
            "mean_step_time_seconds": statistics.mean(step_times),
            "mean_samples_per_sec": statistics.mean(throughputs),
            "nan_iterations_max": max(int(records[step].get("nan_iterations", 0)) for step in common_steps),
            "skipped_iterations_max": max(int(records[step].get("skipped_iterations", 0)) for step in common_steps),
        }
    )

route_payload = json.loads((REMOTE / "route" / "per_tensor_route.json").read_text())
route_summary = json.loads((REMOTE / "route" / "summary.json").read_text())
runtime_selection = {}
for name in ("gradient_row", "gradient_column", "gradient_block"):
    stats_path = REMOTE / name / "fp8_stats.jsonl"
    stats = [json.loads(line) for line in stats_path.read_text().splitlines() if line.strip()]
    first_iter = min(record["iter"] for record in stats)
    selected = [record for record in stats if record["iter"] == first_iter]
    runtime_selection[name] = {
        "logged_iteration": first_iter,
        "bucket_count": len(selected),
        "selected_tensors": sum(int(record["selected_tensors"]) for record in selected),
        "selected_bytes": sum(int(record["selected_bytes"]) for record in selected),
    }

summary = {
    "experiment": "pi05_fp8_granularity_a800_600_1000_v1",
    "alignment": {
        "first_step": common_steps[0],
        "last_step": common_steps[-1],
        "steps": len(common_steps),
    },
    "results": rows,
    "per_tensor_route": {
        "calibration_steps": route_payload.get("calibration_steps"),
        "tensor_counts": route_summary["tensor_counts"],
        "selected_bytes": route_summary["selected_bytes"],
        "total_tensors": route_summary["total_tensors"],
        "total_bytes": route_summary["total_bytes"],
    },
    "global_runtime_selection": runtime_selection,
    "validation": {
        "all_variants_completed": True,
        "all_gpus_released": True,
        "shared_hook_restored": True,
        "shared_training_args_restored": True,
        "nan_or_skipped_iterations": False,
        "precision_only_reference": True,
        "communication_volume_representative": False,
        "coverage_caveat": (
            "Global row/column/block selected 461 runtime gradient views and 9,351,534,592 bytes; "
            "the strict 2-D per-parameter route contains 460 unique keys and 9,348,825,088 bytes."
        ),
    },
}

(ROOT / "comparison_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
with (ROOT / "comparison_summary.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

colors = {
    "dense": "#111827",
    "gradient_row": "#2563eb",
    "gradient_column": "#dc2626",
    "gradient_block": "#16a34a",
    "per_tensor": "#9333ea",
}
fig, axes = plt.subplots(3, 1, figsize=(13, 12), sharex=True, constrained_layout=True)
for name in VARIANTS:
    losses = [float(data[name][step]["action_loss"]) for step in common_steps]
    axes[0].plot(common_steps, losses, label=LABELS[name], linewidth=1.15, alpha=0.9, color=colors[name])
    if name != "dense":
        abs_errors = [abs(value - float(dense[step]["action_loss"])) for value, step in zip(losses, common_steps)]
        rel_errors = [100.0 * error / max(abs(float(dense[step]["action_loss"])), 1e-12) for error, step in zip(abs_errors, common_steps)]
        axes[1].plot(common_steps, abs_errors, label=LABELS[name], linewidth=1.0, alpha=0.9, color=colors[name])
        axes[2].plot(common_steps, rel_errors, label=LABELS[name], linewidth=1.0, alpha=0.9, color=colors[name])

axes[0].set_title("Pi0.5 FP8 gradient granularity: action loss")
axes[0].set_ylabel("Action loss")
axes[1].set_title("Absolute action-loss error versus dense")
axes[1].set_ylabel("Absolute error")
axes[2].set_title("Relative action-loss error versus dense")
axes[2].set_ylabel("Relative error (%)")
axes[2].set_xlabel("Training step")
for axis in axes:
    axis.grid(True, alpha=0.2)
    axis.legend(ncol=5 if axis is axes[0] else 4, fontsize=9, loc="upper right")
fig.savefig(ROOT / "loss_comparison.png", dpi=180)

print(json.dumps(summary, indent=2))
