import json
import statistics

base = "/raid0/fastwam/experiments/pi05_fp8_granularity_precision_a800_v1"


def load(path):
    with open(path) as handle:
        return {int(item["step"]): item for item in map(json.loads, handle)}


reference = load(f"{base}/dense/metrics.jsonl")
print("reference", min(reference), max(reference), reference[max(reference)]["action_loss"])
for mode in ("gradient_row", "gradient_column", "gradient_block"):
    metrics = load(f"{base}/{mode}/metrics.jsonl")
    steps = sorted(set(reference) & set(metrics))
    diffs = [
        abs(metrics[step]["action_loss"] - reference[step]["action_loss"])
        for step in steps
    ]
    rel = [
        diff / max(abs(reference[step]["action_loss"]), 1e-12)
        for diff, step in zip(diffs, steps)
    ]
    tail = [metrics[step] for step in steps if step >= 481]
    stats = [
        json.loads(line)
        for line in open(f"{base}/{mode}/fp8_stats.jsonl")
    ]
    print(
        mode,
        "steps", steps[0], steps[-1],
        "final", metrics[steps[-1]]["action_loss"],
        "mae", statistics.mean(diffs),
        "mape_pct", statistics.mean(rel) * 100,
        "max", max(diffs),
        "tail20_mean", statistics.mean(item["action_loss"] for item in tail),
        "mean_step_time_302+", statistics.mean(metrics[step]["step_time"] for step in steps if step >= 302),
        "mean_sps_302+", statistics.mean(metrics[step]["samples_per_sec"] for step in steps if step >= 302),
        "selected_tensors_sum", sum(item["selected_tensors"] for item in stats),
        "selected_bytes_sum", sum(item["selected_bytes"] for item in stats),
        "stats_records", len(stats),
    )
