"""
Compute bits-per-byte of the target continuation for each sample in an
lm-eval results JSON file (as produced by eval/run.py).

Usage:
    python3 eval/compute_bpb.py <results_file.json> [--task arc_easy]

The output is written to <results_file>_bpb.txt in the same directory.
"""

import argparse
import json
import math
import os
from typing import Optional


def compute_bpb(results_path: str, task: Optional[str] = None) -> None:
    with open(results_path) as f:
        data = json.load(f)

    samples_by_task = data["samples"]

    if task is None:
        if len(samples_by_task) == 1:
            task = next(iter(samples_by_task))
        else:
            raise ValueError(
                f"Multiple tasks found: {list(samples_by_task.keys())}. "
                "Specify one with --task."
            )

    bpb_values = []
    for sample in samples_by_task[task]:
        target = sample["target"]
        if isinstance(target, int):
            # Multiple-choice task: target is the index of the correct choice
            continuation = sample["arguments"][target][1]
            log_likelihood_nats = sample["resps"][target][0][0]
        else:
            # Single-completion task (e.g. lambada): one entry in arguments
            continuation = sample["arguments"][0][1]
            log_likelihood_nats = sample["resps"][0][0][0]

        num_bytes = len(continuation.encode("utf-8"))
        if num_bytes == 0:
            continue

        bpb_values.append(-log_likelihood_nats / (num_bytes * math.log(2)))

    n = len(bpb_values)
    mean = sum(bpb_values) / n
    variance = sum((x - mean) ** 2 for x in bpb_values) / (n - 1)
    stderr = math.sqrt(variance / n)

    out_path = os.path.splitext(results_path)[0] + "_bpb.txt"
    with open(out_path, "w") as f:
        f.write(f"task: {task}\n")
        f.write(f"num_samples: {n}\n")
        f.write(f"mean_bpb: {mean:.6f}\n")
        f.write(f"stderr: {stderr:.6f}\n")

    print(f"task={task}  n={n}  mean_bpb={mean:.6f} ± {stderr:.6f}")
    print(f"Written to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute mean BPB from lm-eval results JSON")
    parser.add_argument("results_file", help="Path to *_eval_results.json")
    parser.add_argument("--task", default=None, help="Task name (inferred if only one task)")
    args = parser.parse_args()

    compute_bpb(args.results_file, args.task)