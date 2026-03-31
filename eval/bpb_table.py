"""
Compute BPB ± stderr for a set of checkpoints and benchmarks, then print
a formatted table.  Values within one standard error of the best (lowest)
BPB in each column are shown in bold.

Usage:
    python3 eval/bpb_table.py
"""

import json
import math
import os
from typing import Optional, Tuple

# ── configuration ──────────────────────────────────────────────────────────────

CHECKPOINTS = [
    "training_random_base_model/checkpoints/147M_hnet_hnet___2026.01.27_04.16",
    "training_random_base_model/checkpoints/147M_nawrot___2025.11.19_15.28",
    "training_random_base_model/checkpoints/147M_random___2025.11.19_22.45",
    "training_random_base_model/checkpoints/147M_sequential_sequential___2026.01.26_15.56",
]

MODEL_NAMES = {
    "147M_hnet_hnet___2026.01.27_04.16":             "H-Net (Hwang et al.)",
    "147M_nawrot___2025.11.19_15.28":                "Dynamic (Nawrot et al.)",
    "147M_random___2025.11.19_22.45":                "Uniform",
    "147M_sequential_sequential___2026.01.26_15.56": "Ours",
}

TASK_DISPLAY = {
    "arc_easy":       "ARC-Easy",
    "hellaswag":      "HellaSwag",
    "lambada_openai": "LAMBADA",
    "piqa":           "PIQA",
}

TASKS = ["arc_easy", "hellaswag", "lambada_openai", "piqa"]

# ── BPB computation ─────────────────────────────────────────────────────────────

def compute_bpb_stats(results_path: str, task: Optional[str] = None) -> Tuple[float, float, int]:
    """Return (mean_bpb, stderr, n) for the target continuation of each sample."""
    with open(results_path) as f:
        data = json.load(f)

    samples_by_task = data["samples"]
    if task is None:
        task = next(iter(samples_by_task))

    bpb_values = []
    for sample in samples_by_task[task]:
        target = sample["target"]
        if isinstance(target, int):
            continuation = sample["arguments"][target][1]
            log_likelihood_nats = sample["resps"][target][0][0]
        else:
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
    return mean, stderr, n


# ── collect results ─────────────────────────────────────────────────────────────

results = {}   # results[model_name][task] = (mean, stderr, n)

for ckpt in CHECKPOINTS:
    model_name = MODEL_NAMES[os.path.basename(ckpt)]
    results[model_name] = {}
    for task in TASKS:
        path = os.path.join(ckpt, f"{task}_eval_results.json")
        if not os.path.exists(path):
            results[model_name][task] = None
            continue
        mean, stderr, n = compute_bpb_stats(path, task)
        results[model_name][task] = (mean, stderr, n)

# ── determine which cells to bold ───────────────────────────────────────────────
# Bold a cell if its mean is within one SE of the best mean in that column,
# i.e. mean_i <= best_mean + SE_best.

bold = {}  # bold[model][task] = True/False
for task in TASKS:
    task_results = {m: results[m][task] for m in results if results[m][task] is not None}
    best_mean, best_se, _ = min(task_results.values(), key=lambda x: x[0])
    for model, stats in task_results.items():
        bold.setdefault(model, {})[task] = stats[0] <= best_mean + best_se

# ── formatting ──────────────────────────────────────────────────────────────────

def fmt_md(stats, is_bold: bool) -> str:
    if stats is None:
        return "—"
    mean, stderr, _ = stats
    s = f"{mean:.3f} ±{stderr:.3f}"
    return f"**{s}**" if is_bold else s


def fmt_latex(stats, is_bold: bool) -> str:
    if stats is None:
        return r"\text{—}"
    mean, stderr, _ = stats
    s = f"{mean:.3f} $\\pm$ {stderr:.3f}"
    return f"\\textbf{{{s}}}" if is_bold else s


def build_markdown(model_order) -> str:
    headers = [TASK_DISPLAY[t] for t in TASKS]
    rows = []
    rows.append("| Model | " + " | ".join(headers) + " |")
    rows.append("|---|" + "|".join("---" for _ in TASKS) + "|")
    for model in model_order:
        cells = [model]
        for task in TASKS:
            cells.append(fmt_md(results[model][task], bold.get(model, {}).get(task, False)))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def build_latex(model_order) -> str:
    headers = [TASK_DISPLAY[t] for t in TASKS]
    col_fmt = "l" + "c" * len(TASKS)
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"  \centering")
    lines.append(f"  \\begin{{tabular}}{{{col_fmt}}}")
    lines.append(r"    \toprule")
    lines.append("    Model & " + " & ".join(f"\\textbf{{{h}}}" for h in headers) + r" \\")
    lines.append(r"    \midrule")
    for model in model_order:
        cells = [model]
        for task in TASKS:
            cells.append(fmt_latex(results[model][task], bold.get(model, {}).get(task, False)))
        lines.append("    " + " & ".join(cells) + r" \\")
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"  \caption{Bits-per-byte on target continuations (lower is better). "
                 r"Bold values are within one standard error of the best result per benchmark.}")
    lines.append(r"  \label{tab:bpb_results}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ── print and save ───────────────────────────────────────────────────────────────

model_order = [MODEL_NAMES[os.path.basename(c)] for c in CHECKPOINTS]

md = build_markdown(model_order)
latex = build_latex(model_order)

print(md)
print()
print(latex)
print()

out_path = "training_random_base_model/checkpoints/bpb_table.txt"
with open(out_path, "w") as f:
    f.write(md + "\n\n")
    f.write(latex + "\n")

print(f"Written to {out_path}")
