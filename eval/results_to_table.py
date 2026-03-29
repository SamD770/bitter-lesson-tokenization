import pandas as pd
import json

checkpoints = [
    ("147M_random___2025.11.19_22.45", "Uniform"),
    ("147M_nawrot___2025.11.19_15.28", "Dynamic (Nawrot et al.)"),
    ("147M_hnet_hnet___2026.01.27_04.16", "H-Net (Hwang et al.)"),
    ("147M_sequential_sequential___2026.01.26_15.56", "Ours"),
]

tasks = [
    ("piqa", "acc,none"),
    ("hellaswag", "acc,none"),
    ("arc_easy", "acc,none"),
    ("lambada_openai", "acc,none"),
    ("fineweb_bpb", "bpb")
]

task_names = [task[0] for task in tasks]

table = pd.DataFrame(columns=["method"] + task_names)

for checkpoint, method in checkpoints:
    for task, results_key in tasks:
        try:
            results_file = f"training_random_base_model/checkpoints/{checkpoint}/{task}_eval_results.json"
            with open(results_file, "r") as f:
                results = json.load(f)
            print(method, task)
            val = results["results"][task][results_key]
        except:
            print(f"Error loading results for {task} from {results_file}")
            val = 100.
        table.loc[method, task] = val

table.drop(columns=["method"], inplace=True)

# Rename columns for nicer LaTeX headers
column_renames = {
    "piqa": "PIQA",
    "hellaswag": "HellaSwag", 
    "arc_easy": "ARC-Easy",
    "lambada_openai": "LAMBADA",
    "fineweb_bpb": "FineWeb Test"
}
table = table.rename(columns=column_renames)

# Convert to LaTeX
latex_table = table.to_latex(
    index=True,
    float_format="%.3f",
    caption="Evaluation results across different tokenization methods.",
    label="tab:eval_results",
    column_format="l" + "c" * len(task_names),
    escape=False
)

print(latex_table)