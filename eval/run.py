
import torch
import json
import os
from typing import Optional, Dict, Any

import lm_eval
from lm_eval import evaluator

from training_random_base_model.model_from_checkpoint import load_model

from .wrapper import ByteLevelLMWrapper

def evaluate_multiple(
    model: torch.nn.Module,
    byte_tokenizer,
    tasks: list[str],
    batch_size: int = 1,
    device: Optional[str] = None,
    num_fewshot: int = 0,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Evaluate a byte-level model on multiple benchmarks.
    
    Args:
        model: A byte-level language model
        byte_tokenizer: A byte-level tokenizer
        tasks: List of task names (e.g., ["piqa", "hellaswag", "arc_easy"])
        batch_size: Batch size for evaluation
        device: Device to run evaluation on
        num_fewshot: Number of few-shot examples
        limit: Limit number of examples per task
        
    Returns:
        Dictionary containing evaluation results for all tasks
    """
    model.eval()
    
    wrapper = ByteLevelLMWrapper(
        model=model,
        tokenizer=byte_tokenizer,
        batch_size=batch_size,
        device=device,
    )
    
    results = evaluator.simple_evaluate(
        model=wrapper,
        tasks=tasks,
        num_fewshot=num_fewshot,
        limit=limit,
        batch_size=batch_size,
    )
    
    return results


if __name__ == "__main__":
    # Example usage / quick test
    import argparse
    from transformers import AutoTokenizer
    
    parser = argparse.ArgumentParser(description="Evaluate byte-level model on PIQA")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples")
    parser.add_argument("--num-fewshot", type=int, default=0, help="Number of few-shot examples")
    parser.add_argument("--tasks", nargs="+", default=["arc_easy", "piqa", "hellaswag", "lambada_openai"], help="Tasks to evaluate")
    
    args = parser.parse_args()
    
    # Load tokenizer
    byte_tokenizer = AutoTokenizer.from_pretrained("evabyte/EvaByte", trust_remote_code=True)
    
    # Load model (adjust import based on your model location)
    model = load_model(args.checkpoint)
    model = model.to("cuda", dtype=torch.bfloat16)
    

    for task in args.tasks:
        results = evaluate_multiple(
            model=model,
            byte_tokenizer=byte_tokenizer,
            tasks=[task],
            batch_size=args.batch_size,
            limit=args.limit,
            num_fewshot=args.num_fewshot,
        )

    
        # Print results
        print("\n" + "=" * 50)
        print("Finished evaluating task: ", task)
        print("=" * 50)
        
        results_file = os.path.join(args.checkpoint, f"{task}_eval_results.json")

        print(json.dumps(results, indent=2))
        with open(results_file, "w") as f:
            json.dump(results, f)
