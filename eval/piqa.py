"""
PIQA evaluation for byte-level language models.

PIQA (Physical Interaction Question Answering) is a multiple-choice benchmark
that tests physical commonsense reasoning.

Usage:
    from eval import evaluate_piqa
    from model.model import AutoregressiveUnet
    from transformers import AutoTokenizer
    
    model = AutoregressiveUnet(...)  # Load your model
    tokenizer = AutoTokenizer.from_pretrained("evabyte/EvaByte", trust_remote_code=True)
    
    results = evaluate_piqa(model, tokenizer)
    print(f"PIQA Accuracy: {results['results']['piqa']['acc']:.2%}")
"""

import torch
import json
import os
from typing import Optional, Dict, Any

import lm_eval
from lm_eval import evaluator

from training_random_base_model.model_from_checkpoint import load_model

from .wrapper import ByteLevelLMWrapper


def evaluate_piqa(
    model: torch.nn.Module,
    byte_tokenizer,
    batch_size: int = 1,
    device: Optional[str] = None,
    num_fewshot: int = 0,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Evaluate a byte-level model on the PIQA benchmark.
    
    Args:
        model: A byte-level language model (e.g., AutoregressiveUnet)
        byte_tokenizer: A byte-level tokenizer (e.g., EvaByte tokenizer)
        batch_size: Batch size for evaluation (default: 1)
        device: Device to run evaluation on (defaults to model's device)
        num_fewshot: Number of few-shot examples (default: 0 for zero-shot)
        limit: Limit number of examples to evaluate (None for full dataset)
        
    Returns:
        Dictionary containing evaluation results with structure:
        {
            "results": {
                "piqa": {
                    "acc": float,  # Accuracy
                    "acc_stderr": float,  # Standard error
                    "acc_norm": float,  # Normalized accuracy
                    "acc_norm_stderr": float,
                }
            },
            "config": {...},  # Evaluation configuration
            "versions": {...},  # Task versions
        }
    """
    # Ensure model is in eval mode
    model.eval()
    
    # Create the wrapper
    wrapper = ByteLevelLMWrapper(
        model=model,
        tokenizer=byte_tokenizer,
        batch_size=batch_size,
        device=device,
    )
    
    # Run evaluation
    results = evaluator.simple_evaluate(
        model=wrapper,
        tasks=["piqa"],
        num_fewshot=num_fewshot,
        limit=limit,
        batch_size=batch_size,
    )
    
    return results


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
    
    args = parser.parse_args()
    
    # Load tokenizer
    byte_tokenizer = AutoTokenizer.from_pretrained("evabyte/EvaByte", trust_remote_code=True)
    
    # Load model (adjust import based on your model location)
    model = load_model(args.checkpoint)
    model = model.to("cuda", dtype=torch.bfloat16)
    
    # Run evaluation
    results = evaluate_piqa(
        model=model,
        byte_tokenizer=byte_tokenizer,
        batch_size=args.batch_size,
        limit=args.limit,
        num_fewshot=args.num_fewshot,
    )
    
    # Print results
    print("\n" + "=" * 50)
    print("PIQA Evaluation Results")
    print("=" * 50)
    
    results_file = os.path.join(args.checkpoint, "piqa_results.json")
    piqa_results = results["results"]["piqa"]

    print(piqa_results)
    with open(results_file, "w") as f:
        json.dump(piqa_results, f)
