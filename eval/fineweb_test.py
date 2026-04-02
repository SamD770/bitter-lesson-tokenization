"""
FineWeb bits-per-byte (BPB) evaluation for byte-level language models.

Computes the average bits-per-UTF8-byte on the FineWeb test set.

Usage:
    from eval.fineweb_test import evaluate_fineweb_bpb
    from model.model import AutoregressiveUnet
    from transformers import AutoTokenizer
    
    model = AutoregressiveUnet(...)  # Load your model
    tokenizer = AutoTokenizer.from_pretrained("evabyte/EvaByte", trust_remote_code=True)
    
    results = evaluate_fineweb_bpb(model, tokenizer)
    print(f"FineWeb BPB: {results['results']['fineweb_bpb']['bpb']:.4f}")
"""

import math
import torch
import json
import os
import numpy as np
from typing import Optional, Dict, Any
from tqdm import tqdm

from data_processing import split_fineweb
from model.model import text_to_tensor
from training_random_base_model.model_from_checkpoint import load_model


def evaluate_fineweb_bpb(
    model: torch.nn.Module,
    tokenizer,
    batch_size: int = 1,
    max_seq_length: int = 4096,
    device: Optional[str] = None,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Evaluate a byte-level model on the FineWeb test set using bits-per-byte.
    
    Args:
        model: A byte-level language model
        tokenizer: A byte-level tokenizer (e.g., EvaByte tokenizer)
        batch_size: Batch size for evaluation (default: 1)
        max_seq_length: Maximum sequence length (default: 4096)
        device: Device to run evaluation on (defaults to model's device)
        limit: Limit number of documents to evaluate (None for full dataset)
        
    Returns:
        Dictionary containing evaluation results:
        {
            "results": {
                "fineweb_bpb": {
                    "bpb": float,  # Average bits-per-byte
                    "bpb_stderr": float,  # Standard error
                }
            },
            "num_documents": int,
            "total_bytes": int,
        }
    """
    model.eval()
    
    # Determine device
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    # Load FineWeb test set
    _, _, test_set = split_fineweb.get_splits()
    
    if limit is not None:
        test_set = test_set.select(range(min(limit, len(test_set))))
    
    # Collect per-document BPB values for computing mean and stderr
    bpb_values = []
    total_bytes = 0
    total_log_likelihood = 0.0
    
    # Process in batches
    num_docs = len(test_set)
    
    for i in tqdm(range(0, num_docs, batch_size), desc="FineWeb BPB"):
        batch_end = min(i + batch_size, num_docs)
        batch_indices = range(i, batch_end)
        
        # Get batch texts
        batch_texts = [test_set[idx]["text"] for idx in batch_indices]
        
        # Truncate texts to max_seq_length bytes
        truncated_texts = []
        byte_counts = []
        for text in batch_texts:
            text_bytes = text.encode('utf-8')
            if len(text_bytes) > max_seq_length:
                # Truncate to max_seq_length bytes, handling UTF-8 properly
                text_bytes = text_bytes[:max_seq_length]
                # Decode back, ignoring incomplete multi-byte chars at the end
                text = text_bytes.decode('utf-8', errors='ignore')
            truncated_texts.append(text)
            byte_counts.append(len(text.encode('utf-8')))
        
        # Create batch dict for text_to_tensor
        batch_dict = {"text": truncated_texts}
        
        # Tokenize and prepare tensors
        input_ids, loss_mask = text_to_tensor(batch_dict, tokenizer, max_seq_length, device)
        
        # Get model log-probabilities
        with torch.no_grad():
            from bpe_tokenizer.bpe_tokenizer import BPEAutoRegressiveUnet
            if isinstance(model, BPEAutoRegressiveUnet):
                out = model(input_ids, texts=truncated_texts)
            else:
                out = model(input_ids)
            logits = out["logits"]  # [batch, seq_len, vocab_size]
        
        # Compute log-likelihood for each document in the batch
        for batch_idx in range(len(truncated_texts)):
            doc_input_ids = input_ids[batch_idx]
            doc_logits = logits[batch_idx]
            doc_loss_mask = loss_mask[batch_idx]
            doc_byte_count = byte_counts[batch_idx]
            
            # Compute log-likelihood: sum of log P(token_i | token_0:i-1)
            # For position i, logits[i] predicts token at position i+1
            log_likelihood = 0.0
            seq_len = doc_loss_mask.sum().item()  # Actual sequence length (excluding padding)
            
            for pos in range(int(seq_len) - 1):
                next_token = doc_input_ids[pos + 1].item()
                token_logprob = logits[batch_idx, pos, next_token].item()
                log_likelihood += token_logprob
            
            # Convert to bits-per-byte
            # BPB = -log_likelihood_nats / (ln(2) * num_bytes)
            if doc_byte_count > 0:
                bpb = -log_likelihood / (math.log(2) * doc_byte_count)
                bpb_values.append(bpb)
                total_bytes += doc_byte_count
                total_log_likelihood += log_likelihood
    
    # Compute aggregate statistics
    bpb_array = np.array(bpb_values)
    mean_bpb = float(np.mean(bpb_array))
    stderr_bpb = float(np.std(bpb_array) / np.sqrt(len(bpb_array)))
    
    # Also compute corpus-level BPB (weighted by document length)
    corpus_bpb = -total_log_likelihood / (math.log(2) * total_bytes)
    
    return {
        "results": {
            "fineweb_bpb": {
                "bpb": mean_bpb,
                "bpb_stderr": stderr_bpb,
                "corpus_bpb": corpus_bpb,
            }
        },
        "num_documents": len(bpb_values),
        "total_bytes": total_bytes,
    }


if __name__ == "__main__":
    import argparse
    from transformers import AutoTokenizer
    
    parser = argparse.ArgumentParser(description="Evaluate byte-level model on FineWeb (bits-per-byte)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--max-seq-length", type=int, default=4096, help="Maximum sequence length")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of documents")
    
    args = parser.parse_args()
    
    # Load tokenizer
    byte_tokenizer = AutoTokenizer.from_pretrained("evabyte/EvaByte", trust_remote_code=True)
    
    # Load model
    model = load_model(args.checkpoint)
    model = model.to("cuda", dtype=torch.bfloat16)
    
    # Run evaluation
    results = evaluate_fineweb_bpb(
        model=model,
        tokenizer=byte_tokenizer,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        limit=args.limit,
    )
    
    # Print results
    print("\n" + "=" * 50)
    print("FineWeb BPB Evaluation Results")
    print("=" * 50)
    
    bpb_results = results["results"]["fineweb_bpb"]
    print(f"Mean BPB: {bpb_results['bpb']:.4f} ± {bpb_results['bpb_stderr']:.4f}")
    print(f"Corpus BPB: {bpb_results['corpus_bpb']:.4f}")
    print(f"Documents evaluated: {results['num_documents']}")
    print(f"Total bytes: {results['total_bytes']:,}")
    
    # Save results
    results_file = os.path.join(args.checkpoint, "fineweb_bpb_results.json")
    print(f"\nSaving results to: {results_file}")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
