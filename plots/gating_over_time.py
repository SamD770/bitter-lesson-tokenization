"""
This script looks at :
1. How frequently the model gates each index of the input sequence
2. The next token cross entropy 
"""

from clean_code.flexible_bitter_llm import FlexibleBitterLLM, LinearGater, text_to_tensor, per_token_losses_backbone
from accelerate import Accelerator
from training_random_base_model.hparam_utils import get_model_kwargs
from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader
from data_processing import split_fineweb

import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

@torch.no_grad()
def main():
    byte_tokenizer = AutoTokenizer.from_pretrained("evabyte/EvaByte", trust_remote_code=True)

    learned_checkpoint_path = "training_random_base_model/checkpoints/130M_2025.09.26_18.06"
    learned_model_kwargs = get_model_kwargs("130M")
    learned_model_kwargs["vocab_size"] = len(byte_tokenizer) 
    learned_model_kwargs["GaterClass"] = LinearGater

    device = torch.device("cuda")

    learned_model = FlexibleBitterLLM(**learned_model_kwargs).to(device, dtype=torch.bfloat16)

    accelerator = Accelerator()
    learned_model = accelerator.prepare(learned_model)
    accelerator.load_state(learned_checkpoint_path)

    # random_checkpoint_path = "training_random_base_model/checkpoints/..."
    # random_model_kwargs = get_model_kwargs("130M")
    # random_model_kwargs["vocab_size"] = len(byte_tokenizer) 

    # TODO once random model is trained: configure the model as an input to this function
    model = learned_model

    _, _, test_set = split_fineweb.get_splits()

    test_dataloader = DataLoader(
        test_set,
        batch_size=32
    )

    max_seq_length = 4096

    next_token_cross_entropies = []
    early_next_token_cross_entropies = []
    all_gate_probs = []

    total_batches = 100 # TODO: add a 0 

    for i, batch in tqdm(enumerate(test_dataloader), total=total_batches, desc="Processing batches"):
        
        batch, loss_mask = text_to_tensor(
            batch, 
            byte_tokenizer, 
            max_seq_length, 
            device
        )

        out_model = model(batch)
        real_gate_probs = out_model["down_gate_probs"]

        all_gate_probs.append(real_gate_probs)

        per_token_losses = per_token_losses_backbone(
            batch, 
            loss_mask, 
            out_model, 
            real_gate_probs, 
            learn_gating=True,
            early_exit_advantage_estimate=True
        )
        # TODO: use gpt2 tokenizer to see the equivalent "gate probs"

        next_token_cross_entropies.append(per_token_losses["next_token_cross_entropy"])
        early_next_token_cross_entropies.append(per_token_losses["early_next_token_cross_entropy"])

        if i > total_batches:
            break

    next_token_cross_entropies = torch.cat(next_token_cross_entropies, dim=0) # stack on the batch dimension
    early_next_token_cross_entropies = torch.cat(early_next_token_cross_entropies, dim=0)
    all_gate_probs = torch.cat(all_gate_probs, dim=0)

    mean_next_token_cross_entropy = next_token_cross_entropies.mean(dim=0)
    mean_early_next_token_cross_entropy = early_next_token_cross_entropies.mean(dim=0)
    mean_gate_probs = all_gate_probs.to(torch.float32).mean(dim=0) # Upcast to float32 for plotting granularity etc.

    # Discard the first and last gate for plotting as they are fixed to 1.
    # Verify that it is not a bug that the second gate is also quite high.
    print(f"{mean_gate_probs[:50]=}")  
    print(f"{mean_gate_probs[-50:]=}")
    mean_gate_probs = mean_gate_probs[1:-1] 

    mean_next_token_cross_entropy = mean_next_token_cross_entropy.cpu().numpy()
    mean_early_next_token_cross_entropy = mean_early_next_token_cross_entropy.cpu().numpy()
    mean_gate_probs = mean_gate_probs.cpu().numpy()

    # Apply smoothing using moving average
    def smooth(data, window_size=10):
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    
    smoothed_next_token = smooth(mean_next_token_cross_entropy)
    smoothed_early_next_token = smooth(mean_early_next_token_cross_entropy)

    # Plot the next token cross entropy over time.
    
    plt.figure(figsize=(10, 5))
    # Plot original values with low alpha
    plt.plot(mean_next_token_cross_entropy, label='Next Token Cross Entropy', alpha=0.2)
    plt.plot(mean_early_next_token_cross_entropy, label='Early Next Token Cross Entropy', alpha=0.2)
    
    # Plot smoothed values with full alpha
    plt.plot(range(len(mean_next_token_cross_entropy) - len(smoothed_next_token), len(mean_next_token_cross_entropy)), 
             smoothed_next_token, label='Next Token Cross Entropy (Smoothed)', alpha=1.0)
    plt.plot(range(len(mean_early_next_token_cross_entropy) - len(smoothed_early_next_token), len(mean_early_next_token_cross_entropy)), 
             smoothed_early_next_token, label='Early Next Token Cross Entropy (Smoothed)', alpha=1.0)
    
    plt.legend()
    plt.savefig("plots/renders/next_token_cross_entropy_over_time.png")

    # Plot the gate probs over time.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))
    
    # Panel 1: Full limits
    ax1.plot(mean_gate_probs, label='Gate Probs', alpha=0.2)
    smoothed_gate_probs = smooth(mean_gate_probs)
    ax1.plot(range(len(mean_gate_probs) - len(smoothed_gate_probs), len(mean_gate_probs)), 
             smoothed_gate_probs, label='Gate Probs (Smoothed)', alpha=1.0)
    ax1.legend()
    ax1.set_title('Gate Probs Over Time (Full Range)')
    
    # Panel 2: Limited y-axis between 0.23 and 0.35
    ax2.plot(mean_gate_probs, label='Gate Probs', alpha=0.2)
    ax2.plot(range(len(mean_gate_probs) - len(smoothed_gate_probs), len(mean_gate_probs)), 
             smoothed_gate_probs, label='Gate Probs (Smoothed)', alpha=1.0)
    ax2.set_ylim(0.23, 0.35)
    ax2.legend()
    ax2.set_title('Gate Probs Over Time (0.23-0.35)')

    plt.tight_layout()
    plt.savefig("plots/renders/gate_probs_over_time.png")


if __name__ == "__main__":
    main()