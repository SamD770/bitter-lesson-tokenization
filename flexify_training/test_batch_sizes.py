"""
Test the batch sizes for the training step. Example usage:
python -m training_random_base_model.test_batch_sizes --model_size 32M
"""
from clean_code.flexible_bitter_llm import FlexibleBitterLLM, off_policy_flexible_training_step
from training_random_base_model.run import get_optimizer_scheduler
from training_random_base_model.hparam_utils import get_model_kwargs, model_sizes

from accelerate import Accelerator

import torch

import argparse

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", default="73M", choices=model_sizes, help="Model size (e.g., 73M, 130M)")
    args = parser.parse_args()
    model_size = args.model_size

    device = "cuda"
    sequence_length = 4096
    vocab_size = 320
    downsample_rate = 1.0

    accelerator = Accelerator()

    model_kwargs = get_model_kwargs(model_size)
    model = FlexibleBitterLLM(**model_kwargs).to(device, dtype=torch.bfloat16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    model, optimizer = accelerator.prepare(model, optimizer)

    batch_sizes = [2**i for i in range(10)]

    for batch_size in batch_sizes:
        print(f"Batch testing batch size: {batch_size}")
        my_batch = torch.randint(0, vocab_size, (batch_size, sequence_length)).to(device)
        loss_mask = torch.ones_like(my_batch).to(torch.bfloat16)

        try:
            loss_dict = off_policy_flexible_training_step(model, optimizer, my_batch, loss_mask, use_off_policy=False, accelerator=accelerator, downsample_rate=downsample_rate)
            print(f"Achieved downsample rate: {loss_dict['true_downsample_rate']}")
        except Exception as e:
            print(f"Error with batch size {batch_size}: {e}")
            exit()

if __name__ == "__main__":
    main()