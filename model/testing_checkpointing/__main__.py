from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import AutoTokenizer

import datasets
import os
import json

import torch
from torch.utils.data import DataLoader

from training_random_base_model.hparam_utils import get_model_kwargs
from clean_code.flexible_bitter_llm import FlexibleBitterLLM, flexible_training_loop_warm_start_accelerate
from clean_code.flexible_bitter_llm import save_checkpoint, load_checkpoint, BatchLimitCondition

username = os.environ.get("USER", "user")
scratch_dir = os.environ.get("SCRATCH_DIR", f"/tmp/{username}/tokenizer_training")
output_dir = "clean_code/testing_checkpointing/"

# Run with two accelerators.
SEED = 42

def init_accelerator(batch_size, gradient_accumulation_steps):

    accelerator = Accelerator(log_with="wandb", gradient_accumulation_steps=gradient_accumulation_steps)

    set_seed(SEED)

    model = FlexibleBitterLLM(**get_model_kwargs("73M")).to(dtype=torch.bfloat16)

    # Load a small portion of OpenWebText (25% of the dataset)
    openwebtext_25p = datasets.load_dataset(
        "openwebtext",
        split="train[:25%]",  # Using only 25% samples of the dataset for now.
        cache_dir=os.path.join(scratch_dir, "openwebtext_25p_cache"),
        trust_remote_code=True
    )

    train_dataloader = DataLoader(
        openwebtext_25p,
        batch_size=batch_size,
        num_workers=8,
        pin_memory=True
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=20
    )

    model, optimizer, scheduler, train_dataloader = accelerator.prepare(
        model, optimizer, scheduler, train_dataloader
    )

    elapsed_vals = {
        "all_bytes_elapsed": 0,
        "non_padding_bytes_elapsed": 0,
        "flops_elapsed": 0,
        "effective_batches_elapsed": 0,
    }

    return accelerator, model, optimizer, scheduler, train_dataloader, elapsed_vals


def train_checkpoint(run_name, n_effective_batches):

    tokenizer = AutoTokenizer.from_pretrained("google/byt5-large")

    effective_batch_size = 96
    batch_size = 16
    gradient_accumulation_steps = 3

    accelerator, model, optimizer, scheduler, train_dataloader, elapsed_vals = \
        init_accelerator(batch_size, gradient_accumulation_steps)

    config = {"run_name": run_name, "n_effective_batches": n_effective_batches}

    accelerator.init_trackers(
        "testing_checkpointing", 
        config=config, 
        init_kwargs={
            "wandb": {
                "config": config,
                "entity": os.environ.get("WANDB_ENTITY"),
        }},
    )
    
    run_dir = os.path.join(output_dir, run_name)

    if os.path.exists(run_dir):
        if os.listdir(run_dir):
            train_dataloader, elapsed_vals = load_checkpoint(run_dir, accelerator, train_dataloader, effective_batch_size, batch_size)
        else:
            if accelerator.is_main_process: 
                print(f"Run directory {run_dir} exists but is empty. Starting new run.")
    else:
        if accelerator.is_main_process:
            print(f"Run directory {run_dir} does not exist. Creating it and starting new run.")
            os.makedirs(run_dir)

    elapsed_vals = flexible_training_loop_warm_start_accelerate(
        model, 
        optimizer, 
        scheduler, 
        train_dataloader, 
        accelerator, 
        tokenizer,
        num_epochs=1, 
        warm_start_steps=None, 
        max_seq_length=4096, 
        step_print_every=1,
        stop_condition=BatchLimitCondition(n_effective_batches),
        **elapsed_vals
    )

    accelerator.end_training()

    save_checkpoint(run_dir, accelerator, elapsed_vals)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--n_effective_batches", type=int, required=True)
    args = parser.parse_args()

    train_checkpoint(args.run_name, args.n_effective_batches)