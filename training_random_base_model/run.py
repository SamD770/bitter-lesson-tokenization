from training_random_base_model.config_loader import (
    load_config, 
    config_to_wandb, 
    get_available_sizes,
    get_available_architectures,
    get_run_types,
    get_model_name,
    save_model_config,
)
from data_processing import split_fineweb

from model.utils import parameter_count_string
from model.model import (
    AutoregressiveUnet, 
    training_loop,
    BytesLimitCondition,
    save_checkpoint,
    load_checkpoint,
)

import os
import argparse

import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer
from accelerate import Accelerator
from accelerate.utils import set_seed

from datetime import datetime

# Explainer:
# An **effective batch** is a parameter update on all GPUs. 
# It is the logical batch size that would be used if we just ran on a single process with sufficient GPU memory.
# But, we don't have enough GPU memory so need to use multiple processes and gradient accumulation. 

# A **step** is a single forward/backward pass across all GPUs.
# A **parameter update** is a single parameter update on a single GPU.
# A **batch** is a single forward/backward pass a single GPU.

# Example: 10 effective batches on 4 processes with 3 gradient accumulation steps would be:
# 3 * 10 = 30 steps
# 4 * 10 = 40 parameter updates
# 3 * 4 * 10 = 120 batches

# accelerate complicates this because it calls scheduler.step() after each parameter update.


def effective_to_device_steps(optimization_kwargs, training_loop_kwargs, accelerator, batch_size):
    # "effective" means the result if we just ran on a single process with sufficient GPU memory.

    bytes_per_effective_batch = optimization_kwargs["effective_batch_size"] * training_loop_kwargs["max_seq_length"]
    lr_total_effective_batches = optimization_kwargs["training_bytes"] / bytes_per_effective_batch
    lr_warmup_effective_batches = optimization_kwargs["warmup_bytes"] / bytes_per_effective_batch

    assert optimization_kwargs["effective_batch_size"] % (batch_size * accelerator.num_processes) == 0, "effective_batch_size must be divisible by batch_size * num_processes"
    gradient_accumulation_steps = optimization_kwargs["effective_batch_size"] // (batch_size * accelerator.num_processes)
    # batch_limit is used to exit the training loop (which is agnostic to the number of processes)
    batch_limit = lr_total_effective_batches * gradient_accumulation_steps
    # total_steps is used for the learning rate scheduler (which is agnostic to gradient accumulation)
    lr_total_updates = lr_total_effective_batches * accelerator.num_processes
    lr_warmup_updates = lr_warmup_effective_batches * accelerator.num_processes

    # stop_condition = BytesLimitCondition(optimization_kwargs["training_bytes"])
    max_training_bytes = optimization_kwargs["training_bytes"]
    checkpoint_conditions = [BytesLimitCondition(max_training_bytes * i / 10) for i in range(1, 11)]

    delta_optimization_kwargs = {
        "batch_size": batch_size,
        "lr_warmup_updates": lr_warmup_updates,
        "lr_total_updates": lr_total_updates,
        "gradient_accumulation_steps": gradient_accumulation_steps
    }

    return delta_optimization_kwargs, checkpoint_conditions


def get_optimizer_scheduler(optimization_kwargs, model):
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=optimization_kwargs["learning_rate"])

    # Create a linear warmup followed by cosine annealing scheduler
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, 
        start_factor=0.1, 
        total_iters=optimization_kwargs["lr_warmup_updates"]
    )

    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=optimization_kwargs["lr_total_updates"] - optimization_kwargs["lr_warmup_updates"]
    )

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, 
        schedulers=[warmup_scheduler, cosine_scheduler], 
        milestones=[optimization_kwargs["lr_warmup_updates"]]
    )

    return optimizer, scheduler


def get_dataloaders(batch_size, log_status, num_processes):

    # Download a portion of OpenWebText dataset
    # This will download a subset of the OpenWebText corpus
    if log_status:
        print("Getting Fineweb splits...")

    train_set, val_set, test_set = split_fineweb.get_splits()

    if log_status:
        print(f"Got: {len(train_set)} examples from Fineweb")

    train_dataloader = DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=8,
        pin_memory=True
    )

    # Ensure that we never try to load more than the val dataset in one batch
    val_batch_size = min(len(val_set) // num_processes, batch_size)
    assert len(val_set) % (val_batch_size * num_processes) == 0, "val set needs to be evenly divisible by the number of processes."

    val_dataloader = DataLoader(
        val_set,
        batch_size=val_batch_size, 
        num_workers=8,
        pin_memory=True
    )

    return train_dataloader, val_dataloader


def main():

    # Get available choices from config files
    sizes = get_available_sizes()
    architectures = get_available_architectures()
    run_types = get_run_types()

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Random base model training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--size", type=str, default="18M", choices=sizes, help="Model size to train")
    parser.add_argument("--architecture", type=str, default="random", choices=architectures, help="Model architecture (gater/downsampler/upsampler)")
    parser.add_argument("--run_type", type=str, default="default", choices=run_types, help="Training run type (training loop configuration)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size used on each GPU")
    parser.add_argument("--resume_checkpoint", type=str, default=None, help="Checkpoint to resume training from")
    parser.add_argument("--aspect_ratio", type=int, default=None, help="Aspect ratio to train at")
    args = parser.parse_args()

    # Validate aspect ratio
    if args.aspect_ratio:
        assert args.aspect_ratio in [1, 2, 3, 4, 5, 6, 7, 8], "Aspect ratio must be one of 1, 2, 3, 4, 5, 6, 7, 8"

    username = "sdauncey"
    scratch_dir = f"/scratch/{username}/tokenizer_training"
    logging_dir = os.path.join(scratch_dir, "wandb_logs")

    byte_tokenizer = AutoTokenizer.from_pretrained("evabyte/EvaByte", trust_remote_code=True)

    # Load configuration from JSON files
    model_kwargs, training_loop_kwargs, optimization_kwargs = load_config(
        size=args.size,
        architecture=args.architecture,
        run_type=args.run_type,
        aspect_ratio=args.aspect_ratio
    )
    
    # Override vocab_size with tokenizer length
    model_kwargs["vocab_size"] = len(byte_tokenizer)

    # Get model name for logging
    model_name = get_model_name(args.size, args.architecture)

    # Initialize accelerator
    accelerator = Accelerator(log_with="wandb")

    # Compute device-specific optimization parameters
    delta_optimization_kwargs, checkpoint_conditions = \
        effective_to_device_steps(optimization_kwargs, training_loop_kwargs, accelerator, args.batch_size)

    optimization_kwargs.update(delta_optimization_kwargs)

    # Build combined config for logging
    config = {
        **vars(args), 
        **training_loop_kwargs, 
        **optimization_kwargs, 
        **model_kwargs, 
        "model_name": model_name,
        "stop_condition": checkpoint_conditions[-1]
    }

    if accelerator.is_main_process: 
        for k, v in config.items():
            print(f"{k:<40}: {v}")
    
    accelerator.gradient_accumulation_steps = delta_optimization_kwargs["gradient_accumulation_steps"]
    
    time_string = datetime.now().strftime('%Y.%m.%d_%H.%M')

    if args.seed == 42:
        seed_string = "_"
    else:
        seed_string = f"{args.seed}_"

    if args.aspect_ratio:
        aspect_ratio_string = f"ar{args.aspect_ratio}_"
    else:
        aspect_ratio_string = "_"
    
    run_id = f"{model_name}_{args.run_type}_{aspect_ratio_string}{seed_string}{time_string}"

    if accelerator.is_main_process:
        print(f"Run ID: {run_id}")

    # Convert config for wandb
    wandb_config = config_to_wandb(model_kwargs, training_loop_kwargs, optimization_kwargs, checkpoint_conditions[-1])
    wandb_config.update(vars(args))
    wandb_config["model_name"] = model_name

    # For some reason, you need to pass the config to the init_kwargs when using wandb with accelerate in offline mode. https://github.com/huggingface/accelerate/issues/3607
    accelerator.init_trackers(
        "training_random_base_model", 
        config=wandb_config, 
        init_kwargs={
            "wandb": {
                "config": wandb_config,
                "entity": "samdauncey-eth-z-rich",
                "id": run_id
        }},
    )

    if accelerator.is_main_process:
        if not os.path.exists(scratch_dir):
            os.makedirs(scratch_dir)

    device = accelerator.device

    set_seed(args.seed)
    if accelerator.is_main_process:
        print(f"Using random seed: {args.seed}")

    train_dataloader, val_dataloader = get_dataloaders(optimization_kwargs["batch_size"], accelerator.is_main_process, accelerator.num_processes)

    model = AutoregressiveUnet(**model_kwargs).to(device, dtype=torch.bfloat16)

    if accelerator.is_main_process:
        print(f"model has {parameter_count_string(model)} parameters")

    optimizer, scheduler = get_optimizer_scheduler(optimization_kwargs, model)

    model, optimizer, scheduler, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, scheduler, train_dataloader, val_dataloader
    )

    if args.resume_checkpoint:
        train_dataloader, elapsed_vals = load_checkpoint(
            args.resume_checkpoint, accelerator, train_dataloader, optimization_kwargs["effective_batch_size"], args.batch_size
        )
    
    elapsed_vals = {}

    intermediate_checkpoint_dir = os.path.join(scratch_dir, "training_random_base_model", "checkpoints", run_id)

    for checkpoint_condition in checkpoint_conditions:

        elapsed_vals = training_loop(
            model, 
            optimizer, 
            scheduler, 
            train_dataloader, 
            val_dataloader,
            accelerator, 
            tokenizer=byte_tokenizer,
            stop_condition=checkpoint_condition,
            **elapsed_vals,
            **training_loop_kwargs
        )
        
        if accelerator.is_main_process:
            print(f"Saving intermediate checkpoint to {intermediate_checkpoint_dir}")
            save_checkpoint(intermediate_checkpoint_dir, accelerator, elapsed_vals)
            # Save model config alongside checkpoint for easy model recreation
            save_model_config(model_kwargs, os.path.join(intermediate_checkpoint_dir, "model_config.json"))

    accelerator.end_training()

    net_scratch_dir = os.path.join("/itet-stor/sdauncey/net_scratch/VScodeProjects/bitter-lesson-tokenization")
    final_checkpoint_dir = os.path.join(net_scratch_dir, "training_random_base_model", "checkpoints", run_id)

    if accelerator.is_main_process: 
        print(f"Saving checkpoint to {final_checkpoint_dir}")

    save_checkpoint(final_checkpoint_dir, accelerator, elapsed_vals)
    # Save model config alongside final checkpoint
    if accelerator.is_main_process:
        save_model_config(model_kwargs, os.path.join(final_checkpoint_dir, "model_config.json"))

if __name__ == "__main__":
    main()
