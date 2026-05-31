# Train three models:
# 1. Downsample rate 1/8
# 2. Downsample rate 1/4
# 3. Variable downsample rate
# Make two checkpoints for all models:
# 1. Checkpoint 1: after 3.4e9 bytes (equal amount of data)
# 2. Checkpoint 2: after equal amount of Flops (1.5e12)

from training_random_base_model.hparam_utils import get_model_kwargs, get_optimization_kwargs, model_sizes, training_loop_hparam_defaults
from data_processing import split_fineweb

from clean_code.utils import parameter_count_string
from clean_code.flexible_bitter_llm import (
    FlexibleBitterLLM, 
    flexible_training_loop_warm_start_accelerate, 
    ExactRandomGater,
    LinearGater,
    BytesLimitCondition, 
    save_checkpoint,
    DownsampleRateEmbedding,
)

from clean_code.downsample_rate_scheduler import RandomChoiceDownsampleRateScheduler

from clean_code.nawrot_plugin import NawrotDownsampler, NawrotUpsampler, NawrotGater

from clean_code.conditional_sequential import OptimizedSequentialyDependentLinearGater, ScaledSequentialyDependentLinearGater

import os

import argparse

import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer
import datasets
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

    stop_condition = BytesLimitCondition(optimization_kwargs["training_bytes"])

    delta_optimization_kwargs = {
        "batch_size": batch_size,
        "lr_warmup_updates": lr_warmup_updates,
        "lr_total_updates": lr_total_updates,
        "gradient_accumulation_steps": gradient_accumulation_steps
    }

    delta_training_loop_kwargs = {
        "stop_condition": stop_condition,
    }

    return delta_optimization_kwargs, delta_training_loop_kwargs


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
        pin_memory=True,
        shuffle=True,
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


def to_wandb_config(config):
    # Convert class objects to string representations for wandb config serialization
    config_for_wandb = config.copy()
    config_for_wandb["GaterClass"] = config["GaterClass"].__name__
    config_for_wandb["DownSamplerClass"] = config["DownSamplerClass"].__name__
    # TODO: We can add a stop condition config
    config_for_wandb["stop_condition"] = config["stop_condition"].__class__.__name__
    config_for_wandb["bytes_limit"] = config["stop_condition"].bytes_limit

    return config_for_wandb


def add_nawrot_model_kwargs(model_kwargs):
    model_kwargs["GaterClass"] = NawrotGater
    model_kwargs["DownSamplerClass"] = NawrotDownsampler
    model_kwargs["UpsamplerClass"] = NawrotUpsampler
    return model_kwargs


def add_nawrot_training_loop_kwargs(training_loop_kwargs):
    training_loop_kwargs["learn_gating"] = True # For the consistency loss
    training_loop_kwargs["relative_gating_loss_weight"] = 0.0 # So no policy gradient is used.
    return training_loop_kwargs


def add_linear_model_kwargs(model_kwargs):
    model_kwargs["GaterClass"] = LinearGater
    return model_kwargs


def add_linear_training_loop_kwargs(training_loop_kwargs):
    training_loop_kwargs["learn_gating"] = True
    training_loop_kwargs["discount_rate"] = 0.99
    training_loop_kwargs["early_exit_advantage_estimate"] = True
    return training_loop_kwargs

def add_sequential_dependent_linear_model_kwargs(model_kwargs):
    model_kwargs["GaterClass"] = ScaledSequentialyDependentLinearGater
    model_kwargs["gater_kwargs"] = {"scale_factor": 1/16., "filter_size": 8}
    return model_kwargs

def add_sequential_dependent_linear_training_loop_kwargs(training_loop_kwargs):
    training_loop_kwargs["learn_gating"] = True
    training_loop_kwargs["discount_rate"] = 0.99
    training_loop_kwargs["early_exit_advantage_estimate"] = True
    training_loop_kwargs["relative_gating_loss_weight"] = 0.1
    training_loop_kwargs["consistency_loss_weight"] = 0.1
    return training_loop_kwargs


def add_flexi_model_kwargs(model_kwargs):
    model_kwargs["DownsampleRateEmbeddingClass"] = DownsampleRateEmbedding
    return model_kwargs


def add_flexi_training_loop_kwargs(training_loop_kwargs):
    # geometric series of 1 + 0.9 + 0.9^2 + ... + 0.9^40 \approx 10, so this will give an expected downsample rate of roughly 10/40 \approx 0.25.
    # Old schedule: list(0.9**i for i in range(1, 35))

    # Uniform betweeen 0.1 and 0.3
    downsample_rates = [0.1 + 0.02 * i for i in range(10)]
    training_loop_kwargs["downsample_rate_schedule"] = RandomChoiceDownsampleRateScheduler(downsample_rates, 0.25)
    
    # training_loop_kwargs["consistency_loss_weight"] = 1.
    return training_loop_kwargs


def main():

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Random base model training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--model_size", type=str, default="18M", choices=model_sizes, help="Model size to train")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size used on each GPU")
    args = parser.parse_args()


    username = os.environ.get("USER", "user")
    scratch_dir = os.environ.get("SCRATCH_DIR", f"/tmp/{username}/tokenizer_training")
    # scratch_dir = "/workspace"
    logging_dir = os.path.join(scratch_dir, "wandb_logs")

    byte_tokenizer = AutoTokenizer.from_pretrained("evabyte/EvaByte", trust_remote_code=True)

    model_kwargs = get_model_kwargs(args.model_size)
    model_kwargs["vocab_size"] = len(byte_tokenizer) # Keep for ExactRandomGater
    model_kwargs["flash_attn"] = True
    add_sequential_dependent_linear_model_kwargs(model_kwargs)
    add_flexi_model_kwargs(model_kwargs)

    accelerator =  Accelerator(log_with="wandb")

    optimization_kwargs = get_optimization_kwargs(args.model_size)
    optimization_kwargs["effective_batch_size"] = optimization_kwargs["effective_batch_size"] // 4 # Reduce this so we get more different downsample rates.

    training_loop_kwargs = training_loop_hparam_defaults
    training_loop_kwargs["early_output_loss_weight"] = 0.2
    training_loop_kwargs["step_print_every"] = 1
    add_sequential_dependent_linear_training_loop_kwargs(training_loop_kwargs)
    add_flexi_training_loop_kwargs(training_loop_kwargs)

    delta_optimization_kwargs, delta_training_loop_kwargs = \
        effective_to_device_steps(optimization_kwargs, training_loop_kwargs, accelerator, args.batch_size)

    optimization_kwargs.update(delta_optimization_kwargs)
    training_loop_kwargs.update(delta_training_loop_kwargs)

    config = {**vars(args), **training_loop_kwargs, **optimization_kwargs, **model_kwargs}

    if accelerator.is_main_process: 
        for k, v in config.items():
            print(f"{k:<40}: {v}")
    
    accelerator.gradient_accumulation_steps = delta_optimization_kwargs["gradient_accumulation_steps"]
    
    time_string = datetime.now().strftime('%Y.%m.%d_%H.%M')
    run_id = f"{args.model_size}_{time_string}"

    # For some reason, you need to pass the config to the init_kwargs when using wandb with accelerate in offline mode. https://github.com/huggingface/accelerate/issues/3607
    accelerator.init_trackers(
        "training_random_base_model", 
        config=to_wandb_config(config), 
        init_kwargs={
            "wandb": {
                "config": to_wandb_config(config),
                "entity": os.environ.get("WANDB_ENTITY"),
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

    model = FlexibleBitterLLM(**model_kwargs).to(device, dtype=torch.bfloat16)

    if accelerator.is_main_process:
        print(f"model has {parameter_count_string(model)} parameters")

    optimizer, scheduler = get_optimizer_scheduler(optimization_kwargs, model)

    model, optimizer, scheduler, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, scheduler, train_dataloader, val_dataloader
    )

    elapsed_vals = flexible_training_loop_warm_start_accelerate(
        model, 
        optimizer, 
        scheduler, 
        train_dataloader, 
        val_dataloader,
        accelerator, 
        tokenizer=byte_tokenizer,
        **training_loop_kwargs
    )

    accelerator.end_training()

    net_scratch_dir = os.environ.get("PROJECT_DIR", os.getcwd())
    checkpoint_dir = os.path.join(net_scratch_dir, "flexify_training", "checkpoints", run_id)

    if accelerator.is_main_process: 
        print(f"Saving checkpoint to {checkpoint_dir}")

    save_checkpoint(checkpoint_dir, accelerator, elapsed_vals)

if __name__ == "__main__":
    main()