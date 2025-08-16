from training_random_base_model.hparam_utils import get_model_kwargs, get_optimization_kwargs, model_sizes
from data_processing import split_fineweb

from clean_code.utils import parameter_count_string
from clean_code.flexible_bitter_llm import FlexibleBitterLLM, flexible_training_loop_warm_start_accelerate, SelectTokenDownsampler, ExactRandomGater, BatchLimitCondition, save_checkpoint

import os

import argparse

import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer
import datasets
from accelerate import Accelerator
from accelerate.utils import set_seed


def main():
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Random base model training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--model_size", type=str, default="18M", choices=model_sizes, help="Model size to train")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size used on each GPU")
    args = parser.parse_args()


    username = "sdauncey"
    scratch_dir = f"/scratch/{username}/tokenizer_training"
    # scratch_dir = "/workspace"
    logging_dir = os.path.join(scratch_dir, "wandb_logs")

    byte_tokenizer = AutoTokenizer.from_pretrained("evabyte/EvaByte", trust_remote_code=True)

    model_kwargs = get_model_kwargs(args.model_size)
    model_kwargs["vocab_size"] = len(byte_tokenizer) # we use len rather than .vocab_size to include the special tokens.

    accelerator =  Accelerator(log_with="wandb")

    # After testing, change these:
    optimization_kwargs = get_optimization_kwargs(args.model_size)
    # optimization_kwargs = {
    #     "learning_rate": 3e-3,
    #     "effective_batch_size": 128,
    #     "warmup_bytes": 1.1e7,
    #     "training_bytes": 1.5e8,
    # }

    training_loop_kwargs = {
        "num_epochs": 1, 
        "max_seq_length": 4096,
        "step_print_every": 100, 
        "warm_start_steps": None,
        "learn_gating": False,
        "early_output_loss_weight": 0.2,
    }

    all_bytes_per_effective_batch = optimization_kwargs["effective_batch_size"] * training_loop_kwargs["max_seq_length"]
    optimization_kwargs["effective_step_count"] = optimization_kwargs["training_bytes"] / all_bytes_per_effective_batch
    optimization_kwargs["effective_warmup_steps"] = optimization_kwargs["warmup_bytes"] / all_bytes_per_effective_batch


    # "effective" means the result if we just ran on a single process.

    batch_size = args.batch_size
    assert optimization_kwargs["effective_batch_size"] % (batch_size * accelerator.num_processes) == 0, "effective_batch_size must be divisible by batch_size * num_processes"
    gradient_accumulation_steps = optimization_kwargs["effective_batch_size"] // (batch_size * accelerator.num_processes)
    # batch_limit is used to exit the training loop (which is agnostic to the number of processes)
    batch_limit = optimization_kwargs["effective_step_count"] * gradient_accumulation_steps
    # total_steps is used for the learning rate scheduler (which is agnostic to gradient accumulation)
    total_steps = optimization_kwargs["effective_step_count"] * accelerator.num_processes
    warmup_steps = optimization_kwargs["effective_warmup_steps"] * accelerator.num_processes


    optimization_kwargs.update({
        "batch_size": batch_size,
        "warmup_steps": warmup_steps,
        "total_steps": total_steps, # 200 batches processed for 2 GPUs
        "gradient_accumulation_steps": gradient_accumulation_steps
    })

    stop_condition = BatchLimitCondition(batch_limit)
        
    training_loop_kwargs.update({
        "stop_condition": stop_condition,
        "validate_every": 100,
    })

    config = {**vars(args), **training_loop_kwargs, **optimization_kwargs, **model_kwargs}
    # Convert class objects to string representations for wandb config serialization
    config_for_wandb = config.copy()
    config_for_wandb["GaterClass"] = config["GaterClass"].__name__
    config_for_wandb["DownSamplerClass"] = config["DownSamplerClass"].__name__
    # TODO: We can add a stop condition config
    config_for_wandb["stop_condition"] = config["stop_condition"].__class__.__name__
    config_for_wandb["batch_limit"] = batch_limit

    accelerator.gradient_accumulation_steps = gradient_accumulation_steps

    if accelerator.is_main_process: 
        for k, v in config.items():
            print(f"{k:<40}: {v}")

    # For some reason, you need to pass the config to the init_kwargs when using wandb with accelerate in offline mode. https://github.com/huggingface/accelerate/issues/3607
    accelerator.init_trackers(
        "training_random_base_model", 
        config=config_for_wandb, 
        init_kwargs={
            "wandb": {
                "config": config_for_wandb,
                "entity": "samdauncey-eth-z-rich"
        }},
    )


    if accelerator.is_main_process:
        if not os.path.exists(scratch_dir):
            os.makedirs(scratch_dir)

    device = "cuda"

    # Download a portion of OpenWebText dataset
    # This will download a subset of the OpenWebText corpus
    if accelerator.is_main_process:
        print("Getting Fineweb splits...")

    train_set, val_set, test_set = split_fineweb.get_splits()

    if accelerator.is_main_process:
        print(f"Got: {len(train_set)} examples from Fineweb")

    set_seed(args.seed)
    if accelerator.is_main_process:
        print(f"Using random seed: {args.seed}")
    
    model = FlexibleBitterLLM(**model_kwargs).to(device, dtype=torch.bfloat16)

    if accelerator.is_main_process:
        print(f"model has {parameter_count_string(model)} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=optimization_kwargs["learning_rate"])

    # Create a linear warmup followed by cosine annealing scheduler
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, 
        start_factor=0.1, 
        total_iters=optimization_kwargs["warmup_steps"]
    )

    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=optimization_kwargs["total_steps"] - optimization_kwargs["warmup_steps"]
    )

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, 
        schedulers=[warmup_scheduler, cosine_scheduler], 
        milestones=[optimization_kwargs["warmup_steps"]]
    )

    train_dataloader = DataLoader(
        train_set,
        batch_size=optimization_kwargs["batch_size"],
        num_workers=4,
        pin_memory=True
    )

    # Ensure that we never try to load more than the val dataset in one batch
    val_batch_size = min(len(val_set) // accelerator.num_processes, optimization_kwargs["batch_size"])
    assert len(val_set) % (val_batch_size * accelerator.num_processes) == 0, "val set needs to be evenly divisible by the number of processes."

    val_dataloader = DataLoader(
        val_set,
        batch_size=val_batch_size, 
        num_workers=4,
        pin_memory=True
    )

    model, optimizer, scheduler = accelerator.prepare(
        model, optimizer, scheduler #, train_dataloader, val_dataloader
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

    net_scratch_dir = os.path.join("/itet-stor/sdauncey/net_scratch/VScodeProjects/bitter-lesson-tokenization")
    checkpoint_dir = os.path.join(net_scratch_dir, "training_random_base_model", args.model_size)

    if accelerator.is_main_process: 
        print(f"Saving checkpoint to {checkpoint_dir}")

    save_checkpoint(checkpoint_dir, accelerator, elapsed_vals)

if __name__ == "__main__":
    main()