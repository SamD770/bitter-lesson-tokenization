from training_random_base_model.hparam_utils import get_model_kwargs, get_optimization_kwargs, model_sizes

from clean_code.bitter_llm import parameter_count_string
from clean_code.flexible_bitter_llm import FlexibleBitterLLM, flexible_training_loop_warm_start_accelerate, SelectTokenDownsampler, ExactRandomGater

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
    parser.add_argument("--model_size", type=str, default="32M", choices=model_sizes, help="Model size to train")
    args = parser.parse_args()


    username = "sdauncey"
    scratch_dir = f"/scratch/{username}/tokenizer_training"
    logging_dir = os.path.join(scratch_dir, "wandb_logs")

    model_kwargs = get_model_kwargs(args.model_size)

    # After testing, change these:
    # optimization_kwargs = get_optimization_kwargs(args.model_size)
    optimization_kwargs = {
        "learning_rate": 3e-3,
        "effective_batch_size": 64,
        "warmup_bytes": 1.1e6,
        "training_bytes": 1.5e7,
    }

    training_loop_kwargs = {
        "num_epochs": 1, 
        "max_seq_length": 4096,
        "batch_print_every": 1, 
        "warm_start_steps": None,
        "learn_gating": False,
        "early_output_loss_weight": 0.2,
    }

    optimization_kwargs["effective_batch_size"] = 64
    bytes_per_batch = optimization_kwargs["effective_batch_size"] * training_loop_kwargs["max_seq_length"]
    optimization_kwargs["effective_step_count"] = optimization_kwargs["training_bytes"] / bytes_per_batch
    optimization_kwargs["effective_warmup_steps"] = optimization_kwargs["warmup_bytes"] / bytes_per_batch


    # "effective" means the result if we just ran on a single process.
    # effective_batch_size = batch_size * num_processes * gradient_accumulation_steps
    # effective_step_count = step_count / (num_processes * gradient_accumulation_steps)

    batch_size = 16
    gradient_accumulation_steps = optimization_kwargs["effective_batch_size"] / (batch_size * accelerator.num_processes)
    
    total_steps = optimization_kwargs["effective_step_count"] * accelerator.num_processes * gradient_accumulation_steps
    warmup_steps = optimization_kwargs["effective_warmup_steps"] * accelerator.num_processes * gradient_accumulation_steps


    optimization_kwargs.update({
        "batch_size": 16,
        "warmup_steps": warmup_steps,
        "total_steps": total_steps, # 200 batches processed for 2 GPUs
        "gradient_accumulation_steps": 2
    })

    training_loop_kwargs.update({
        "batch_limit": total_steps,
    })



    config = {**vars(args), **training_loop_kwargs, **optimization_kwargs, **model_kwargs}
    # Convert class objects to string representations for wandb config serialization
    config_for_wandb = config.copy()
    config_for_wandb["GaterClass"] = config["GaterClass"].__name__
    config_for_wandb["DownSamplerClass"] = config["DownSamplerClass"].__name__


    accelerator =  Accelerator(log_with="wandb", gradient_accumulation_steps=gradient_accumulation_steps)

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
        print("Downloading OpenWebText dataset...")

    # Load a small portion of OpenWebText (25% of the dataset)
    openwebtext_25p = datasets.load_dataset(
        "openwebtext",
        split="train[:25%]",  # Using only 25% samples of the dataset for now.
        cache_dir=os.path.join(scratch_dir, "openwebtext_25p_cache"),
        trust_remote_code=True
    )

    if accelerator.is_main_process:
        print(f"Downloaded {len(openwebtext_25p)} examples from OpenWebText")

    set_seed(args.seed)
    if accelerator.is_main_process:
        print(f"Using random seed: {args.seed}")
    
    
    byte5_tokenizer = AutoTokenizer.from_pretrained("google/byt5-large")

    model = FlexibleBitterLLM(**model_kwargs).to(device, dtype=torch.bfloat16)

    if accelerator.is_main_process:
        print(f"model has {parameter_count_string(model)} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=optimization_kwargs["learning_rate"])

    # Create a linear warmup followed by cosine annealing scheduler
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, 
        start_factor=0.1, 
        total_iters=optimization_kwargs["warmup_steps"] * accelerator.num_processes
    )

    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=optimization_kwargs["total_steps"] * accelerator.num_processes - optimization_kwargs["warmup_steps"] * accelerator.num_processes
    )

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, 
        schedulers=[warmup_scheduler, cosine_scheduler], 
        milestones=[optimization_kwargs["warmup_steps"] * accelerator.num_processes]
    )

    train_dataloader = DataLoader(
        openwebtext_25p,
        batch_size=optimization_kwargs["batch_size"],
        num_workers=32,
        pin_memory=True
    )

    model, optimizer, scheduler, train_dataloader = accelerator.prepare(
        model, optimizer, scheduler, train_dataloader
    )

    flexible_training_loop_warm_start_accelerate(
        model, 
        optimizer, 
        scheduler, 
        train_dataloader, 
        accelerator, 
        tokenizer=byte5_tokenizer,
        **training_loop_kwargs
    )

    accelerator.end_training()

    net_scratch_dir = os.path.join("/itet-stor/sdauncey/net_scratch/VScodeProjects/bitter-lesson-tokenization")
    checkpoint_dir = os.path.join(net_scratch_dir, "training_random_base_model", args.model_size)

    if accelerator.is_main_process: 
        print(f"Saving checkpoint to {checkpoint_dir}")

    accelerator.save_state(checkpoint_dir)

if __name__ == "__main__":
    main()