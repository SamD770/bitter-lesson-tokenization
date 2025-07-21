"""
Train a larger random base model with 138M parameters.
"""
from clean_code.bitter_llm import parameter_count_string
from clean_code.flexible_bitter_llm import FlexibleBitterLLM, flexible_training_loop_warm_start_accelerate, SelectTokenDownsampler, ExactRandomGater

import os

import argparse

import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer
import datasets
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed

import wandb

# Hyperparameters following:
# Tomer Porian∗ Mitchell Wortsman† Jenia Jitsev‡ Ludwig Schmidt† Yair Carmon∗
# Resolving Discrepancies in Compute-Optimal Scaling of Language Models
# https://arxiv.org/pdf/2406.19146

def main():

    username = "sdauncey"
    scratch_dir = f"/scratch/{username}/tokenizer_training"
    logging_dir = os.path.join(scratch_dir, "wandb_logs")

    project_config = ProjectConfiguration(
        project_dir=".",
        logging_dir=logging_dir,
        # project_name="training_random_base_model",
        # project_description="Training a random base model for tokenization",
        # project_tags=["tokenization", "language-modeling", "transformer-models"],
    )

    accelerator =  Accelerator(log_with="wandb") # log_with="wandb", project_config=project_config)

    if accelerator.is_main_process:
        if not os.path.exists(scratch_dir):
            os.makedirs(scratch_dir)

    device = "cuda"

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Random base model training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

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
    
    training_loop_kwargs = {
        "num_epochs": 1, 
        "max_seq_length": 4096,
        "batch_print_every": 10**2, 
        "batch_limit": 10**4,
        "warm_start_steps": None,
        "learn_gating": False,
        "early_output_loss_weight": 0.2,
    }

    # According to Yair Carmon, we should use a learning rate of 5e-3 and a warmup of 800M bytes for a ~100M parameter model. 
    optimization_kwargs = {
        "learning_rate": 5e-3,
        "warmup_steps": 8 * 10**8 / (32 * 4096),
        "batch_size": 32,
    }

    # Quite similar to GPT-2-small. 
    model_kwargs = {
        "vocab_size": 256,
        "embedding_dim": 768,
        "num_heads": 12,
        "downsample_rate": 0.25,
        "sliding_window": 64,
        "n_down_layers": 3,
        "n_mid_layers": 12,
        "n_up_layers": 3,
        "GaterClass": ExactRandomGater,
        "DownSamplerClass": SelectTokenDownsampler,
    }

    config = {**vars(args), **training_loop_kwargs, **optimization_kwargs, **model_kwargs}
    config_for_wandb = config.copy()
    config_for_wandb["GaterClass"] = config["GaterClass"].__name__
    config_for_wandb["DownSamplerClass"] = config["DownSamplerClass"].__name__
    
    accelerator.init_trackers("training_random_base_model", config=config_for_wandb)

    byte5_tokenizer = AutoTokenizer.from_pretrained("google/byt5-large")

    model = FlexibleBitterLLM(**model_kwargs).to(device, dtype=torch.bfloat16)

    if accelerator.is_main_process:
        print(f"model has {parameter_count_string(model)} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=optimization_kwargs["learning_rate"])

    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=optimization_kwargs["warmup_steps"])

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
    checkpoint_dir = os.path.join(net_scratch_dir, "training_random_base_model", "checkpoints")

    if accelerator.is_main_process: 
        print(f"Saving checkpoint to {checkpoint_dir}")

    accelerator.save_state(checkpoint_dir)

if __name__ == "__main__":
    main()