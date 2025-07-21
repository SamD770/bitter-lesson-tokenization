from clean_code.bitter_llm import parameter_count_string
from clean_code.flexible_bitter_llm import FlexibleBitterLLM, flexible_training_loop_warm_start_accelerate, SelectTokenDownsampler, ExactRandomGater, select_next_token_cross_entropy, LinearGater
from clean_code.conditional_sequential import SequentiallyDependentLinearGater

import os

import argparse

import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer
import datasets
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed

def main():
    # Copied from training_random_base_model/run_distributed_large.py:

    ############################################################################################################################################################################################

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Random base model training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    username = "sdauncey"
    scratch_dir = f"/scratch/{username}/tokenizer_training"

    accelerator =  Accelerator(log_with="wandb")

    training_loop_kwargs = {
        "num_epochs": 1, 
        "max_seq_length": 4096,
        "batch_print_every": 100, 
        "batch_limit": 5*10**3,
        "warm_start_steps": None,
        "learn_gating": True,
        "discount_rate": 0.9,
        "early_output_loss_weight": 0.2,
        "relative_gating_loss_weight": 0.5*(1 - 0.9),
        "consistency_loss_weight": 2.,
        "early_exit_advantage_estimate": True
    }

    # According to Yair Carmon/Deepseek, we should use a learning rate of 1.5e-3 and a warmup of ~1.2B bytes for a ~300M parameter model. 
    # Adjusting this down as the model requires a lower batch size to avoid OOMing. 
    optimization_kwargs = {
        "learning_rate": 1e-3,
        "warmup_steps": 1.2 * 10**9 / (accelerator.num_processes * 16 * 4096),
        "batch_size": 4, # we need to reduce the batch size as initially the model will OOM due to a high downsample rate.
    }

    # Quite similar to GPT-2-small. 
    model_kwargs = {
        "vocab_size": 256,
        "embedding_dim": 1024,
        "num_heads": 16,
        "downsample_rate": 0.25,
        "sliding_window": 64,
        "n_down_layers": 3,
        "n_mid_layers": 18,
        "n_up_layers": 3,
        "GaterClass": ExactRandomGater,
        "DownSamplerClass": SelectTokenDownsampler,
    }

    config = {**vars(args), **training_loop_kwargs, **optimization_kwargs, **model_kwargs}
    # Convert class objects to string representations for wandb config serialization
    config_for_wandb = config.copy()
    config_for_wandb["GaterClass"] = SequentiallyDependentLinearGater.__name__
    config_for_wandb["DownSamplerClass"] = config["DownSamplerClass"].__name__

    # For some reason, you need to pass the config to the init_kwargs when using wandb with accelerate in offline mode. https://github.com/huggingface/accelerate/issues/3607
    accelerator.init_trackers("training_random_base_model", config=config_for_wandb,  init_kwargs={"wandb": {"config": config_for_wandb}})


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

    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=optimization_kwargs["warmup_steps"])

    train_dataloader = DataLoader(
        openwebtext_25p,
        batch_size=optimization_kwargs["batch_size"],
        num_workers=32,
        pin_memory=True
    )

    ############################################################################################################################################################################################
    
    net_scratch_dir = os.path.join("/itet-stor/sdauncey/net_scratch/VScodeProjects/bitter-lesson-tokenization")
    checkpoint_dir = os.path.join(net_scratch_dir, "training_random_base_model", "medium_checkpoints")

    model, optimizer, scheduler, train_dataloader = accelerator.prepare(
        model, optimizer, scheduler, train_dataloader
    )
    accelerator.load_state(checkpoint_dir)
    
    # To add a new layer, we unwrap the model, add the new layer to the model and optimizer, and then wrap it again. This ensures synchronization across all devices (see commented code below).
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.down_layer_gate = SequentiallyDependentLinearGater(embedding_dim=1024, downsample_rate=0.25).to(accelerator.device, dtype=torch.bfloat16)
    optimizer.add_param_group({'params': unwrapped_model.down_layer_gate.parameters()})

    model = accelerator.prepare(unwrapped_model)

    # print(f"{accelerator.device}: {unwrapped_model.down_layer_gate.filter_layer.weight[:3,:3]=}")

    train_dataloader = accelerator.skip_first_batches(train_dataloader, 4*10**4)

    flexible_training_loop_warm_start_accelerate(
        model, 
        optimizer, 
        scheduler, 
        train_dataloader, 
        accelerator, 
        tokenizer=byte5_tokenizer,
        **training_loop_kwargs
    )

    accelerator.wait_for_everyone()


    accelerator.end_training()
    
    net_scratch_dir = os.path.join("/itet-stor/sdauncey/net_scratch/VScodeProjects/bitter-lesson-tokenization")
    save_dir = os.path.join(net_scratch_dir, "experiment_33", "medium_checkpoints_test_2")

    if accelerator.is_main_process: 
        print(f"Saving checkpoint to {save_dir}")

    accelerator.save_state(save_dir)

    # TODO: unwrap the model and save the final model as a .pt file.



if __name__ == "__main__":
    main()