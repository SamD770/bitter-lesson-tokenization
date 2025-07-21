"""
YOLO run of using normalisation by the early exit. of test of sequentially dependent gater with a linear filter prediction.
"""
from clean_code.bitter_llm import set_seed, parameter_count_string
from clean_code.bitter_llm import RandomGater, set_seed, parameter_count_string
from clean_code.flexible_bitter_llm import FlexibleBitterLLM, flexible_training_loop_warm_start, SelectTokenDownsampler, IndependentWrapperGater
from clean_code.conditional_sequential import SequentiallyDependentLinearGater
import datasets
from transformers import AutoTokenizer
import os
import torch
import argparse

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Experiment 31 for bitter tokenization")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    
    byte5_tokenizer = AutoTokenizer.from_pretrained("google/byt5-large")

    username = "sdauncey"
    scratch_dir = f"/scratch/{username}/tokenizer_training"

    if not os.path.exists(scratch_dir):
        os.makedirs(scratch_dir)

    # Download a portion of OpenWebText dataset
    # This will download a subset of the OpenWebText corpus
    print("Downloading OpenWebText dataset...")

    # Load a small portion of OpenWebText (25% of the dataset)
    openwebtext_25p = datasets.load_dataset(
        "openwebtext",
        split="train[:25%]",  # Using only 25% samples of the dataset for now.
        cache_dir=os.path.join(scratch_dir, "openwebtext_25p_cache"),
        trust_remote_code=True
    )

    print(f"Downloaded {len(openwebtext_25p)} examples from OpenWebText")

    set_seed(args.seed)
    print(f"Using random seed: {args.seed}")

    net_scratch_dir = os.path.join("/itet-stor/sdauncey/net_scratch/VScodeProjects/bitter-lesson-tokenization")

    saved_model_file_name = f"training_random_base_model/random_select_early_output_base_model_42.pt"
    model = torch.load(os.path.join(net_scratch_dir, saved_model_file_name), weights_only=False)
    model.down_layer_gate = SequentiallyDependentLinearGater(embedding_dim=768, downsample_rate=0.20)   

    model.to(device="cuda")

    # Check model architecture
    print(model)
    model.attn_implementation = "eager" # Need to debug the flash attention implementation. for variable length sequences (specifically: the mid layers)

    print(f"my_model has {parameter_count_string(model)} parameters")

    training_loop_kwargs = {
        "num_epochs": 1,
        "batch_size": 32,
        "max_seq_length": 1024,
        "batch_print_every": 50,
        "batch_limit": 5*10**3,
        "warm_start_steps": 2*10**3,
        "learn_gating": True,
        "discount_rate": 0.9,
        "early_output_loss_weight": 0.2,
        "relative_gating_loss_weight": 0.01,
        "consistency_loss_weight": 2.,
        "early_exit_advantage_estimate": True
    }

    print("\nTraining Loop Configuration:")
    for key, value in training_loop_kwargs.items():
        print(f"  {key:.<30} {value}")
    print()

    train_losses = flexible_training_loop_warm_start(
        model, 
        openwebtext_25p, 
        tokenizer=byte5_tokenizer,
        **training_loop_kwargs
    )

    # Save the train losses to the specified directory
    train_losses_file = os.path.join(net_scratch_dir, f"experiment_31/train_losses_{args.seed}.csv")
    train_losses.to_csv(train_losses_file, index=False)
    print(f"Train losses saved to {train_losses_file}")


    model_file_name = f"experiment_31/model_{args.seed}.pt"

    # Save the model to the specified directory
    os.makedirs(net_scratch_dir, exist_ok=True)
    model_save_file = os.path.join(net_scratch_dir, model_file_name)
    torch.save(model, model_save_file)
    print(f"Model saved to {model_save_file}")

